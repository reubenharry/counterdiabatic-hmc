import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from src.utils import check_nans, compute_energy_stats, compute_expectation_over_equilibrium, compute_expectation_over_particles, generate_initial_samples, normalize_log_weights, systematic_resample

from .physics import make_cd_leapfrog_step, make_cd_implicit_midpoint_step, make_leapfrog_step, make_cd_euler_step, partially_refresh_momentum, with_maruyama
from .fitting import fit_gauge_potential, calculate_gauge_potential_loss
from .ansatze import AnalyticAnsatz, PolynomialAnsatz, NeuralNetworkAnsatz, HermiteAnsatz

def initialize(M, make_T, make_V, key, dim, lam_fn, initial_sigma=1.0):
    snapshots = {'particles': [], 'weights': [], 'lam': [], 'resampling_events': [], 'times': []}
    snapshots['pre_equilibration'] = []
    resampling_count = 0
    loss_histories = []
    param_history = []
    prev_H_vals = None
    detailed_energy_stats = []
    detailed_times = []
    t_k = 0.0
    log_weights = jnp.zeros(M)
    initial_lam = float(lam_fn(0.0))
    q, p = generate_initial_samples(M, make_T, make_V, initial_lam, key, dim, initial_sigma)
    # Check initial samples
    check_nans(f"initial_q", q)
    check_nans(f"initial_p", p)
    return snapshots, resampling_count, loss_histories, param_history, prev_H_vals, detailed_energy_stats, detailed_times, t_k, log_weights, q, p

# =============================================================================
# UNIFIED SIMULATION FUNCTION
# =============================================================================
def smc(
    M, 
    N_steps, 
    momentum_refresh_interval, 
    make_T, 
    make_V, 
    lam_fn, 
    dot_lam_fn, 
    key, 
    dim, 
    final_time,
    A_ansatz=None, 
    fit_every=1, 
    num_iters=10000,
    learning_rate=1e-4, 
    use_weights=False, 
    ess_threshold=None, # 0.5 
    snapshot_every=1,
    adaptive_step_size=False, 
    integrator_type="leapfrog", 
    equilibration_steps=0,
    initial_sigma=1.0,
    resample_fn=systematic_resample,
    next_time=lambda t, k: t + 0.2,
    ):
    """
    Unified simulation function that handles both naive HMC and counterdiabatic HMC.
    
    Args:
        M: Number of particles
        N_steps: Number of simulation steps
        eps: HMC step size
        momentum_refresh_interval: Momentum refresh interval
        make_T: Function to create kinetic energy
        make_V: Function to create potential energy
        lam_fn: Lambda function
        dot_lam_fn: Derivative of lambda function
        key: Random key
        dim: Dimension
        A_ansatz: Ansatz for counterdiabatic simulation (required for 'cd' type)
        fit_every: How often to fit the ansatz (for 'cd' type)
        num_initial_iterations: Initial optimization iterations (for 'cd' type)
        num_iterations: Optimization iterations per step (for 'cd' type)
        learning_rate: Learning rate for optimization (for 'cd' type)
        use_weights: Whether to use importance weights
        ess_threshold: Effective sample size threshold for resampling
        snapshot_every: Rate at which snapshots are taken
        equilibration_steps: Number of equilibration HMC steps after each CD step
        
    Returns:
        For 'naive' type: snapshots
        For 'cd' type: (A_ansatz, snapshots, loss_histories, param_history)
    """
    
    
    
    
    
    snapshots, resampling_count, loss_histories, param_history, prev_H_vals, detailed_energy_stats, detailed_times, t_k, log_weights, q, p = initialize(M, make_T, make_V, key, dim, lam_fn, initial_sigma)
    
    for k in range(N_steps):

    
           
        lam_k = float(lam_fn(t_k))
        dot_lam_k = float(dot_lam_fn(t_k))
        
        # Check lambda values for NaNs
        if jnp.isnan(lam_k) or jnp.isnan(dot_lam_k): raise Exception()
        
            
        # Handle ansatz fitting and loss calculation
        if A_ansatz is not None and (k % fit_every == 0):
            print(f"Fitting ansatz at step {k} with λ = {lam_k}")

            samples = np.concatenate([np.array(q), np.array(p)], axis=1)
            check_nans("fitting_samples", samples, k)
            weights = jnp.exp(normalize_log_weights(log_weights))
            
            A_ansatz, loss_history = fit_gauge_potential(
                lam_k, samples,
                make_T=make_T, make_V=make_V, A_ansatz=A_ansatz,  # Pass current ansatz for warm start
                num_iters=num_iters, lr=learning_rate,
                use_regularization=False, weights=weights)
            print(f"  Final loss: {loss_history[-1]:.6f}")
            if any(jnp.isnan(loss) for loss in loss_history):
                print(f"⚠️  NaN detected in loss history at step {k}")
            
            loss_histories.append(loss_history)
        
        # Compute energy statistics at every timestep
        stats = compute_energy_stats(q, p, lam_k, make_T, make_V, A_ansatz, log_weights)
        
        # Compute energy changes if we have previous values
        if k % snapshot_every == 0:
            if prev_H_vals is not None:
                # Compute individual particle energy changes
                delta_H_vals = stats['H_vals'] - prev_H_vals
                
                # Compute expectations of energy changes
                E_p_delta_H = compute_expectation_over_particles(delta_H_vals)
                E_p_delta_H_sq = compute_expectation_over_particles(delta_H_vals ** 2)
                
                # Add change statistics to the stats dictionaries
                stats['E_p_delta_H'] = float(E_p_delta_H)
                stats['E_p_delta_H_sq'] = float(E_p_delta_H_sq)
            else:
                # First timestep - no change to compute
                stats['E_p_delta_H'] = 0.0
                stats['E_p_delta_H_sq'] = 0.0

            detailed_energy_stats.append(stats)
            detailed_times.append(t_k)

            snapshots['particles'].append(np.array(q))
            snapshots['weights'].append(np.array(log_weights))
            snapshots['lam'].append(lam_k)
            snapshots['times'].append(t_k)

            if A_ansatz is not None and isinstance(A_ansatz, PolynomialAnsatz):
                param_history.append(np.array(A_ansatz.params))
                check_nans("polynomial_params", A_ansatz.params, k)
            elif A_ansatz is not None and isinstance(A_ansatz, NeuralNetworkAnsatz):
                # Store just the parameters as a tuple of arrays
                params = []
                for layer in A_ansatz.layers:
                    if isinstance(layer, eqx.nn.Linear):
                        params.append(np.array(layer.weight))
                        params.append(np.array(layer.bias))
                param_history.append(tuple(params))
            elif A_ansatz is not None and isinstance(A_ansatz, AnalyticAnsatz):
                param_history.append(np.array(A_ansatz.params))
                check_nans("analytic_params", A_ansatz.params, k)
            elif A_ansatz is not None and isinstance(A_ansatz, HermiteAnsatz):
                # For HermiteAnsatz, save the parameters as a dictionary
                param_dict = A_ansatz.params
                param_history.append(param_dict)
                # Check for NaNs in both f_params and g_params
                if param_dict['f_params'] is not None:
                    check_nans("hermite_f_params", param_dict['f_params'], k)
                check_nans("hermite_g_params", param_dict['g_params'], k)

            snapshots['detailed_energy_stats'] = detailed_energy_stats
            snapshots['detailed_times'] = detailed_times
            
            
        
        # Update previous energy values for next iteration
        prev_H_vals = stats['H_vals']
        
       
            
        
        # Record parameters for counterdiabatic simulations (do this before potential equilibration)

        t_k1 = next_time(t_k, k)
        current_delta_t = t_k1 - t_k
            
        
        # Use adaptive step size for time progression if enabled
        # step_delta_t = current_delta_t if simulation_type == 'cd' and adaptive_step_size else delta_t
        lam_k1 = float(lam_fn(t_k1))
        dot_lam_k1 = float(dot_lam_fn(t_k1))
        
  
        if A_ansatz is None:
            step = jax.vmap(lambda q, p, lam, lam_next, dot_lam, dot_lam_next, delta_t: (make_leapfrog_step(make_T(lam), make_V(lam), make_T(lam_next), make_V(lam_next), lam_fn, dot_lam_fn))(q, p, delta_t), in_axes=(0, 0, None, None, None, None, None))
        # TODO vvv
        # else:  # cd
             # Compute adaptive step size for CD if enabled (at beginning of step)
            #  current_delta_t = delta_t
            # if adaptive_step_size:
            #     var_A = stats['var_A']
            #     if var_A > 0:
            #         print("\n\nvar_A", var_A)
            #         current_delta_t = K / jnp.sqrt(var_A).item()
            #         # Bound the step size between 0.05 and 1.0 for stability
            #         # current_delta_t = max(0.05, min(1.0, current_delta_t))
            #         if k % 10 == 0:  # Print every 10 steps to avoid spam
            #             print(f"  Step {k}: Var[A] = {var_A:.6f}, adaptive delta_t = {current_delta_t:.6f}")
            #     else:
            #         if k % 10 == 0:
            #             print(f"  Step {k}: Var[A] = {var_A:.6f}, using fixed delta_t = {current_delta_t:.6f}")
             
            # Choose integrator based on integrator_type
        elif integrator_type == "leapfrog":
                step = jax.vmap(lambda q, p, lam, lam_next, dot_lam, dot_lam_next, delta_t: (make_cd_leapfrog_step(make_T, make_V, A_ansatz, lam, lam_next, dot_lam, dot_lam_next, lam_fn, dot_lam_fn))(q=q, p=p, eps=delta_t), in_axes=(0, 0, None, None, None, None, None))
        elif integrator_type == "implicit_midpoint":
                step = jax.vmap(lambda q, p, lam, lam_next, dot_lam, dot_lam_next, delta_t: (make_cd_implicit_midpoint_step(make_T, make_V, A_ansatz, lam, lam_next, dot_lam, dot_lam_next))(q=q, p=p, eps=delta_t), in_axes=(0, 0, None, None, None, None, None))
        else:
                raise ValueError(f"Unknown integrator type: {integrator_type}. Must be 'leapfrog' or 'implicit_midpoint'")



        while True:
        
            q_new, p_new, step_weights = jax.jit(step)(q, p, lam_k, lam_k1, dot_lam_k, dot_lam_k1, current_delta_t)
            if False:
                print(f"  Step {k}: Acceptance rate too low, decreasing delta_t to {current_delta_t * 0.5:.6f}")
                current_delta_t = current_delta_t * 0.9
            else:
                q,p = q_new, p_new
                break

        print("next iter")


        log_weights += step_weights
        weights = jnp.exp(normalize_log_weights(log_weights))
        ess = jnp.sum(weights)**2 / jnp.sum(weights ** 2)


        

        if use_weights: 
             
            # Resample if ESS falls below threshold
            # if ess_threshold is not None: 
            if True:
            # and ess < ess_threshold:
                 print(f"  Resampling at step {k} (ESS = {ess:.2f})")
                 q, p, log_weights = resample_fn(q, p, log_weights, key, M)
                 resampling_count += 1
                 # Record the resampling event time
                 snapshots['resampling_events'].append(t_k)

        # refresh momentum
        if k % momentum_refresh_interval == 0 and k > 0:
        # if True:
            # old_energy = jax.vmap(lambda qr, pr: make_T(lam_k)(pr) + make_V(lam_k)(qr))(q, p)
            # first resample
            
            key, sub = jax.random.split(key)
            q, p, log_weights = resample_fn(q, p, log_weights, sub, M)
            key, sub = jax.random.split(key)
            p = jax.random.normal(sub, (M, dim))
            # new_energy = jax.vmap(lambda qr, pr: make_T(lam_k)(pr) + make_V(lam_k)(qr))(q, p)
            # log_weights += old_energy - new_energy

            T = make_T(lam_k)
            V = make_V(lam_k)
            prev_H_vals = jax.vmap(lambda qr, pr: T(pr) + V(qr))(q, p)

             
           
         # Check for NaNs after step
        if check_nans(f"q", q, k) or check_nans(f"p", p, k) or check_nans(f"log_weights", log_weights, k): raise Exception(f"  Stopping simulation due to NaNs in HMC at step {k}")
         

        t_k = next_time(t_k, k)
        print(f"t_k: {t_k}")
        
        # t_k += current_delta_t
        if t_k > final_time:
            break

    return A_ansatz, snapshots, loss_histories, param_history



 # # 1. Run equilibration at current time (if enabled and taking snapshots)
        # if equilibration_steps > 0 and k % snapshot_every == 0 and k < N_steps:
        #     # Store pre-equilibration particles
        #     pre_equil_q = np.array(q)
        #     if 'pre_equilibration' in snapshots:
        #         snapshots['pre_equilibration'].append(pre_equil_q)
            
        #     # Run equilibration steps using standard HMC at current lambda
        #     # T_current = make_T(lam_k)
        #     # V_current = make_V(lam_k)
            
        #     # Create HMC step function for equilibration
        #     hmc_step = jax.vmap(lambda q, p, lam, lam_next, delta_t: (make_leapfrog_step(make_T(lam), make_V(lam), make_T(lam_next), make_V(lam_next), lam_fn, dot_lam_fn))(q, p, delta_t), in_axes=(0, 0, None, None, None))
            
        #     # Run equilibration steps
        #     print("\n\nEquilibrating at time", t_k, "with lambda", lam_k)
        #     key, equil_key, refresh_key = jax.random.split(key, 3)
        #     p = jax.random.normal(refresh_key, (M, dim))
            
        #     for equil_iter in range(equilibration_steps):
        #         # refresh_key, update_key = jax.random.split(equil_key)
        #         # Refresh momentum before each equilibration step
                
        #         # Take HMC step
        #         # subs = jax.random.split(update_key, M)
        #         # print("\n\nlam_k\n\n", lam_k)
        #         if k%5 == 0:
        #             q, p, _ = jax.jit(hmc_step)(q, p, lam_k, lam_k, 1e-2)
        #         if equil_iter%10 == 0:
        #             equil_key, refresh_key = jax.random.split(equil_key)
        #             p = jax.random.normal(refresh_key, (M, dim))
            
        #     # Store post-equilibration particles
        #     post_equil_q = np.array(q)
        #     if 'post_equilibration' in snapshots:
        #         snapshots['post_equilibration'].append(post_equil_q)
        
        # 2. Save snapshot (after potential equilibration) - before evolution steps