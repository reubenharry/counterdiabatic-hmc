import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from src.utils import check_nans, compute_energy_stats, compute_expectation_over_equilibrium, compute_expectation_over_particles, generate_initial_samples

from .physics import make_cd_leapfrog_step, make_cd_implicit_midpoint_step, make_leapfrog_step, make_cd_euler_step, partially_refresh_momentum, with_maruyama
from .fitting import fit_gauge_potential, calculate_gauge_potential_loss
from .ansatze import AnalyticAnsatz, PolynomialAnsatz, NeuralNetworkAnsatz

# =============================================================================
# UNIFIED SIMULATION FUNCTION
# =============================================================================
def simulate(
    M, 
    N_steps, 
    delta_t, 
    momentum_refresh_interval, 
    make_T, 
    make_V, 
    lam_fn, 
    dot_lam_fn, 
    key, 
    dim, 
    A_ansatz=None, 
    fit_every=1, 
    num_iters=10000,
    learning_rate=1e-4, 
    use_weights=False, 
    ess_threshold=0.5, 
    snapshot_every=1,
    adaptive_step_size=False, 
    K=0.2, 
    integrator_type="leapfrog", 
    equilibration_steps=0
    ):
    """
    Unified simulation function that handles both naive HMC and counterdiabatic HMC.
    
    Args:
        M: Number of particles
        N_steps: Number of simulation steps
        delta_t: Time step
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
    
    # Generate initial samples from the correct distribution
    initial_lam = float(lam_fn(0.0))
    print(f"Generating initial samples with λ = {initial_lam}")
    q, p = generate_initial_samples(M, make_T, make_V, initial_lam, key, dim)
    
    # Check initial samples
    check_nans(f"initial_q", q)
    check_nans(f"initial_p", p)
    
    # Initialize weights for SMC (Sequential Monte Carlo)
    log_weights = jnp.zeros(M)
    
    # Initialize snapshots and tracking variables
    snapshots = {'particles': [], 'weights': [], 'lam': [], 'resampling_events': [], 'times': []}
    
    # Add equilibration tracking for CD simulations
    if A_ansatz is not None and equilibration_steps > 0:
        snapshots['pre_equilibration'] = []
        snapshots['post_equilibration'] = []
    resampling_count = 0
    loss_histories = []
    param_history = []
    first_fit = A_ansatz is not None
    
    # Store previous energy values for computing changes
    prev_H_vals = None
    
    # Store detailed energy statistics for plotting
    detailed_energy_stats = []
    detailed_times = []
    
    t_k = 0.0
    current_delta_t = delta_t
    
    for k in range(N_steps + 1):
           
        lam_k = float(lam_fn(t_k))
        dot_lam_k = float(dot_lam_fn(t_k))
        
        # Check lambda values for NaNs
        if jnp.isnan(lam_k) or jnp.isnan(dot_lam_k): raise Exception()
        
        if A_ansatz is not None:
            
            # Handle ansatz fitting and loss calculation
            if (k % fit_every == 0) and (k < N_steps):
                print(f"Fitting ansatz at step {k} with λ = {lam_k}")
                
                # For multi-dimensional case, stack q and p along the last axis
                samples = np.concatenate([np.array(q), np.array(p)], axis=1)
                check_nans("fitting_samples", samples, k)

                weights = jnp.exp(log_weights - jnp.max(log_weights))  # Numerical stability
                weights = weights / jnp.sum(weights)  # Normalize
                
                
                    
                
                # For trainable ansatzes, fit and print loss
                print(f"  Fitting trainable ansatz...")
                
               
            
                A_ansatz, loss_history = fit_gauge_potential(lam_k, samples,
                                            make_T=make_T, make_V=make_V,
                                            A_ansatz=A_ansatz,  # Pass current ansatz for warm start
                                            num_iters=num_iters, lr=learning_rate,
                                            use_regularization=False, weights=weights)
                
                # Print final loss
                final_loss = loss_history[-1] if loss_history else float('inf')
                print(f"  Final loss: {final_loss:.6f}")
                
                # Check loss history for NaNs
                if any(jnp.isnan(loss) for loss in loss_history):
                    print(f"⚠️  NaN detected in loss history at step {k}")
                    nan_indices = [i for i, loss in enumerate(loss_history) if jnp.isnan(loss)]
                    print(f"  NaN losses at iterations: {nan_indices}")
                
                loss_histories.append(loss_history)
        
        # Compute energy statistics at every timestep
        stats = compute_energy_stats(q, p, lam_k, make_T, make_V, A_ansatz, log_weights)
        
        # Compute energy changes if we have previous values
        if prev_H_vals is not None:
            # Compute individual particle energy changes
            delta_H_vals = stats['H_vals'] - prev_H_vals
            
            # Average the individual changes
            avg_delta_H = jnp.mean(delta_H_vals)
            
            # Add change statistics to the stats dictionaries
            stats['avg_delta_H'] = float(avg_delta_H)
            stats['avg_delta_H_sq'] = float(jnp.mean(delta_H_vals ** 2))
        else:
            # First timestep - no change to compute
            stats['avg_delta_H'] = 0.0
            stats['avg_delta_H_sq'] = 0.0
        
        
        # Store detailed energy statistics for plotting (only when taking snapshots)
        if k % snapshot_every == 0:
            detailed_energy_stats.append(stats)
            detailed_times.append(t_k)
        
        # Update previous energy values for next iteration
        prev_H_vals = stats['H_vals']
        
        # 1. Run equilibration at current time (if enabled and taking snapshots)
        if equilibration_steps > 0 and k % snapshot_every == 0 and k < N_steps:
            # Store pre-equilibration particles
            pre_equil_q = np.array(q)
            if 'pre_equilibration' in snapshots:
                snapshots['pre_equilibration'].append(pre_equil_q)
            
            # Run equilibration steps using standard HMC at current lambda
            # T_current = make_T(lam_k)
            # V_current = make_V(lam_k)
            
            # Create HMC step function for equilibration
            hmc_step = jax.vmap(lambda q, p, lam, lam_next, delta_t: (make_leapfrog_step(make_T(lam), make_V(lam), make_T(lam_next), make_V(lam_next), lam_fn, dot_lam_fn))(q, p, delta_t), in_axes=(0, 0, None, None, None))
            
            # Run equilibration steps
            print("\n\nEquilibrating at time", t_k, "with lambda", lam_k)
            key, equil_key, refresh_key = jax.random.split(key, 3)
            p = jax.random.normal(refresh_key, (M, dim))
            
            for equil_iter in range(equilibration_steps):
                # refresh_key, update_key = jax.random.split(equil_key)
                # Refresh momentum before each equilibration step
                
                # Take HMC step
                # subs = jax.random.split(update_key, M)
                # print("\n\nlam_k\n\n", lam_k)
                if k%5 == 0:
                    q, p, _ = jax.jit(hmc_step)(q, p, lam_k, lam_k, 1e-2)
                if equil_iter%10 == 0:
                    equil_key, refresh_key = jax.random.split(equil_key)
                    p = jax.random.normal(refresh_key, (M, dim))
            
            # Store post-equilibration particles
            post_equil_q = np.array(q)
            if 'post_equilibration' in snapshots:
                snapshots['post_equilibration'].append(post_equil_q)
        
        # 2. Save snapshot (after potential equilibration) - before evolution steps
        if k % snapshot_every == 0 and k < N_steps:
            snapshots['particles'].append(np.array(q))  # This will be post-equilibration if equilibration was done
            snapshots['weights'].append(np.array(log_weights))
            snapshots['resampling_events'].append(resampling_count)
            snapshots['lam'].append(lam_k)
            snapshots['times'].append(t_k)
            print("\n\n\n\n\n times", t_k, lam_k)
        
        # Record parameters for counterdiabatic simulations (do this before potential equilibration)
        if k % snapshot_every == 0 and A_ansatz is not None:
            if isinstance(A_ansatz, PolynomialAnsatz):
                param_history.append(np.array(A_ansatz.params))
                check_nans("polynomial_params", A_ansatz.params, k)
            elif isinstance(A_ansatz, NeuralNetworkAnsatz):
                # Store just the parameters as a tuple of arrays
                params = []
                for layer in A_ansatz.layers:
                    if isinstance(layer, eqx.nn.Linear):
                        params.append(np.array(layer.weight))
                        params.append(np.array(layer.bias))
                param_history.append(tuple(params))
            elif isinstance(A_ansatz, AnalyticAnsatz):
                param_history.append(np.array(A_ansatz.params))
                check_nans("analytic_params", A_ansatz.params, k)
        
        # Use adaptive step size for time progression if enabled
        # step_delta_t = current_delta_t if simulation_type == 'cd' and adaptive_step_size else delta_t
        lam_k1 = float(lam_fn(t_k + current_delta_t))
        dot_lam_k1 = float(dot_lam_fn(t_k + current_delta_t))
        
        
        # Continue with evolution step
            
        key, sub = jax.random.split(key)
        subs = jax.random.split(sub, M)
         
        # 2. Evolve - Execute the appropriate step based on simulation type
        if A_ansatz is None:
             naive_step = jax.vmap(lambda q, p, lam, lam_next, delta_t, L, rng_key: with_maruyama(make_leapfrog_step(make_T(lam), make_V(lam), make_T(lam_next), make_V(lam_next), lam_fn, dot_lam_fn))(q, p, delta_t, L=L, rng_key=rng_key), in_axes=(0, 0, None, None, None, None, 0))
             
             q, p, step_weights = jax.jit(naive_step)(q, p, lam_k, lam_k1, delta_t, delta_t*momentum_refresh_interval, subs)
             
             
             # Naive snapshots are now recorded in the unified section above

        else:  # cd
             # Compute adaptive step size for CD if enabled (at beginning of step)
            #  current_delta_t = delta_t
            if adaptive_step_size:
                var_A = stats['var_A']
                if var_A > 0:
                    print("\n\nvar_A", var_A)
                    current_delta_t = K / jnp.sqrt(var_A).item()
                    # Bound the step size between 0.05 and 1.0 for stability
                    # current_delta_t = max(0.05, min(1.0, current_delta_t))
                    if k % 10 == 0:  # Print every 10 steps to avoid spam
                        print(f"  Step {k}: Var[A] = {var_A:.6f}, adaptive delta_t = {current_delta_t:.6f}")
                else:
                    if k % 10 == 0:
                        print(f"  Step {k}: Var[A] = {var_A:.6f}, using fixed delta_t = {current_delta_t:.6f}")
             
            # Choose integrator based on integrator_type
            if integrator_type == "leapfrog":
                cd_step = jax.vmap(lambda q, p, lam, lam_next, dot_lam, dot_lam_next, delta_t, t: (make_cd_leapfrog_step(make_T, make_V, A_ansatz, lam, lam_next, dot_lam, dot_lam_next, lam_fn, dot_lam_fn))(q=q, p=p, eps=delta_t, t=t), in_axes=(0, 0, None, None, None, None, None, None))
            elif integrator_type == "implicit_midpoint":
                cd_step = jax.vmap(lambda q, p, lam, lam_next, dot_lam, dot_lam_next, delta_t, t: (make_cd_implicit_midpoint_step(make_T, make_V, A_ansatz, lam, lam_next, dot_lam, dot_lam_next))(q=q, p=p, eps=delta_t, t=t), in_axes=(0, 0, None, None, None, None, None, None))
            else:
                raise ValueError(f"Unknown integrator type: {integrator_type}. Must be 'leapfrog' or 'implicit_midpoint'")
            q, p, step_weights = jax.jit(cd_step)(q, p, lam_k, lam_k1, dot_lam_k, dot_lam_k1, current_delta_t, t_k)

            # Equilibration now happens before evolution, not after

            if k % momentum_refresh_interval == 0 and k > 0:
                key, sub = jax.random.split(key)
                p = jax.random.normal(sub, (M, dim))

        if use_weights: log_weights = log_weights + step_weights
             
           
         # Check for NaNs after step
        if check_nans(f"q", q, k) or check_nans(f"p", p, k) or check_nans(f"log_weights", log_weights, k):
             raise Exception(f"  Stopping simulation due to NaNs in HMC at step {k}")
         
         # Handle weights and resampling
        if use_weights:
            
             # Compute effective sample size
             weights = jnp.exp(log_weights - jnp.max(log_weights))  # Numerical stability
             weights = weights / jnp.sum(weights)  # Normalize
             ess = jnp.sum(weights)**2 / jnp.sum(weights ** 2)
             
             # Resample if ESS is too low
             # if ess < ess_threshold * M:
             # normalize weights
             weights = weights / jnp.sum(weights)
             if False:
                 print(f"  Resampling at step {k} (ESS = {ess:.2f})")
                 q, p, log_weights = multinomial_resample(q, p, log_weights, key, M)
                 resampling_count += 1
         
        t_k += current_delta_t
        
       
    snapshots['detailed_energy_stats'] = detailed_energy_stats
    snapshots['detailed_times'] = detailed_times
    
    print(f"Simulation completed after {k} steps")
    if A_ansatz is None:
        return A_ansatz, snapshots, loss_histories, []
    else:  # cd
        return A_ansatz, snapshots, loss_histories, param_history





# def save_simulation_data(snapshots, system_name, method_name, delta_t, lam_fn, ansatz_params=None, loss_histories=None, param_history=None):
#     """Save simulation data to a pickle file."""
#     import pickle
#     import os
    
#     # Create data directory if it doesn't exist
#     os.makedirs("data", exist_ok=True)
    
#     # Extract lambda values at each snapshot time
#     # times = jnp.arange(len(snapshots.get('particles', []))) * delta_t
#     # lambda_values = [float(lam_fn(t)) for t in times]

    
#     # Prepare data to save
#     data = {
#         'snapshots': snapshots,
#         'system_name': system_name,
#         'method_name': method_name,
#         'delta_t': delta_t,
#         'ansatz_params': ansatz_params,
#         'loss_histories': loss_histories,
#         'param_history': param_history
#     }
    
#     # Save to pickle file
#     filename = f"data/{system_name}_{method_name}.pkl"
#     with open(filename, 'wb') as f:
#         pickle.dump(data, f)
    
#     print(f"Saved simulation data to {filename}")

# def load_simulation_data(system_name, method_name):
#     """Load simulation data from a pickle file."""
#     import pickle
#     import os
    
#     filename = f"data/{system_name}_{method_name}.pkl"
    
#     if not os.path.exists(filename):
#         print(f"Data file {filename} not found.")
#         return None
    
#     with open(filename, 'rb') as f:
#         data = pickle.load(f)
    
#     print(f"Loaded simulation data from {filename}")
#     return data

# def run_simulation_and_save_data(system_name, ansatz, lam_fn, dot_lam_fn, run_simulations=True, snapshot_every=1, 
#                                  M=1000, N_steps=40, delta_t=0.05, 
#                                  momentum_refresh_interval=5.0, fit_every=1, 
#                                  num_initial_iterations=10000, num_iterations=10000, 
#                                  learning_rate=1e-4, equilibration_steps=0, ess_threshold=0.5,
#                                  adaptive_step_size=False, K=0.2, integrator_type="implicit_midpoint",
#                                  ):
#     """
#     Run simulations and save data for the specified system and ansatz.
    
#     Args:
#         system_name: Name of the system ('gaussian_annealing', 'gaussian_moving_mean', etc.)
#         ansatz: The ansatz object (PolynomialAnsatz, NeuralNetworkAnsatz, etc.)
#         run_simulations: Whether to run simulations or load from saved data
#         snapshot_every: Rate at which snapshots are taken
        
#     Returns:
#         dict: Dictionary containing successful simulation results
#     """
#     # Set up the system based on system_name
#     from src.systems import get_system
    
#     make_T, make_V, system_description, dim = get_system(system_name)
    
#     # Storage for all simulation results
#     successful_simulations = {}
    
#     if run_simulations:
#         # Run simulations and save data
#         key = jax.random.PRNGKey(0)
        
#         # Define simulation configurations
#         all_configs = {
#             'naive_unweighted': {'name': 'naive_unweighted', 'use_weights': False, 'ess_threshold': None},
#             'naive_weighted': {'name': 'naive_weighted', 'use_weights': True, 'ess_threshold': ess_threshold},
#             'cd_unweighted': {'name': f'cd_unweighted_{integrator_type}', 'use_weights': False, 'ess_threshold': None},
#             'cd_weighted': {'name': f'cd_weighted_{integrator_type}', 'use_weights': True, 'ess_threshold': ess_threshold}
#         }
        
#         configs_to_run = list(all_configs.values())
        
#         # Run simulations using the unified simulate function
#         for config in configs_to_run:
#             print(f"\n{'='*50}")
#             print(f"Running {config['name'].replace('_', ' ').title()}")
#             print(f"{'='*50}")
            
#             # try:
#             # Determine simulation type
            
#                                 # Prepare parameters
#             kwargs = {
#                     'M': M, 'N_steps': N_steps, 'delta_t': delta_t,
#                     'momentum_refresh_interval': momentum_refresh_interval,
#                     'make_T': make_T, 'make_V': make_V, 'lam_fn': lam_fn, 'dot_lam_fn': dot_lam_fn,
#                     'key': key, 'dim': dim, 'use_weights': config['use_weights'], 
#                     'snapshot_every': snapshot_every, 'adaptive_step_size': adaptive_step_size, 'K': K
#                 }
            
#             # Add counterdiabatic-specific parameters
#             if A_ansatz is not None:
#                 kwargs.update({
#                     'A_ansatz': ansatz,
#                     'fit_every': fit_every,
#                     'num_initial_iterations': num_initial_iterations,
#                     'num_iterations': num_iterations,
#                     'learning_rate': learning_rate,
#                     'integrator_type': integrator_type,
#                     'equilibration_steps': equilibration_steps
#                 })
            
#             if config['ess_threshold'] is not None:
#                 kwargs['ess_threshold'] = config['ess_threshold']
            
#             # Run simulation
#             result = simulate(**kwargs)
            
#             # Handle different return types
#             if A_ansatz is None:
#                 snapshots = result
#                 loss_histories = []
#                 param_history = []
#             else:  # cd
#                 A_ansatz, snapshots, loss_histories, param_history = result
            
#             successful_simulations[config['name']] = snapshots
#             # Add loss histories and parameter history
#             successful_simulations[f'loss_histories_{config["name"]}'] = loss_histories
#             successful_simulations[f'param_history_{config["name"]}'] = param_history
            
#             # Save data
#             save_simulation_data(snapshots, system_name, config['name'], delta_t, lam_fn, 
#                                 ansatz_params=A_ansatz, 
#                                 loss_histories=loss_histories, 
#                                 param_history=param_history)
#             print(f"✓ {config['name'].replace('_', ' ').title()} completed successfully")
                
#             # except Exception as e:
#             #     print(f"✗ {config['name'].replace('_', ' ').title()} failed: {e}")
#     else:
#         # Load data from saved files
#         print("Loading simulation data from saved files...")
        
#         # Try to load each method's data
#         methods = ['naive_unweighted', 'naive_weighted', 'cd_unweighted', 'cd_weighted']
#         for method in methods:
#             data = load_simulation_data(system_name, method)
#             if data is not None:
#                 successful_simulations[method] = data['snapshots']
#                 # Also include the saved times and lambda_values
#                 if 'times' in data:
#                     successful_simulations[f'times_{method}'] = data['times']
#                 if 'lambda_values' in data:
#                     successful_simulations[f'lambda_values_{method}'] = data['lambda_values']
#                 if 'loss_histories' in data and data['loss_histories'] is not None:
#                     successful_simulations[f'loss_histories_{method}'] = data['loss_histories']
#                 if 'param_history' in data and data['param_history'] is not None:
#                     successful_simulations[f'param_history_{method}'] = data['param_history']
#                 print(f"✓ Loaded {method} data")
#             else:
#                 print(f"✗ Could not load {method} data")
    
#     return successful_simulations 