import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from .physics import make_leapfrog_step, make_cd_leapfrog_step, partially_refresh_momentum, with_maruyama
from .fitting import fit_gauge_potential
from .ansatze import AnalyticAnsatz, PolynomialAnsatz, NeuralNetworkAnsatz

def check_nans(name, value, step=None):
    """Helper function to check for NaNs and print warnings."""
    # Convert to numpy for checking to avoid JAX tracing issues
    if hasattr(value, 'numpy'):
        value_np = value.numpy()
    else:
        value_np = value
    
    if jnp.isnan(value_np).any():
        count = jnp.isnan(value_np).sum()
        step_info = f" at step {step}" if step is not None else ""
        print(f"⚠️  NaN detected in {name}{step_info} (count: {count})")
        return True
    return False

# =============================================================================
# 6) SImuLATION: NAÏVE HMC VS CD WITH ONLINE FITTING
# =============================================================================
def generate_initial_samples(M, make_T, make_V, lam, key, dim, num_steps=1000, eps=0.05, L=4.0):
    """Generate M samples from the distribution at the given temperature using HMC.
    
    Args:
        M: Number of samples
        make_T: Function to create kinetic energy
        make_V: Function to create potential energy
        lam: Current lambda value
        key: JAX random key
        dim: Dimension of the system
        num_steps: Number of HMC steps for equilibration
        eps: Step size for HMC
    """
    # Start from random positions
    key, sub = jax.random.split(key)
    q = jax.random.normal(sub, (M, dim))
    key, sub = jax.random.split(key)
    p = jax.random.normal(sub, (M, dim)) 
    

    # Check initial samples for NaNs
    check_nans("initial_q", q)
    check_nans("initial_p", p)

    # Run HMC for a while to get to equilibrium
    T = make_T(lam)
    V = make_V(lam)
    step = jax.jit(with_maruyama(make_leapfrog_step(T, V)))
    
    # qs = []
    # ps = []
    for step_idx in range(num_steps):
        key, sub = jax.random.split(key)
        subs = jax.random.split(sub, M)
        q, p = jax.vmap(lambda q, p, k: step(q, p, eps, L=L, rng_key=k))(q, p, subs)
        
        # Check for NaNs during equilibration
        if check_nans("equilibration_q", q, step_idx):
            print(f"  Stopping equilibration early due to NaNs at step {step_idx}")
            break
        if check_nans("equilibration_p", p, step_idx):
            print(f"  Stopping equilibration early due to NaNs at step {step_idx}")
            break

        # qs.append(np.array(q))
        # ps.append(np.array(p))
        # Randomize momenta periodically
        # if step_idx % 20 == 0:
        #     key, sub = jax.random.split(key)
        #     p = jax.random.normal(sub, (M, dim))
    
    # return np.array(qs), np.array(ps)
    return q, p

def run_simulation(M, N_steps, delta_t, eps, momentum_refresh_interval, fit_every, num_initial_iterations, num_iterations, make_T, make_V, A_ansatz, lam_fn, dot_lam_fn, key, dim, learning_rate=1e-4, re_equil_steps=0):
    # Generate initial samples from the correct distribution
    initial_lam = float(lam_fn(0.0))
    print(f"Generating initial samples with λ = {initial_lam}")
    q_naive, p_naive = generate_initial_samples(M, make_T, make_V, initial_lam, key, dim)
    q_cd = (q_naive).copy()
    p_cd = p_naive.copy()

    # Check initial samples
    check_nans("initial_q_naive", q_naive)
    check_nans("initial_p_naive", p_naive)
    check_nans("initial_q_cd", q_cd)
    check_nans("initial_p_cd", p_cd)

    loss_histories = []
    snapshots = {'naive': [], 'cd_pre_equil': [], 'cd_post_equil': [], 'lam_pre_equil': [], 'lam_post_equil': [], 'energy_stats': []}
    param_history = []  # Track parameter history
    first_fit = True

    # Function to compute energy statistics
    def compute_energy_stats(q, p, lam):
        """Compute average H, H², ∂H/∂λ, and (∂H/∂λ)² over particles."""
        T = make_T(lam)
        V = make_V(lam)
        
        # Compute H for each particle
        H_vals = jax.vmap(lambda qr, pr: T(pr) + V(qr))(q, p)
        
        # Compute ∂H/∂λ for each particle
        # def dH_dlam(qr, pr, lam_val):
        #     # ∂H/∂λ = ∂T/∂λ + ∂V/∂λ
        #     # For standard kinetic energy, ∂T/∂λ = 0
        #     # So ∂H/∂λ = ∂V/∂λ
        #     return jax.grad(lambda l: make_V(l)(qr))(lam_val)

        dH_dlam = lambda q, p: (jax.grad(lambda q, p, lam: make_V(lam)(q), argnums=2)(q, p, lam))
        
        dH_dlam_vals = jax.vmap(lambda qr, pr: dH_dlam(qr, pr))(q, p)
        
        # Compute averages
        avg_H = jnp.mean(H_vals)
        avg_H_sq = jnp.mean(H_vals ** 2)
        avg_dH_dlam = jnp.mean(dH_dlam_vals)
        avg_dH_dlam_sq = jnp.mean(dH_dlam_vals ** 2)
        
        return {
            'avg_H': float(avg_H),
            'avg_H_sq': float(avg_H_sq),
            'avg_dH_dlam': float(avg_dH_dlam),
            'avg_dH_dlam_sq': float(avg_dH_dlam_sq),
            'H_vals': H_vals  # Store individual H values
        }

    # Store previous energy values for computing changes
    prev_naive_H_vals = None
    prev_cd_H_vals = None
    
    # Arrays to store energy statistics at every timestep
    all_energy_stats = []
    all_times = []

    for k in range(N_steps + 1):
        t_k = k * delta_t
        lam_k = float(lam_fn(t_k))
        dot_lam_k = float(dot_lam_fn(t_k))

        # Check lambda values for NaNs
        if jnp.isnan(lam_k):
            print(f"⚠️  NaN detected in lam_k at step {k}")
            break
        if jnp.isnan(dot_lam_k):
            print(f"⚠️  NaN detected in dot_lam_k at step {k}")
            break

        # Update lambda parameter for analytic ansatz
        if isinstance(A_ansatz, AnalyticAnsatz):
            A_ansatz = eqx.tree_at(lambda m: m.params, A_ansatz, jnp.array([lam_k]))

        # Re-fit A every fit_every steps
        if not isinstance(A_ansatz, AnalyticAnsatz) and (k % fit_every == 0) and (k < N_steps):
            print(f"Fitting ansatz at step {k} with λ = {lam_k}")
            
            # Check samples before fitting
            check_nans("fitting_samples_q", q_cd, k)
            check_nans("fitting_samples_p", p_cd, k)
            
            # For multi-dimensional case, stack q and p along the last axis
            samples = np.concatenate([np.array(q_cd), np.array(p_cd)], axis=1)
            
            # Use different number of iterations for first vs subsequent fittings
            if first_fit:
                num_iters = num_initial_iterations
                first_fit = False
            else:
                num_iters = num_iterations
                
            # Warm start: use current ansatz parameters as initialization
            A_ansatz, loss_history = fit_gauge_potential(lam_k, samples,
                                        make_T=make_T, make_V=make_V,
                                        A_ansatz=A_ansatz,  # Pass current ansatz for warm start
                                        num_iters=num_iters, lr=learning_rate,
                                        use_regularization=False)
            
            # Check loss history for NaNs
            if any(jnp.isnan(loss) for loss in loss_history):
                print(f"⚠️  NaN detected in loss history at step {k}")
                nan_indices = [i for i, loss in enumerate(loss_history) if jnp.isnan(loss)]
                print(f"  NaN losses at iterations: {nan_indices}")
            
            loss_histories.append(loss_history)

        # Compute energy statistics at every timestep
        naive_stats = compute_energy_stats(q_naive, p_naive, lam_k)
        cd_stats = compute_energy_stats(q_cd, p_cd, lam_k)
        
        # Compute energy changes if we have previous values
        if prev_naive_H_vals is not None and prev_cd_H_vals is not None:
            # Compute individual particle energy changes
            naive_delta_H_vals = naive_stats['H_vals'] - prev_naive_H_vals
            cd_delta_H_vals = cd_stats['H_vals'] - prev_cd_H_vals
            
            # Average the individual changes
            naive_avg_delta_H = jnp.mean(naive_delta_H_vals)
            cd_avg_delta_H = jnp.mean(cd_delta_H_vals)
            
            # Add change statistics to the stats dictionaries
            naive_stats['avg_delta_H'] = float(naive_avg_delta_H)
            naive_stats['avg_delta_H_sq'] = float(jnp.mean(naive_delta_H_vals ** 2))
            cd_stats['avg_delta_H'] = float(cd_avg_delta_H)
            cd_stats['avg_delta_H_sq'] = float(jnp.mean(cd_delta_H_vals ** 2))
        else:
            # First timestep - no change to compute
            naive_stats['avg_delta_H'] = 0.0
            naive_stats['avg_delta_H_sq'] = 0.0
            cd_stats['avg_delta_H'] = 0.0
            cd_stats['avg_delta_H_sq'] = 0.0
        
        # Store energy statistics for this timestep
        all_energy_stats.append({
            'naive': naive_stats,
            'cd': cd_stats
        })
        all_times.append(t_k)
        
        # Update previous energy values for next iteration
        prev_naive_H_vals = naive_stats['H_vals']
        prev_cd_H_vals = cd_stats['H_vals']

        # Record snapshots and parameters every 10 steps
        if k % 10 == 0:
            snapshots['naive'].append(np.array(q_naive))
            snapshots['cd_pre_equil'].append(np.array(q_cd)) # Store pre-equilibration state
            snapshots['lam_pre_equil'].append(lam_k) # Store lambda at pre-equilibration
            
            # Record parameters
            if isinstance(A_ansatz, PolynomialAnsatz):
                param_history.append(np.array(A_ansatz.params))
                # Check parameters for NaNs
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

        # Randomize momenta for naive HMC every momentum_refresh_interval steps
        # if (k % momentum_refresh_interval == 0) and (k < N_steps):
        #     key, sub = jax.random.split(key)

        #     p_naive = jax.random.normal(sub, (M, dim))
        #     p_cd = p_naive.copy()

        
        

        if k == N_steps:
            break

        # TODO: consider if correct to add delta_t here

        lam_k1 = float(lam_fn(t_k + delta_t))
        dot_lam_k1 = float(dot_lam_fn(t_k + delta_t))

        # Check next lambda values for NaNs
        if jnp.isnan(lam_k1):
            print(f"⚠️  NaN detected in lam_k1 at step {k}")
            break
        if jnp.isnan(dot_lam_k1):
            print(f"⚠️  NaN detected in dot_lam_k1 at step {k}")
            break

        key, sub = jax.random.split(key)
        subs = jax.random.split(sub, M)
        naive_step = jax.vmap(lambda q, p, lam, lam_next, eps, L, rng_key: with_maruyama(make_leapfrog_step(make_T(lam), make_V(lam)))(q,p,eps,L=L, rng_key=rng_key), in_axes=(0, 0, None, None, None, None, 0))

        # --- Naïve step ---
        q_naive, p_naive = jax.jit(naive_step)(q_naive, p_naive, lam_k, lam_k1, eps, eps*momentum_refresh_interval, subs)

        # Check for NaNs in naive HMC
        if check_nans("q_naive", q_naive, k):
            print(f"  Stopping simulation due to NaNs in naive HMC at step {k}")
            break
        if check_nans("p_naive", p_naive, k):
            print(f"  Stopping simulation due to NaNs in naive HMC at step {k}")
            break

        # --- CD step ---
        try:
            key, sub = jax.random.split(key)
            cd_step = jax.vmap(lambda q, p, lam, lam_next, dot_lam, dot_lam_next, eps: make_cd_leapfrog_step(make_T(lam), make_V(lam), A_ansatz, lam, lam_next, dot_lam, dot_lam_next)(q, p, eps), in_axes=(0, 0, None, None, None, None, None))
            q_cd, p_cd = jax.jit(cd_step)(q_cd, p_cd, lam_k, lam_k1, dot_lam_k, dot_lam_k1, eps)

            # if k % 10 == 0 and k > 0:
            #         key, sub = jax.random.split(key)
            #         p_cd = jax.random.normal(sub, (M, dim))
            
            # Check for NaNs in CD HMC
            if check_nans("q_cd", q_cd, k):
                print(f"  Stopping simulation due to NaNs in CD HMC at step {k}")
                # Debug: check ansatz values for the problematic samples
                if not isinstance(A_ansatz, AnalyticAnsatz):
                    print(f"  Debugging ansatz values at step {k}:")
                    # Check a few samples for ansatz values
                    for i in range(min(5, len(q_cd))):
                        try:
                            ansatz_val = A_ansatz(q_cd[i], p_cd[i])
                            print(f"    Sample {i}: q={q_cd[i]}, p={p_cd[i]}, A={ansatz_val}")
                        except:
                            print(f"    Sample {i}: Error computing ansatz value")
                break
            if check_nans("p_cd", p_cd, k):
                print(f"  Stopping simulation due to NaNs in CD HMC at step {k}")
                break
                
        except Exception as e:
            print(f"⚠️  Error in CD step at step {k}: {e}")
            break

        # --- Re-equilibration steps after CD step ---
        if re_equil_steps > 0:
            # Store pre-equilibration state
            q_cd_pre_equil = q_cd.copy()
            p_cd_pre_equil = p_cd.copy()
            
            # Perform re-equilibration using naive HMC steps at the CURRENT lambda value
            # T_current = make_T(lam_k1)  # Use the next lambda value
            # V_current = make_V(lam_k1)
            # equil_step = with_maruyama(make_leapfrog_step(T_current, V_current))
            
            for equil_idx in range(re_equil_steps):
                key, sub = jax.random.split(key)
                subs = jax.random.split(sub, M)
                equil_eps = 0.05
                equil_L = 4.0 # eps*momentum_refresh_interval
                # q_cd_pre_equil, p_cd_pre_equil = generate_initial_samples(M, make_T, make_V, lam_k1, sub, dim, num_steps=1000, eps=0.1, L=4.0)
                q_cd_pre_equil, p_cd_pre_equil = jax.jit(naive_step)(q_cd_pre_equil, p_cd_pre_equil, lam_k1, None, equil_eps, equil_L, subs)
                # q_cd_pre_equil, p_cd_pre_equil = jax.vmap(lambda q, p: equil_step(q, p, eps/10, L=eps*momentum_refresh_interval, rng_key=sub))(q_cd_pre_equil, p_cd_pre_equil)

                # q_cd_pre_equil = q_cd_pre_equil
                # p_cd_pre_equil = p_cd_pre_equil

                
                # Check for NaNs during re-equilibration
                if check_nans("re_equil_q", q_cd_pre_equil, f"{k}_{equil_idx}"):
                    print(f"  Stopping re-equilibration due to NaNs at step {k}, equil {equil_idx}")
                    break
                if check_nans("re_equil_p", p_cd_pre_equil, f"{k}_{equil_idx}"):
                    print(f"  Stopping re-equilibration due to NaNs at step {k}, equil {equil_idx}")
                    break
                
                # Randomize momenta periodically during re-equilibration
                # if equil_idx % 10 == 0 and equil_idx > 0:
                #     key, sub = jax.random.split(key)
                #     p_cd_pre_equil = jax.random.normal(sub, (M, dim))
            
            
            
            # Store post-equilibration state at the current snapshot time
            # This represents the state after CD step + re-equilibration at the current lambda
            if k % 10 == 0:  # Store at the current snapshot time
                snapshots['cd_post_equil'].append(np.array(q_cd_pre_equil))
                snapshots['lam_post_equil'].append(lam_k1) # Store lambda at post-equilibration (use lam_k1 since re-equilibration happens at next lambda)

            q_cd = q_cd_pre_equil.copy()
            p_cd = p_cd_pre_equil.copy()

    # Add the detailed energy statistics to snapshots
    snapshots['detailed_energy_stats'] = all_energy_stats
    snapshots['detailed_times'] = all_times

    print(f"Simulation completed after {k} steps")
    if re_equil_steps > 0:
        print(f"Performed {re_equil_steps} re-equilibration steps after each CD step")
    
    return A_ansatz, snapshots, loss_histories, param_history 