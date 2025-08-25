import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from .physics import make_cd_leapfrog_step, make_leapfrog_step, make_cd_euler_step, partially_refresh_momentum, with_maruyama
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

def compute_ess(log_weights):
    """Compute effective sample size from log weights using the formula:
    ESS = (Σ_{n=1}^{N} w_t(x_{t-1}^n))^2 / Σ_{n=1}^{N} w_t(x_{t-1}^n)^2
    """
    # Convert log weights to regular weights
    weights = jnp.exp(log_weights - jnp.max(log_weights))  # Numerical stability
    
    # Compute the formula: (sum of weights)^2 / sum of squared weights
    sum_weights = jnp.sum(weights)
    sum_squared_weights = jnp.sum(weights ** 2)
    
    # ESS = (sum_weights)^2 / sum_squared_weights
    ess = (sum_weights ** 2) / sum_squared_weights
    
    return ess

def multinomial_resample(q, p, log_weights, rng_key, M):
    """Perform multinomial resampling and reset weights to uniform."""
    weights = jnp.exp(log_weights - jnp.max(log_weights))  # Numerical stability
    weights = weights / jnp.sum(weights)  # Normalize
    
    # Generate multinomial samples
    indices = jax.random.choice(rng_key, M, shape=(M,), p=weights, replace=True)
    
    # Resample particles
    q_resampled = q[indices]
    p_resampled = p[indices]
    
    # Reset weights to uniform (log weights = 0)
    log_weights_reset = jnp.zeros(M)
    
    return q_resampled, p_resampled, log_weights_reset

def compute_energy_stats(q, p, lam, make_T, make_V, A_ansatz=None):
    """Compute average H, H², ∂H/∂λ, (∂H/∂λ)², and {A,H} over particles."""
    T = make_T(lam)
    V = make_V(lam)
    
    # Compute H for each particle
    H_vals = jax.vmap(lambda qr, pr: T(pr) + V(qr))(q, p)
    
    # Compute ∂H/∂λ for each particle
    dH_dlam = lambda q, p: (jax.grad(lambda q, p, lam: make_V(lam)(q), argnums=2)(q, p, lam))
    
    dH_dlam_vals = jax.vmap(lambda qr, pr: dH_dlam(qr, pr))(q, p)
    
    # Compute averages
    avg_H = jnp.mean(H_vals)
    avg_H_sq = jnp.mean(H_vals ** 2)
    avg_dH_dlam = jnp.mean(dH_dlam_vals)
    avg_dH_dlam_sq = jnp.mean(dH_dlam_vals ** 2)
    
    # Compute {A,H} if A_ansatz is provided
    avg_A_H = 0.0
    avg_A_H_sq = 0.0
    if A_ansatz is not None:
        from .physics import poisson_bracket_fn
        H_fixed = lambda q, p: T(p) + V(q)
        A_H_vals = jax.vmap(lambda qr, pr: poisson_bracket_fn(A_ansatz, H_fixed)(qr, pr))(q, p)
        avg_A_H = float(jnp.mean(A_H_vals))
        avg_A_H_sq = float(jnp.mean(A_H_vals ** 2))
    
    return {
        'avg_H': float(avg_H),
        'avg_H_sq': float(avg_H_sq),
        'avg_dH_dlam': float(avg_dH_dlam),
        'avg_dH_dlam_sq': float(avg_dH_dlam_sq),
        'avg_A_H': avg_A_H,
        'avg_A_H_sq': avg_A_H_sq,
        'H_vals': H_vals  # Store individual H values
    }

def compute_naive_weights(q, p, lam_k, lam_k1, make_T, make_V):
    """Compute importance weights for naive HMC based on energy difference."""
    # Compute energy at old and new lambda values for each particle
    T_old = make_T(lam_k)
    V_old = make_V(lam_k)
    T_new = make_T(lam_k1)
    V_new = make_V(lam_k1)
    
    # Compute energies for each particle
    def compute_energy(q, p, T_fn, V_fn):
        return T_fn(p) + V_fn(q)
    
    # Vectorize over particles
    energy_old = jax.vmap(lambda q, p: compute_energy(q, p, T_old, V_old))(q, p)
    energy_new = jax.vmap(lambda q, p: compute_energy(q, p, T_new, V_new))(q, p)

    # H = lambda lam: lambda q, p: make_T(lam)(p) + make_V(lam)(q)
    # dH_dt = lambda qq, pp: (jax.grad(lambda q, p, lam: H(lam)(q, p), argnums=2)(qq, pp, lam_k))
    
    # Log weight update: log(w_new) = log(w_old) - (H_new - H_old)
    # This is the negative change in energy (Boltzmann factor)

    direct = (energy_new - energy_old)
    # indirect = jax.vmap(lambda q, p: dH_dt(q,p)*(lam_k1 - lam_k))(q,p)
    # print("fpp", direct, indirect, np.abs(direct - indirect))
    log_weight_update = -(energy_new - energy_old)
    # log_weight_update = -indirect
    
    return log_weight_update



def record_snapshots(snapshots, k, q_cd, lam_k, use_weights, log_weights_cd, 
                    resampling_count_cd, param_history, A_ansatz):
    """Record snapshots and parameters every step."""
    if True:  # Record every step instead of every 10 steps
        # Store weighted samples for CD-HMC
        snapshots['cd_weighted'].append(np.array(q_cd))
        snapshots['weights_cd'].append(np.array(log_weights_cd))
        snapshots['resampling_events_cd'].append(resampling_count_cd)
        
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

# =============================================================================
# 6) SImuLATION: NAÏVE HMC VS CD WITH ONLINE FITTING
# =============================================================================
def generate_initial_samples(M, make_T, make_V, lam, key, dim, variance=None):
    """Generate M samples from a Gaussian distribution with variance matching the potential.
    
    Args:
        M: Number of samples
        make_T: Function to create kinetic energy (unused, kept for compatibility)
        make_V: Function to create potential energy (used to determine variance)
        lam: Current lambda value (used to determine variance)
        key: JAX random key
        dim: Dimension of the system
        variance: Variance of the Gaussian distribution (if None, computed from potential)
    """
    # If variance is not provided, compute it from the potential
    if variance is None:
        # For a potential V(q) = 0.5 * k * q², the variance is 1/k
        # We can compute this by evaluating the potential at a test point
        test_q = jnp.ones(dim)
        V = make_V(lam)
        potential_value = V(test_q)
        # V(q) = 0.5 * k * ||q||², so k = 2 * V(q) / ||q||²
        k = 2.0 * potential_value / jnp.sum(test_q ** 2)
        variance = 1.0 / k
        print(f"Computed variance from potential: {variance:.3f} (k = {k:.3f})")
    
    # Draw independent samples from Gaussian with given variance
    key, sub = jax.random.split(key)
    q = jax.random.normal(sub, (M, dim)) * jnp.sqrt(variance)
    key, sub = jax.random.split(key)
    p = jax.random.normal(sub, (M, dim))
    
    # Check initial samples for NaNs
    check_nans("initial_q", q)
    check_nans("initial_p", p)
    
    return q, p

def run_naive_hmc_simulation(M, N_steps, delta_t, eps, momentum_refresh_interval, make_T, make_V, lam_fn, dot_lam_fn, key, dim, use_weights=False, ess_threshold=0.5, snapshot_every=1):
    """Run naive HMC simulation without fitting the ansatz (for efficiency)."""
    
    # Generate initial samples from the correct distribution
    initial_lam = float(lam_fn(0.0))
    print(f"Generating initial samples with λ = {initial_lam}")
    q_naive, p_naive = generate_initial_samples(M, make_T, make_V, initial_lam, key, dim)
    # jax.debug.print("q_naive {x}", x=q_naive[0])

    # Check initial samples
    check_nans("initial_q_naive", q_naive)
    check_nans("initial_p_naive", p_naive)

    # Initialize weights for SMC (Sequential Monte Carlo) - always use weights
    log_weights = jnp.zeros(M)  # Initial weights are uniform (log weights = 0)
    if use_weights:
        print("Using Sequential Monte Carlo with importance weights")
    else:
        print("Using Sequential Monte Carlo with unit weights (no resampling)")

    snapshots = {'naive': [], 'naive_weighted': [], 'lam_pre_equil': [], 'weights_naive': [], 'resampling_events_naive': []}
    resampling_count = 0  # Track number of resampling events

    # Store previous energy values for computing changes
    prev_naive_H_vals = None
    
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

        # Compute energy statistics at every timestep
        naive_stats = compute_energy_stats(q_naive, p_naive, lam_k, make_T, make_V, A_ansatz=None)
        
        # Compute energy changes if we have previous values
        if prev_naive_H_vals is not None:
            # Compute individual particle energy changes
            naive_delta_H_vals = naive_stats['H_vals'] - prev_naive_H_vals
            
            # Average the individual changes
            naive_avg_delta_H = jnp.mean(naive_delta_H_vals)
            
            # Add change statistics to the stats dictionaries
            naive_stats['avg_delta_H'] = float(naive_avg_delta_H)
            naive_stats['avg_delta_H_sq'] = float(jnp.mean(naive_delta_H_vals ** 2))
        else:
            # First timestep - no change to compute
            naive_stats['avg_delta_H'] = 0.0
            naive_stats['avg_delta_H_sq'] = 0.0
        
        # Store energy statistics for this timestep
        all_energy_stats.append({
            'naive': naive_stats
        })
        all_times.append(t_k)
        
        # Update previous energy values for next iteration
        prev_naive_H_vals = naive_stats['H_vals']

        # Record snapshots every snapshot_every steps
        if k % snapshot_every == 0:
            snapshots['naive'].append(np.array(q_naive))
            snapshots['naive_weighted'].append(np.array(q_naive))
            snapshots['weights_naive'].append(np.array(log_weights))
            snapshots['resampling_events_naive'].append(resampling_count)
            snapshots['lam_pre_equil'].append(lam_k)

        if k == N_steps:
            break

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
        
        naive_step = jax.vmap(lambda q, p, lam, lam_next, eps, L, rng_key, t: with_maruyama(make_leapfrog_step(make_T(lam), make_V(lam), make_T(lam_next), make_V(lam_next), lam_fn, dot_lam_fn))(q, p, eps, L=L, rng_key=rng_key, t=t), in_axes=(0, 0, None, None, None, None, 0, None))
        # Create integrator with weight calculation if using weights
        if use_weights:
            # --- Naïve step with weight calculation ---
            q_naive, p_naive, step_weights = jax.jit(naive_step)(q_naive, p_naive, lam_k, lam_k1, eps, eps*momentum_refresh_interval, subs, t_k)
            
            # Update log weights using the step log weights
            log_weights = log_weights + step_weights
        else:
            # naive_step = jax.vmap(lambda q, p, lam, lam_next, eps, L, rng_key: with_maruyama(make_leapfrog_step(make_T(lam), make_V(lam), make_T(lam), make_V(lam_next)))(q, p, eps, L=L, rng_key=rng_key), in_axes=(0, 0, None, None, None, None, 0))
            # --- Naïve step without weight calculation ---
            q_naive, p_naive, _ = jax.jit(naive_step)(q_naive, p_naive, lam_k, lam_k1, eps, eps*momentum_refresh_interval, subs, t_k)

        # Check for NaNs in naive HMC
        if check_nans("q_naive", q_naive, k):
            print(f"  Stopping simulation due to NaNs in naive HMC at step {k}")
            break
        if check_nans("p_naive", p_naive, k):
            print(f"  Stopping simulation due to NaNs in naive HMC at step {k}")
            break

        # --- Check for NaNs in weights and handle resampling ---
        if use_weights:
            # Check for NaNs in weights
            if check_nans("log_weights", log_weights, k):
                print(f"  Stopping simulation due to NaNs in weights at step {k}")
                break
            
            # Check if resampling is needed
            ess = compute_ess(log_weights)
            ess_ratio = ess / M
            
            if ess_ratio < ess_threshold:
                print(f"  Resampling naive HMC at step {k}: ESS = {ess:.1f}/{M} ({ess_ratio:.3f})")
                key, sub = jax.random.split(key)
                q_naive, p_naive, log_weights = multinomial_resample(q_naive, p_naive, log_weights, sub, M)
                resampling_count += 1
                print(f"    Naive HMC resampling completed. Total resampling events: {resampling_count}")
        else:
            # Keep weights at 1 (log_weights = 0) when not using weights
            log_weights = jnp.zeros(M)

    # Add the detailed energy statistics to snapshots
    snapshots['detailed_energy_stats'] = all_energy_stats
    snapshots['detailed_times'] = all_times

    print(f"Simulation completed after {k} steps")
    if use_weights:
        print(f"Total resampling events: {resampling_count}")
    
    return snapshots

def run_simulation(M, N_steps, delta_t, eps, momentum_refresh_interval, fit_every, num_initial_iterations, num_iterations, make_T, make_V, A_ansatz, lam_fn, dot_lam_fn, key, dim, learning_rate=1e-4, re_equil_steps=0, use_weights=False, ess_threshold=0.5, snapshot_every=1):
    
    # Generate initial samples from the correct distribution
    initial_lam = float(lam_fn(0.0))
    print(f"Generating initial samples with λ = {initial_lam}")
    q_cd, p_cd = generate_initial_samples(M, make_T, make_V, initial_lam, key, dim)
    # jax.debug.print("q_cd {x}", x=q_cd[0])
    

    # Check initial samples
    check_nans("initial_q_cd", q_cd)
    check_nans("initial_p_cd", p_cd)

    # Initialize weights for SMC (Sequential Monte Carlo) - always use weights
    log_weights_cd = jnp.zeros(M)     # Initial weights for CD-HMC
    if use_weights:
        print("Using Sequential Monte Carlo with importance weights")
    else:
        print("Using Sequential Monte Carlo with unit weights (no resampling)")

    loss_histories = []
    snapshots = {'cd_pre_equil': [], 'cd_post_equil': [], 'cd_weighted': [], 'lam_pre_equil': [], 'lam_post_equil': [], 'energy_stats': [], 'weights_cd': [], 'resampling_events_cd': []}
    param_history = []  # Track parameter history
    first_fit = True
    resampling_count_cd = 0     # Track number of resampling events for CD



    # Store previous energy values for computing changes
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
            # Convert log weights to regular weights if using weights
            weights = None
            if use_weights:
                weights = jnp.exp(log_weights_cd - jnp.max(log_weights_cd))  # Numerical stability
                weights = weights / jnp.sum(weights)  # Normalize
            
            A_ansatz, loss_history = fit_gauge_potential(lam_k, samples,
                                        make_T=make_T, make_V=make_V,
                                        A_ansatz=A_ansatz,  # Pass current ansatz for warm start
                                        num_iters=num_iters, lr=learning_rate,
                                        use_regularization=False, weights=weights)
            
            # Check loss history for NaNs
            if any(jnp.isnan(loss) for loss in loss_history):
                print(f"⚠️  NaN detected in loss history at step {k}")
                nan_indices = [i for i, loss in enumerate(loss_history) if jnp.isnan(loss)]
                print(f"  NaN losses at iterations: {nan_indices}")
            
            loss_histories.append(loss_history)

        # Compute energy statistics at every timestep
        cd_stats = compute_energy_stats(q_cd, p_cd, lam_k, make_T, make_V, A_ansatz)
        
        # Compute energy changes if we have previous values
        if prev_cd_H_vals is not None:
            # Compute individual particle energy changes
            cd_delta_H_vals = cd_stats['H_vals'] - prev_cd_H_vals
            
            # Average the individual changes
            cd_avg_delta_H = jnp.mean(cd_delta_H_vals)
            
            # Add change statistics to the stats dictionaries
            cd_stats['avg_delta_H'] = float(cd_avg_delta_H)
            cd_stats['avg_delta_H_sq'] = float(jnp.mean(cd_delta_H_vals ** 2))
        else:
            # First timestep - no change to compute
            cd_stats['avg_delta_H'] = 0.0
            cd_stats['avg_delta_H_sq'] = 0.0
        
        # Store energy statistics for this timestep
        all_energy_stats.append({
            'cd': cd_stats
        })
        all_times.append(t_k)
        
        # Update previous energy values for next iteration
        prev_cd_H_vals = cd_stats['H_vals']

        # Record snapshots and parameters every snapshot_every steps
        if k % snapshot_every == 0:
            record_snapshots(snapshots, k, q_cd, lam_k, use_weights, log_weights_cd, 
                            resampling_count_cd, param_history, A_ansatz)

        # Note: Momentum randomization removed - only CD-HMC is performed here

        
        

        if k == N_steps:
            break

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
        

        # Note: Naive HMC step removed from run_simulation - only CD-HMC is performed here
        # Naive HMC is only used in re-equilibration

        # --- CD step ---
        try:
            L = eps*momentum_refresh_interval
            key, sub = jax.random.split(key)
            subs = jax.random.split(sub, M)
            
            # Create integrator with weight calculation if using weights
            cd_step = jax.vmap(lambda q, p, lam, lam_next, dot_lam, dot_lam_next, eps, rng_key, t: with_maruyama(make_cd_leapfrog_step(make_T, make_V, A_ansatz, lam, lam_next, dot_lam, dot_lam_next, lam_fn, dot_lam_fn))(q=q, p=p, eps=eps, L=L, rng_key=rng_key, t=t), in_axes=(0, 0, None, None, None, None, None, 0, None))
            q_cd, p_cd, step_weights_cd = jax.jit(cd_step)(q_cd, p_cd, lam_k, lam_k1, dot_lam_k, dot_lam_k1, eps, subs, t_k)
            if use_weights:
                
                # Update log weights using the step log weights
                # print("shape", step_weights_cd.shape)
                log_weights_cd = log_weights_cd + step_weights_cd
            # else:
            #     q_cd, p_cd, _ = jax.jit(cd_step)(q_cd, p_cd, lam_k, lam_k1, dot_lam_k, dot_lam_k1, eps, subs, t_k)

            # --- Check for NaNs in CD weights and handle resampling ---
            if use_weights:
                # Check for NaNs in CD weights
                if check_nans("log_weights_cd", log_weights_cd, k):
                    print(f"  Stopping simulation due to NaNs in CD weights at step {k}")
                    break
                
                # Check if resampling is needed for CD-HMC
                ess_cd = compute_ess(log_weights_cd)
                ess_ratio_cd = ess_cd / M
                
                if ess_ratio_cd < ess_threshold:
                    print(f"  Resampling CD-HMC at step {k}: ESS = {ess_cd:.1f}/{M} ({ess_ratio_cd:.3f})")
                    key, sub = jax.random.split(key)
                    q_cd, p_cd, log_weights_cd = multinomial_resample(q_cd, p_cd, log_weights_cd, sub, M)
                    resampling_count_cd += 1
                    print(f"    CD-HMC resampling completed. Total resampling events: {resampling_count_cd}")
            else:
                # Keep weights at 1 (log_weights = 0) when not using weights
                log_weights_cd = jnp.zeros(M)

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

            naive_step = jax.vmap(lambda q, p, lam, lam_next, eps, L, rng_key: with_maruyama(make_leapfrog_step(make_T(lam), make_V(lam)))(q,p,eps,L=L, rng_key=rng_key), in_axes=(0, 0, None, None, None, None, 0))

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
            if True:  # Store every step instead of every 10 steps
                snapshots['cd_post_equil'].append(np.array(q_cd_pre_equil))
                snapshots['lam_post_equil'].append(lam_k1) # Store lambda at post-equilibration (use lam_k1 since re-equilibration happens at next lambda)

            q_cd = q_cd_pre_equil.copy()
            p_cd = p_cd_pre_equil.copy()
        else:
            # When re_equil_steps = 0, store the CD state directly as post-equilibration
            if True:  # Store every step instead of every 10 steps
                snapshots['cd_post_equil'].append(np.array(q_cd))
                snapshots['lam_post_equil'].append(lam_k)
    
    # Ensure the final state is captured in post-equilibration snapshots
    # This is especially important when re_equil_steps = 0
    if len(snapshots['cd_post_equil']) == 0:  # Always capture final state since we're recording every step
        # If no post-equilibration snapshots exist or the last step wasn't captured
        snapshots['cd_post_equil'].append(np.array(q_cd))
        snapshots['lam_post_equil'].append(lam_k)
    
    # Also ensure the final state is captured in pre-equilibration snapshots
    # This is needed when the loop ends at k = N_steps
    if len(snapshots['cd_pre_equil']) == 0 or snapshots['cd_pre_equil'][-1].shape != q_cd.shape:
        # If the last step wasn't captured in pre-equilibration snapshots
        snapshots['cd_pre_equil'].append(np.array(q_cd))
        snapshots['lam_pre_equil'].append(lam_k)
        if use_weights:
            snapshots['cd_weighted'].append(np.array(q_cd))
            snapshots['weights_cd'].append(np.array(log_weights_cd))
            snapshots['resampling_events_cd'].append(resampling_count_cd)

    # Add the detailed energy statistics to snapshots
    snapshots['detailed_energy_stats'] = all_energy_stats
    snapshots['detailed_times'] = all_times

    print(f"Simulation completed after {k} steps")
    if re_equil_steps > 0:
        print(f"Performed {re_equil_steps} re-equilibration steps after each CD step")
    if use_weights:
        print(f"Total resampling events - CD-HMC: {resampling_count_cd}")
    
    return A_ansatz, snapshots, loss_histories, param_history 