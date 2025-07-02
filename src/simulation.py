import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from .physics import m, make_leapfrog_step, make_cd_leapfrog_step
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
def generate_initial_samples(M, make_T, make_V, lam, key, dim, num_steps=1000, eps=0.05):
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
    p = jax.random.normal(sub, (M, dim)) * jnp.sqrt(m)

    # Check initial samples for NaNs
    check_nans("initial_q", q)
    check_nans("initial_p", p)

    # Run HMC for a while to get to equilibrium
    T = make_T(lam)
    V = make_V(lam)
    step = make_leapfrog_step(T, V)
    
    for step_idx in range(num_steps):
        q, p = jax.vmap(lambda q, p: step(q, p, eps))(q, p)
        
        # Check for NaNs during equilibration
        if check_nans("equilibration_q", q, step_idx):
            print(f"  Stopping equilibration early due to NaNs at step {step_idx}")
            break
        if check_nans("equilibration_p", p, step_idx):
            print(f"  Stopping equilibration early due to NaNs at step {step_idx}")
            break
            
        # Randomize momenta periodically
        if step_idx % 20 == 0:
            key, sub = jax.random.split(key)
            p = jax.random.normal(sub, (M, dim)) * jnp.sqrt(m)
    
    return q, p

def run_simulation(M, N_steps, delta_t, eps, momentum_refresh_interval, fit_every, num_initial_iterations, num_iterations, make_T, make_V, A_ansatz, lam_fn, dot_lam_fn, key, dim, learning_rate=1e-4):
    # Generate initial samples from the correct distribution
    initial_lam = float(lam_fn(0.0))
    print(f"Generating initial samples with λ = {initial_lam}")
    q_naive, p_naive = generate_initial_samples(M, make_T, make_V, initial_lam, key, dim)
    q_cd = q_naive.copy()
    p_cd = p_naive.copy()

    # Check initial samples
    check_nans("initial_q_naive", q_naive)
    check_nans("initial_p_naive", p_naive)
    check_nans("initial_q_cd", q_cd)
    check_nans("initial_p_cd", p_cd)

    loss_histories = []
    snapshots = {'naive': [], 'cd': [], 'lam': []}
    param_history = []  # Track parameter history
    first_fit = True  # Track if this is the first fitting

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
                
            A_ansatz, loss_history = fit_gauge_potential(lam_k, samples,
                                        make_T=make_T, make_V=make_V,
                                        A_ansatz=A_ansatz,
                                        num_iters=num_iters, lr=learning_rate)
            
            # Check loss history for NaNs
            if any(jnp.isnan(loss) for loss in loss_history):
                print(f"⚠️  NaN detected in loss history at step {k}")
                nan_indices = [i for i, loss in enumerate(loss_history) if jnp.isnan(loss)]
                print(f"  NaN losses at iterations: {nan_indices}")
            
            loss_histories.append(loss_history)

        # Record snapshots and parameters every 10 steps
        if k % 10 == 0:
            snapshots['naive'].append(np.array(q_naive))
            snapshots['cd'].append(np.array(q_cd))
            snapshots['lam'].append(lam_k)
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
        if (k % momentum_refresh_interval == 0) and (k < N_steps):
            key, sub = jax.random.split(key)
            p_naive = jax.random.normal(sub, (M, dim)) * jnp.sqrt(m)
            p_cd = p_naive.copy()

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

        naive_step = jax.vmap(lambda q, p, lam, lam_next, eps: make_leapfrog_step(make_T(lam), make_V(lam))(q,p,eps), in_axes=(0, 0, None, None, None))

        # --- Naïve step ---
        q_naive, p_naive = naive_step(q_naive, p_naive, lam_k, lam_k1, eps)

        # Check for NaNs in naive HMC
        if check_nans("q_naive", q_naive, k):
            print(f"  Stopping simulation due to NaNs in naive HMC at step {k}")
            break
        if check_nans("p_naive", p_naive, k):
            print(f"  Stopping simulation due to NaNs in naive HMC at step {k}")
            break

        # --- CD step ---
        try:
            cd_step = jax.vmap(lambda q, p: make_cd_leapfrog_step(make_T(lam_k), make_V(lam_k), A_ansatz, lam_k, lam_k1, dot_lam_k, dot_lam_k1)(q, p, eps))
            q_cd, p_cd = cd_step(q_cd, p_cd)
            
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

    print(f"Simulation completed after {k} steps")
    return A_ansatz, snapshots, loss_histories, param_history 