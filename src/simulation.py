import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from .physics import m, make_leapfrog_step, make_cd_leapfrog_step
from .fitting import fit_gauge_potential
from .ansatze import AnalyticAnsatz, PolynomialAnsatz, NeuralNetworkAnsatz

# =============================================================================
# 6) SImuLATION: NAÏVE HMC VS CD WITH ONLINE FITTING
# =============================================================================
def generate_initial_samples(M, make_T, make_V, lam, key, num_steps=1000, eps=0.05):
    """Generate M samples from the distribution at the given temperature using HMC."""
    # Start from random positions
    key, sub = jax.random.split(key)
    q = jax.random.normal(sub, (M,))
    key, sub = jax.random.split(key)
    p = jax.random.normal(sub, (M,)) * jnp.sqrt(m)

    # Run HMC for a while to get to equilibrium
    T = make_T(lam)
    V = make_V(lam)
    step = make_leapfrog_step(T, V)
    
    for _ in range(num_steps):
        q, p = jax.vmap(lambda q, p: step(q, p, eps))(q, p)
        # Randomize momenta periodically
        if _ % 20 == 0:
            key, sub = jax.random.split(key)
            p = jax.random.normal(sub, (M,)) * jnp.sqrt(m)
    
    return q, p

def run_simulation(M, N_steps, delta_t, eps, momentum_refresh_interval, fit_every, make_T, make_V, A_ansatz, lam_fn, dot_lam_fn, key):
    # Generate initial samples from the correct distribution
    initial_lam = float(lam_fn(0.0))
    q_naive, p_naive = generate_initial_samples(M, make_T, make_V, initial_lam, key)
    q_cd = q_naive.copy()
    p_cd = p_naive.copy()

    loss_histories = []
    snapshots = {'naive': [], 'cd': [], 'lam': []}
    param_history = []  # Track parameter history

    for k in range(N_steps + 1):
        t_k = k * delta_t
        lam_k = float(lam_fn(t_k))
        dot_lam_k = float(dot_lam_fn(t_k))

        # Update lambda parameter for analytic ansatz
        if isinstance(A_ansatz, AnalyticAnsatz):
            A_ansatz = eqx.tree_at(lambda m: m.params, A_ansatz, jnp.array([lam_k]))

        # Re-fit A every fit_every steps
        if not isinstance(A_ansatz, AnalyticAnsatz) and (k % fit_every == 0) and (k < N_steps):
            samples = np.stack([np.array(q_cd), np.array(p_cd)], axis=1)
            A_ansatz, loss_history = fit_gauge_potential(lam_k, samples,
                                        make_T=make_T, make_V=make_V,
                                        A_ansatz=A_ansatz,
                                        num_iters=50000, lr=1e-4)
            loss_histories.append(loss_history)

        # Record snapshots and parameters every 10 steps
        if k % 10 == 0:
            snapshots['naive'].append(np.array(q_naive))
            snapshots['cd'].append(np.array(q_cd))
            snapshots['lam'].append(lam_k)
            # Record parameters
            if isinstance(A_ansatz, PolynomialAnsatz):
                param_history.append(np.array(A_ansatz.params))
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

        # Randomize momenta for naive HMC every momentum_refresh_interval steps
        if (k % momentum_refresh_interval == 0) and (k < N_steps):
            key, sub = jax.random.split(key)
            p_naive = jax.random.normal(sub, (M,)) * jnp.sqrt(m)
            p_cd = p_naive.copy()

        if k == N_steps:
            break

        lam_k1 = float(lam_fn(t_k + delta_t))
        dot_lam_k1 = float(dot_lam_fn(t_k + delta_t))

        naive_step = jax.vmap(lambda q, p, lam, lam_next, eps: make_leapfrog_step(make_T(lam), make_V(lam))(q,p,eps), in_axes=(0, 0, None, None, None))

        # --- Naïve step ---
        q_naive, p_naive = naive_step(q_naive, p_naive, lam_k, lam_k1, eps)

        # Check for NaNs in naive HMC
        if jnp.isnan(q_naive).any():
            print(f"Warning: NaNs detected in q_naive at step {k} (count: {jnp.isnan(q_naive).sum()})")
        if jnp.isnan(p_naive).any():
            print(f"Warning: NaNs detected in p_naive at step {k} (count: {jnp.isnan(p_naive).sum()})")

        # --- CD step ---
        cd_step = jax.vmap(lambda q, p: make_cd_leapfrog_step(make_T(lam_k), make_V(lam_k), A_ansatz, lam_k, lam_k1, dot_lam_k, dot_lam_k1)(q, p, eps))
        q_cd, p_cd = cd_step(q_cd, p_cd)

    return A_ansatz, snapshots, loss_histories, param_history 