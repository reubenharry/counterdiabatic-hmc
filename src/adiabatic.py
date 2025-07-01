import jax
import jax.numpy as jnp
import numpy as np

def run_adiabatic(M, N_steps, eps, make_T, make_V, make_dV, make_dT_dp, make_dV_dq, make_dVB_dq, E_deltaV_fn, key):
    """
    Run the adiabatic Monte Carlo integrator (Algorithm 1 from Betancourt 2014) for M parallel chains.
    Args:
        M: number of parallel chains
        N_steps: number of integration steps
        eps: step size
        make_T: function returning T(q, p)
        make_V: function returning V(q)
        make_dV: function returning ΔV(q)
        make_dT_dp: function returning ∂T/∂p (as a function of p)
        make_dV_dq: function returning ∂V/∂q (as a function of q)
        make_dVB_dq: function returning ∂V_B/∂q (as a function of q)
        E_deltaV_fn: function returning E[ΔV] (scalar or vectorized over chains)
        key: JAX PRNGKey
    Returns:
        Dict with trajectories: {'q': q_traj, 'p': p_traj, 'beta': beta_traj}
    """
    # Initialize β = 0, q ~ π_β, p ~ ξ
    beta = jnp.zeros(M)
    key, sub = jax.random.split(key)
    q = jax.random.normal(sub, (M,))
    key, sub = jax.random.split(key)
    p = jax.random.normal(sub, (M,))

    # Preallocate trajectories
    q_traj = [q]
    p_traj = [p]
    beta_traj = [beta]

    for step in range(N_steps):
        # 1. β ← β − (ε/2) p · ∂T/∂p
        dT_dp = make_dT_dp(p)
        beta = beta - (eps/2) * p * dT_dp

        # 2. q ← q + (−ε/2) ∂T/∂p
        q = q + (-eps/2) * dT_dp

        # 3. p ← p − (−ε/2) ∂V_B/∂q
        dVB_dq = make_dVB_dq(q)
        p = p - (-eps/2) * dVB_dq

        # 4. p ← p − (−ε) β ∂ΔV/∂q
        dDeltaV_dq = jax.grad(make_dV)(q)
        p = p - (-eps) * beta * dDeltaV_dq

        #    + (−ε) (ΔV − E[ΔV]) p
        deltaV = make_dV(q)
        E_deltaV = E_deltaV_fn(q, p, beta)  # Should be shape (M,)
        p = p + eps * (deltaV - E_deltaV) * p

        # 5. p ← p − (−ε/2) ∂V_B/∂q
        dVB_dq = make_dVB_dq(q)
        p = p - (-eps/2) * dVB_dq

        # 6. β ← β − (ε/2) p · ∂T/∂p
        dT_dp = make_dT_dp(p)
        beta = beta - (eps/2) * p * dT_dp

        # 7. q ← q + (−ε/2) ∂T/∂p
        q = q + (-eps/2) * dT_dp

        # Store trajectory
        q_traj.append(q)
        p_traj.append(p)
        beta_traj.append(beta)

    # Stack trajectories
    q_traj = jnp.stack(q_traj)
    p_traj = jnp.stack(p_traj)
    beta_traj = jnp.stack(beta_traj)
    return {'q': q_traj, 'p': p_traj, 'beta': beta_traj} 