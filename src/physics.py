import jax
import jax.numpy as jnp


def partially_refresh_momentum(momentum, rng_key, step_size, L):

    # return momentum

    c1 = jnp.exp(-step_size/L)
    c2 = jnp.sqrt((1-c1**2))
    z = jax.random.normal(rng_key, shape=momentum.shape, dtype=momentum.dtype)
    new_momentum = c1*momentum + c2*z

    return jax.lax.cond(
        jnp.isinf(L),
        lambda _: momentum,
        lambda _: new_momentum,
        operand=None,
    )

# =============================================================================
# 1) GENERAL POISSON BRACKET FUNCTION (scalar)
# =============================================================================
def poisson_bracket_fn(f, g):
    """Compute the Poisson bracket {f, g} for multi-dimensional q and p."""
    df_dq = jax.grad(lambda qr, pr: f(qr, pr), argnums=0)
    df_dp = jax.grad(lambda qr, pr: f(qr, pr), argnums=1)
    dg_dq = jax.grad(lambda qr, pr: g(qr, pr), argnums=0)
    dg_dp = jax.grad(lambda qr, pr: g(qr, pr), argnums=1)

    def poisson_bracket(q, p):
        df_dq_val = df_dq(q, p)
        df_dp_val = df_dp(q, p)
        dg_dq_val = dg_dq(q, p)
        dg_dp_val = dg_dp(q, p)
        
        # For multi-dimensional case, compute dot product of gradients
        return jnp.dot(df_dq_val, dg_dp_val) - jnp.dot(df_dp_val, dg_dq_val)
    
    return poisson_bracket

def make_p_update(V):
    """Create momentum update function for potential V."""
    def p_update(q, p, eps):
        # Assume V returns a scalar
        return p - eps * jax.grad(V)(q)
    return p_update

def make_x_update(T):
    """Create position update function for kinetic energy T."""
    def x_update(q, p, eps):
        # Assume T returns a scalar
        return q + eps * jax.grad(T)(p)
    return x_update

# =============================================================================
# GENERAL LEAPFROG INTEGRATOR FOR SEPARABLE HAMILTONIAN
# =============================================================================
def make_leapfrog_step(T, V):
    """Create a leapfrog step function for separable Hamiltonian."""
    p_update = make_p_update(V)
    x_update = make_x_update(T)

    def leapfrog(q, p, eps):
        p_half = p_update(q, p, eps*0.5)
        q_new = x_update(q, p_half, eps)
        p_new = p_update(q_new, p_half, eps*0.5)
        return q_new, p_new
    return leapfrog

def with_maruyama(integrator):
    def maruyama(q, p, eps, L, rng_key):
        key1, key2 = jax.random.split(rng_key)
        p = partially_refresh_momentum(p, key1, eps/2, L)
        q, p = integrator(q, p, eps)
        p = partially_refresh_momentum(p, key2, eps/2, L)
        return q, p
    return maruyama

clip_value = 20.0

def make_cd_euler_step(T, V, A_ansatz, lam, lam_next, dot_lam, dot_lam_next):
    """Create a counterdiabatic Euler step function."""
    dA_dq_scalar = jax.grad(A_ansatz, argnums=0)
    dA_dp_scalar = jax.grad(A_ansatz, argnums=1)
    
    def cd_euler(q, p, eps):
        # Compute gradients of the gauge potential
        dA_dq = dA_dq_scalar(q, p)
        dA_dp = dA_dp_scalar(q, p)
        
        # Clip the gradients to prevent large forces
        dA_dq = jnp.clip(dA_dq, -clip_value, clip_value)
        dA_dp = jnp.clip(dA_dp, -clip_value, clip_value)
        
        # Euler integration for counterdiabatic dynamics
        # dq/dt = dot_lam * dA/dp
        # dp/dt = -dot_lam * dA/dq
        q_new = q + eps * dot_lam * dA_dp
        p_new = p - eps * dot_lam * dA_dq
        
        return q_new, p_new
    return cd_euler 


def make_cd_leapfrog_step(T, V, A_ansatz, lam, lam_next, dot_lam, dot_lam_next):
    """Create a counterdiabatic leapfrog step function.
    
    The full Hamiltonian is H + dot_lam*A where H = T + V.
    Note: Consider updating lambda to lam_next at the midpoint of the step
    for better accuracy, i.e., use (lam + lam_next)/2 and (dot_lam + dot_lam_next)/2.
    """
    dA_dq_scalar = jax.grad(A_ansatz, argnums=0)
    dA_dp_scalar = jax.grad(A_ansatz, argnums=1)
    
    def cd_leapfrog(q, p, eps):
        # Compute gradients of the gauge potential at initial point
        dA_dq = dA_dq_scalar(q, p)
        dA_dp = dA_dp_scalar(q, p)
        
        # Clip the gradients to prevent large forces
        dA_dq = jnp.clip(dA_dq, -clip_value, clip_value)
        dA_dp = jnp.clip(dA_dp, -clip_value, clip_value)

        # q half-step: q_half = q + (eps/2) * (dT/dp + dot_lam * dA/dp)
        q_half = q + 0.5 * eps * (jax.grad(T)(p) + dot_lam * dA_dp)
        
        # p full-step: p_new = p - eps * (dV/dq + dot_lam * dA/dq) at q_half
        dA_dq_half = dA_dq_scalar(q_half, p)
        dA_dq_half = jnp.clip(dA_dq_half, -clip_value, clip_value)
        p_new = p - eps * (jax.grad(V)(q_half) + dot_lam * dA_dq_half)
        
        # q half-step: q_new = q_half + (eps/2) * (dT/dp + dot_lam * dA/dp) at p_new
        dA_dp_new = dA_dp_scalar(q_half, p_new)
        dA_dp_new = jnp.clip(dA_dp_new, -clip_value, clip_value)
        q_new = q_half + 0.5 * eps * (jax.grad(T)(p_new) + dot_lam * dA_dp_new)
        
        return q_new, p_new
    return cd_leapfrog 