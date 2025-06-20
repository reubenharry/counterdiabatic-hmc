import jax
import jax.numpy as jnp

m = 1.0

# =============================================================================
# 1) GENERAL POISSON BRACKET FUNCTION (scalar)
# =============================================================================
def poisson_bracket_fn(f, g):

    df_dq = jax.grad(lambda qr, pr: f(qr, pr), argnums=0)
    df_dp = jax.grad(lambda qr, pr: f(qr, pr), argnums=1)
    dg_dq = jax.grad(lambda qr, pr: g(qr, pr), argnums=0)
    dg_dp = jax.grad(lambda qr, pr: g(qr, pr), argnums=1)

    return lambda q,p: df_dq(q, p) * dg_dp(q, p) - df_dp(q, p) * dg_dq(q, p)

make_p_update = lambda V: lambda q, p, eps: p - eps * jax.grad(V)(q)
make_x_update = lambda T: lambda q, p, eps: q + eps * jax.grad(T)(p)

# =============================================================================
# GENERAL LEAPFROG INTEGRATOR FOR SEPARABLE HAMILTONIAN
# =============================================================================
def make_leapfrog_step(T, V):

    p_update = make_p_update(V)
    x_update = make_x_update(T)

    def leapfrog(q, p, eps):
        p_half = p_update(q, p, eps*0.5)
        q_new = x_update(q, p_half, eps)
        p_new = p_update(q_new, p_half, eps*0.5)
        return q_new, p_new
    return leapfrog

def make_cd_leapfrog_step(T, V, A_ansatz, lam, lam_next, dot_lam, dot_lam_next):
    dA_dq_scalar = jax.grad(A_ansatz, argnums=0)
    dA_dp_scalar = jax.grad(A_ansatz, argnums=1)
    def cd_leapfrog(q, p, eps):
        p_half = p - 0.5 * eps * (jax.grad(V)(q) + dot_lam * dA_dq_scalar(q, p))
        q_new = q + eps * (jax.grad(T)(p_half) + dot_lam * dA_dp_scalar(q, p_half))
        p_new = p_half - 0.5 * eps * (jax.grad(V)(q_new) + dot_lam * dA_dq_scalar(q_new, p_half))
        return q_new, p_new
    return cd_leapfrog 