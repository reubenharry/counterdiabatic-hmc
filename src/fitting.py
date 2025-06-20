import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from .physics import poisson_bracket_fn

# =============================================================================
#  FIT FUNCTION USING GENERAL POISSON BRACKET
# =============================================================================
def fit_gauge_potential(lam, samples, make_T, make_V, A_ansatz, num_iters=200, lr=0.01):
    """
    Fit A(q,p; θ) by minimizing mean_{samples}[ ( {A,H} - ∂H/∂μ )^2 ].
    Returns both the optimized parameters and the loss history.
    """
    qp_batch = jnp.array(samples)  # shape (N,2)

    H = lambda lam: lambda q, p: make_T(lam)(p) + make_V(lam)(q)
    H_fixed = H(lam)
    dH_fixed = lambda q, p: (jax.grad(lambda q, p, lam: H(lam)(q, p), argnums=2)(q, p, lam))

    def R(A_ansatz, q, p):
        return poisson_bracket_fn(A_ansatz, H_fixed)(q, p) - dH_fixed(q, p)

    def loss_fn(A_ansatz, qp_batch):
        qs = qp_batch[:, 0]
        ps = qp_batch[:, 1]
        R_vals = jax.vmap(lambda qr, pr, A_ansatz: R(A_ansatz, qr, pr), in_axes=(0, 0, None))(qs, ps, A_ansatz)
        return jnp.mean(R_vals ** 2)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(A_ansatz, eqx.is_array))

    @jax.jit
    def update(A_ansatz, opt_state, qp_batch):
        loss, grads = jax.value_and_grad(loss_fn)(A_ansatz, qp_batch)
        updates, opt_state = optimizer.update(eqx.filter(grads, eqx.is_array), opt_state)
        A_ansatz = eqx.apply_updates(A_ansatz, updates)
        return A_ansatz, opt_state, loss

    loss_history = []
    for _ in range(num_iters):
        A_ansatz, opt_state, loss = update(A_ansatz, opt_state, qp_batch)
        loss_history.append(float(loss))

    return A_ansatz, loss_history 