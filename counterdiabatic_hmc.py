# from functools import partial
import jax
import jax.numpy as jnp
import optax
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import equinox as eqx

# raise Exception

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


# =============================================================================
#  FIT FUNCTION USING GENERAL POISSON BRACKET
# =============================================================================
def fit_gauge_potential(lam, samples, init_params, make_T, make_V, A_ansatz, num_iters=200, lr=0.01):
    """
    Fit A(q,p; θ) by minimizing mean_{samples}[ ( {A,H} - ∂H/∂μ )^2 ].
    Returns both the optimized parameters and the loss history.
    """
    qp_batch = jnp.array(samples)  # shape (N,2)
    params = init_params

    H = lambda lam: lambda q, p: make_T(lam)(p) + make_V(lam)(q)
    H_fixed = H(lam)
    dH_fixed = lambda q, p: (jax.grad(lambda q, p, lam: H(lam)(q, p), argnums=2)(q, p, lam))

    def R(params, q, p):
        return poisson_bracket_fn(A_ansatz(params), H_fixed)(q, p) - dH_fixed(q, p)

    def loss_fn(params, qp_batch):
        qs = qp_batch[:, 0]
        ps = qp_batch[:, 1]
        R_vals = jax.vmap(lambda qr, pr, θ: R(θ, qr, pr), in_axes=(0, 0, None))(qs, ps, params)
        return jnp.mean(R_vals ** 2)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))

    @jax.jit
    def update(params, opt_state, qp_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, qp_batch)
        updates, opt_state = optimizer.update(eqx.filter(grads, eqx.is_array), opt_state)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss

    loss_history = []
    for _ in range(num_iters):
        params, opt_state, loss = update(params, opt_state, qp_batch)
        loss_history.append(float(loss))

    return params, loss_history


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

# def make_T(lam):
#     return lambda p: 0.5 * (p ** 2) / m
# def make_V(lam):
#     return lambda q: 0.5 * (q - lam) ** 2
    # return a weighted sum of a double well potential and a harmonic potential, both centered at 0, where lam is the weighting param (not the mean)
    # return lambda q: (1 - lam) * (q ** 2) + lam * (q**2 -1)

    
# Create the step function and vectorized version for naive HMC


def make_cd_leapfrog_step(T, V, A_ansatz, theta, lam, lam_next, dot_lam, dot_lam_next):
    p_update = make_p_update(V)
    x_update = make_x_update(T)
    dA_dq_scalar = lambda θ: jax.grad(A_ansatz(θ), argnums=0)
    dA_dp_scalar = lambda θ: jax.grad(A_ansatz(θ), argnums=1)
    def cd_leapfrog(q, p, eps):
        p_half = p - 0.5 * eps * (jax.grad(V)(q) + dot_lam * dA_dq_scalar(theta)(q, p))
        q_new = q + eps * (jax.grad(T)(p_half) + dot_lam * dA_dp_scalar(theta)(q, p_half))
        p_new = p_half - 0.5 * eps * (jax.grad(V)(q_new) + dot_lam * dA_dq_scalar(theta)(q_new, p_half))
        return q_new, p_new
    return cd_leapfrog


# =============================================================================
# 6) SImuLATION: NAÏVE HMC VS CD WITH ONLINE FITTING
# =============================================================================
def run_simulation(M, N_steps, delta_t, eps, momentum_refresh_interval, make_T, make_V, A_ansatz, lam_fn, dot_lam_fn, fit_gauge_potential, model):
    key = jax.random.PRNGKey(0)
    key, sub = jax.random.split(key)
    q_naive = jax.random.normal(sub, (M,))
    key, sub = jax.random.split(key)
    p_naive = jax.random.normal(sub, (M,)) * jnp.sqrt(m)

    q_cd = q_naive.copy()
    p_cd = p_naive.copy()

    theta = model

    loss_histories = []
    snapshots = {'naive': [], 'cd': [], 'lam': []}

    for k in range(N_steps + 1):
        t_k = k * delta_t
        lam_k = float(lam_fn(t_k))
        dot_lam_k = float(dot_lam_fn(t_k))

        # Record histograms every 10 steps
        if k % 10 == 0:
            snapshots['naive'].append(np.array(q_naive))
            snapshots['cd'].append(np.array(q_cd))
            snapshots['lam'].append(lam_k)

        # Re-fit A every 10 steps
        if (k % 10 == 0) and (k < N_steps):
            samples = np.stack([np.array(q_cd), np.array(p_cd)], axis=1)
            theta, loss_history = fit_gauge_potential(lam_k, samples, init_params=theta,
                                        make_T=make_T, make_V=make_V,
                                        A_ansatz=A_ansatz,
                                        num_iters=1000, lr=0.01)
            loss_histories.append(loss_history)

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
        cd_step = jax.vmap(lambda q, p: make_cd_leapfrog_step(make_T(lam_k), make_V(lam_k), A_ansatz, theta, lam_k, lam_k1, dot_lam_k, dot_lam_k1)(q, p, eps))
        q_cd, p_cd = cd_step(q_cd, p_cd)

    return theta, snapshots, loss_histories

def plot_results(theta_history, snapshots, loss_histories, delta_t, make_V, lam_fn):
    fig, axes = plt.subplots(3, 6, figsize=(28, 14))
    times = np.arange(len(snapshots['naive'])) * delta_t * 10  # *10 because we record every 10 steps
    axes = axes.flatten()
    
    # Plot loss histories
    axes[0].set_title("Loss during optimization")
    axes[0].set_xlabel("Optimization iteration")
    axes[0].set_ylabel("Loss")
    for i, loss_history in enumerate(loss_histories):
        axes[0].plot(loss_history, label=f'Fit {i+1}')
    axes[0].legend()
    # axes[0].set_yscale('log')

    num_hist_axes = 15
    num_snaps = len(snapshots['naive'])
    if num_snaps > num_hist_axes:
        selected_indices = np.linspace(0, num_snaps - 1, num_hist_axes, dtype=int)
    else:
        selected_indices = np.arange(num_snaps)

    for plot_idx, snap_idx in enumerate(selected_indices):
        ax = axes[plot_idx + 1]  # +1 because we only have one plot at the start now
        naive_snap = snapshots['naive'][snap_idx]
        cd_snap = snapshots['cd'][snap_idx]
        lam_val = snapshots['lam'][snap_idx]

        sns.histplot(naive_snap, bins=50, stat='density',
                     color='C0', alpha=0.4, label='Naïve', ax=ax)
        sns.histplot(cd_snap, bins=50, stat='density',
                     color='C1', alpha=0.4, label='CD', ax=ax)
        xs = np.linspace(lam_val - 4, lam_val + 4, 400)
        rho = np.array(jax.vmap(lambda x: jnp.exp(-make_V(lam_val)(x)))(xs))
        rho /= np.trapz(rho, xs)
        ax.plot(xs, rho, 'r-', lw=2, label='True')
        ax.set_title(f"t={snap_idx*10*delta_t:.2f}, lam={lam_val:.2f}")
        ax.set_xlabel("q")
        ax.set_ylabel("Density")
        x_min = min(-2, lam_val - 4)
        x_max = max(5, lam_val + 4)
        ax.set_xlim(x_min, x_max)
        ax.legend()

    plt.tight_layout()
    plt.savefig("counterdiabatic.png")


class MLP(eqx.Module):
    layers: list

    def __init__(self, dims, key):
        # dims: list of layer sizes, e.g. [2, 64, 32, 1]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i+1], key=keys[i])
            for i in range(len(dims) - 1)
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = (layer)(x)
            x = jax.nn.relu(x)
        x = (self.layers[-1])(x)
        return x

def A_ansatz(params):
    def A(q, p):
        # Stack q and p for input to MLP
        inputs = jnp.stack([q, p], axis=-1)
        # Get MLP output
        output = params(inputs)
        return output.squeeze()
    return A

def main():
    # Define all routines and parameters here
    M = 3000
    N_steps = 100
    eps = 0.05
    delta_t = eps # should this even be a parameter?
    momentum_refresh_interval = 20
    v = 0.5
    max_lam = 1.0
    lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
    dot_lam_fn = jax.grad(lam_fn)
    def make_T(lam):
        return lambda p: 0.5 * (p ** 2) / m
    def make_V(lam):
        # return lambda q: (1-lam)*0.5*(q**2) + lam*(q**2 - 3)**2
        # return lambda q: 0.5*((q-lam)**2 -1)**2
        return lambda q: 0.5 * (q - lam) ** 2
    
    # Initialize MLP with appropriate architecture
    key = jax.random.PRNGKey(0)
    mlp = MLP([2, 64, 32, 1], key)  # 2 inputs (q,p), 2 hidden layers, 1 output
    
    theta_history, snapshots, loss_histories = run_simulation(
        M, N_steps, delta_t, eps, momentum_refresh_interval,
        make_T, make_V, A_ansatz, lam_fn, dot_lam_fn, fit_gauge_potential, mlp
    )
    plot_results(theta_history, snapshots, loss_histories, delta_t, make_V, lam_fn)

if __name__ == '__main__':
    main()