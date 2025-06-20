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
def fit_gauge_potential(lam, samples, make_T, make_V, A_ansatz, num_iters=200, lr=0.01):
    """
    Fit A(q,p; θ) by minimizing mean_{samples}[ ( {A,H} - ∂H/∂μ )^2 ].
    Returns both the optimized parameters and the loss history.
    """


    qp_batch = jnp.array(samples)  # shape (N,2)
    # params = init_params

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


def make_cd_leapfrog_step(T, V, A_ansatz, lam, lam_next, dot_lam, dot_lam_next):
    # p_update = make_p_update(V)
    # x_update = make_x_update(T)
    dA_dq_scalar = jax.grad(A_ansatz, argnums=0)
    dA_dp_scalar = jax.grad(A_ansatz, argnums=1)
    def cd_leapfrog(q, p, eps):
        p_half = p - 0.5 * eps * (jax.grad(V)(q) + dot_lam * dA_dq_scalar(q, p))
        q_new = q + eps * (jax.grad(T)(p_half) + dot_lam * dA_dp_scalar(q, p_half))
        p_new = p_half - 0.5 * eps * (jax.grad(V)(q_new) + dot_lam * dA_dq_scalar(q_new, p_half))
        return q_new, p_new
    return cd_leapfrog


# =============================================================================
# 6) SIMULATION: NAÏVE HMC VS CD WITH ONLINE FITTING
# =============================================================================
def generate_initial_samples(M, make_T, make_V, lam, key, num_steps=5000, eps=0.05):
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

def run_simulation(M, N_steps, delta_t, eps, momentum_refresh_interval, make_T, make_V, A_ansatz, lam_fn, dot_lam_fn, key):
    # key = jax.random.PRNGKey(0)

    # print(A_ansatz(model.params), "what?")
    # raise Exception("stop")
    
    # Generate initial samples from the correct distribution
    initial_lam = float(lam_fn(0.0))
    q_naive, p_naive = generate_initial_samples(M, make_T, make_V, initial_lam, key)
    q_cd = q_naive.copy()
    p_cd = p_naive.copy()

    # theta = model

    loss_histories = []
    snapshots = {'naive': [], 'cd': [], 'lam': []}
    theta_history = []  # Track parameter history

    for k in range(N_steps + 1):
        t_k = k * delta_t
        lam_k = float(lam_fn(t_k))
        dot_lam_k = float(dot_lam_fn(t_k))


        # Re-fit A every 1 steps
        # print(A_ansatz(A.params), "what 1?")
        if not isinstance(A_ansatz, AnalyticAnsatz) and (k % 1 == 0) and (k < N_steps):
            samples = np.stack([np.array(q_cd), np.array(p_cd)], axis=1)
            A_ansatz, loss_history = fit_gauge_potential(lam_k, samples,
                                        make_T=make_T, make_V=make_V,
                                        A_ansatz=A_ansatz,
                                        num_iters=2000, lr=1e-3)
            loss_histories.append(loss_history)

        # Record histograms every 10 steps
        if k % 10 == 0:
            snapshots['naive'].append(np.array(q_naive))
            snapshots['cd'].append(np.array(q_cd))
            snapshots['lam'].append(lam_k)
            # Record parameters
            if isinstance(A_ansatz, PolynomialAnsatz):
                theta_history.append(np.array(A_ansatz.params))
            elif isinstance(A_ansatz, NeuralNetworkAnsatz):
                # Store just the parameters as a tuple of arrays
                params = []
                for layer in A_ansatz.layers:
                    if isinstance(layer, eqx.nn.Linear):
                        params.append(np.array(layer.weight))
                        params.append(np.array(layer.bias))
                theta_history.append(tuple(params))

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

    return A_ansatz, snapshots, loss_histories, theta_history

def plot_learned_ansatz(ax, theta, ansatz, q_range=(-3, 3), p_range=(-3, 3), n_points=50):
    """Plot the learned ansatz function A(q,p) as a 2D surface.
    
    Args:
        ax: matplotlib axis to plot on
        theta: parameters for the ansatz. For analytic ansatz, this is None.
        ansatz: the ansatz object (either PolynomialAnsatz or NeuralNetworkAnsatz)
        q_range: tuple of (min_q, max_q)
        p_range: tuple of (min_p, max_p)
        n_points: number of points in each dimension for the grid
    """
    q = np.linspace(q_range[0], q_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    Q, P = np.meshgrid(q, p)
    
    # Create an ansatz instance with the parameters for the current timestep
    if isinstance(ansatz, PolynomialAnsatz):
        # Create a new ansatz object and update its parameters
        current_ansatz = eqx.tree_at(lambda m: m.params, ansatz, theta)
    elif isinstance(ansatz, NeuralNetworkAnsatz):
        # Create a new ansatz and update its parameters layer by layer
        current_ansatz = ansatz
        param_idx = 0
        for i, layer in enumerate(ansatz.layers):
            if isinstance(layer, eqx.nn.Linear):
                current_ansatz = eqx.tree_at(lambda m: m.layers[i].weight, current_ansatz, theta[param_idx])
                current_ansatz = eqx.tree_at(lambda m: m.layers[i].bias, current_ansatz, theta[param_idx + 1])
                param_idx += 2
    elif isinstance(ansatz, AnalyticAnsatz):
        current_ansatz = ansatz  # No parameters to update
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz.ansatz_type}")

    # Evaluate A(q,p) at each point
    A_values = np.zeros_like(Q)
    for i in range(n_points):
        for j in range(n_points):
            A_values[i,j] = float(current_ansatz(Q[i,j], P[i,j]))
    
    # Plot the surface
    im = ax.imshow(A_values, extent=[q_range[0], q_range[1], p_range[0], p_range[1]], 
                   origin='lower', aspect='auto', cmap='RdBu')
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    plt.colorbar(im, ax=ax, label='A(q,p)')

def plot_results(snapshots, loss_histories, delta_t, make_V, lam_fn, param_history=None, ansatz=None):
    # Create two figures: one for distributions and one for the learned ansatz
    fig1, axes1 = plt.subplots(3, 6, figsize=(28, 14))
    fig2, axes2 = plt.subplots(3, 6, figsize=(28, 14))
    times = np.arange(len(snapshots['naive'])) * delta_t * 10  # *10 because we record every 10 steps
    axes1 = axes1.flatten()
    axes2 = axes2.flatten()
    
    # Plot loss histories
    axes1[0].set_title("Loss during optimization")
    axes1[0].set_xlabel("Optimization iteration")
    axes1[0].set_ylabel("Loss")
    for i, loss_history in enumerate(loss_histories):
        axes1[0].plot(loss_history, label=f'Fit {i+1}')
    axes1[0].legend()

    # Plot parameter history if available (only for polynomial ansatz)
    if param_history is not None and len(param_history) > 0 and isinstance(ansatz, PolynomialAnsatz):
        param_times = np.arange(len(param_history)) * delta_t * 10
        axes1[1].set_title("Learned parameters over time")
        axes1[1].set_xlabel("t")
        axes1[1].set_ylabel("θ value")
        
        # Get term descriptions for legend labels
        term_descriptions = ansatz.get_term_description()
        term_labels = []
        for desc in term_descriptions:
            # Extract just the term part (e.g., "pq", "q²", etc.)
            term_part = desc.split(": ")[1]
            term_labels.append(term_part)
        
        for i in range(param_history[0].shape[0]):
            axes1[1].plot(param_times, [p[i] for p in param_history], label=term_labels[i])
        axes1[1].legend()

    num_hist_axes = 13  # Reduced to make room for parameter plot
    num_snaps = len(snapshots['naive'])
    if num_snaps > num_hist_axes:
        selected_indices = np.linspace(0, num_snaps - 1, num_hist_axes, dtype=int)
    else:
        selected_indices = np.arange(num_snaps)

    # Find global min and max for consistent x-axis
    all_qs = np.concatenate([snapshots['naive'][i] for i in selected_indices] + 
                           [snapshots['cd'][i] for i in selected_indices])
    x_min = np.min(all_qs) - 0.5
    x_max = np.max(all_qs) + 0.5

    for plot_idx, snap_idx in enumerate(selected_indices):
        # Plot distributions
        ax1 = axes1[plot_idx + 2]  # +2 because we have loss and parameter plots at the start
        naive_snap = snapshots['naive'][snap_idx]
        cd_snap = snapshots['cd'][snap_idx]
        lam_val = snapshots['lam'][snap_idx]

        sns.histplot(naive_snap, bins=50, stat='density',
                     color='C0', alpha=0.4, label='Naïve', ax=ax1)
        sns.histplot(cd_snap, bins=50, stat='density',
                     color='C1', alpha=0.4, label='CD', ax=ax1)
        xs = np.linspace(x_min, x_max, 400)
        rho = np.array(jax.vmap(lambda x: jnp.exp(-make_V(lam_val)(x)))(xs))
        rho /= np.trapezoid(rho, xs)
        ax1.plot(xs, rho, 'r-', lw=2, label='True')
        ax1.set_title(f"t={snap_idx*10*delta_t:.2f}, lam={lam_val:.2f}")
        ax1.set_xlabel("q")
        ax1.set_ylabel("Density")
        ax1.set_xlim(x_min, x_max)
        ax1.legend()

        # Plot learned ansatz
        ax2 = axes2[plot_idx + 2]
        if isinstance(ansatz, AnalyticAnsatz):
            plot_learned_ansatz(ax2, None, ansatz, q_range=(x_min, x_max), p_range=(-3, 3))
            ax2.set_title(f"Analytic A(q,p) at t={snap_idx*10*delta_t:.2f}")

        elif param_history is not None and snap_idx < len(param_history):
            theta = param_history[snap_idx]
            plot_learned_ansatz(ax2, theta, ansatz, q_range=(x_min, x_max), p_range=(-3, 3))
            ax2.set_title(f"Learned A(q,p) at t={snap_idx*10*delta_t:.2f}")

    if isinstance(ansatz, PolynomialAnsatz):
        ansatz_type = 'polynomial'
    elif isinstance(ansatz, NeuralNetworkAnsatz):
        ansatz_type = 'neural_network'
    elif isinstance(ansatz, AnalyticAnsatz):
        ansatz_type = 'analytic'
    else:
        raise ValueError(f"Unknown ansatz type")

    plt.figure(fig1.number)
    plt.tight_layout()
    plt.savefig(f"counterdiabatic_distributions_{ansatz_type}.png")
    
    plt.figure(fig2.number)
    plt.tight_layout()
    plt.savefig(f"counterdiabatic_ansatz_{ansatz_type}.png")


class A_ansatz(eqx.Module):
    """Base class for gauge potential ansatz."""
    def __call__(self, q, p):
        raise NotImplementedError

def generate_polynomial_terms(max_degree):
    """Generate all polynomial terms up to max_degree in p and q.
    
    Returns a list of tuples (coeff_name, q_power, p_power) representing terms like:
    - (θ1, 0, 1) for p
    - (θ2, 1, 1) for p*q
    - (θ3, 2, 0) for q^2
    etc.
    """
    terms = []
    term_idx = 1
    
    for total_degree in range(max_degree + 1):
        for q_power in range(total_degree + 1):
            p_power = total_degree - q_power
            terms.append((f"θ{term_idx}", q_power, p_power))
            term_idx += 1
    
    return terms

max_degree = 4

class PolynomialAnsatz(A_ansatz):
    """Polynomial ansatz for the gauge potential."""
    params: jnp.ndarray
    # terms: list

    def __init__(self):
        # self.terms = 
        # Initialize parameters to zero
        self.params = jnp.zeros(len(generate_polynomial_terms(max_degree)))

    def __call__(self, q, p):
        result = 0.0
        for i, (_, q_power, p_power) in enumerate(generate_polynomial_terms(max_degree)):
            result += self.params[i] * (q ** q_power) * (p ** p_power)
        return result

    def get_term_description(self):
        """Return a description of what each parameter represents."""
        descriptions = []
        for coeff_name, q_power, p_power in generate_polynomial_terms(max_degree):
            term_str = ""
            if q_power > 0:
                term_str += f"q^{q_power}" if q_power > 1 else "q"
            if p_power > 0:
                term_str += f"p^{p_power}" if p_power > 1 else "p"
            if not term_str:
                term_str = "1"
            descriptions.append(f"{coeff_name}: {term_str}")
        return descriptions

class NeuralNetworkAnsatz(A_ansatz):
    """Neural network ansatz for the gauge potential."""
    layers: list

    def __init__(self, dims, key):
        # dims: list of layer sizes, e.g. [2, 64, 32, 1]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i+1], key=keys[i])
            for i in range(len(dims) - 1)
        ]

    def __call__(self, q, p):
        # Stack q and p for input to MLP
        x = jnp.stack([q, p], axis=-1)
        for layer in self.layers[:-1]:
            x = (layer)(x)
            x = jax.nn.relu(x)
        x = (self.layers[-1])(x)
        return x.squeeze()

class AnalyticAnsatz(A_ansatz):
    """An ansatz with a fixed analytical form from a screenshot."""
    sigma: float
    ansatz_type: str

    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.ansatz_type = 'analytic'

    def __call__(self, q, p):
        # A(q, p) = - (sigma * p^2 + q^2 / sigma) / 2 * arctan(p * sigma / q)
        # Use arctan2 to handle q=0 case
        return p
        # return - (self.sigma * p**2 + q**2 / self.sigma) / 2 * jnp.arctan2(p * self.sigma, q)

def main():

    # todos
    # 1. see the function that gets learned
    # 2. understand delta_t vs eps

    # Define all routines and parameters here
    M = 3000
    N_steps = 20
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
        # return lambda q: 0.5 * (q - lam) ** 2
        return lambda q: 0.5 * (lam + 0.1) * (q ** 2)
    
    # Initialize ansatz (either neural network or polynomial)
    key = jax.random.PRNGKey(0)
    # For neural network:
    # d = 1
    # ansatz = NeuralNetworkAnsatz([2*d, 128, 256, 128, d], key)
    # ansatz_type = 'neural_network'
    # For polynomial:
    ansatz = PolynomialAnsatz()
    ansatz_type = 'polynomial'
    # For analytic solution:
    # ansatz = AnalyticAnsatz(sigma=1.0)
    # ansatz_type = 'analytic'
    
    # Print polynomial terms if using polynomial ansatz
    if ansatz_type == 'polynomial':
        print("Polynomial terms:")
        for desc in ansatz.get_term_description():
            print(f"  {desc}")
        print(f"Total number of parameters: {len(ansatz.params)}")
    
    A_ansatz, snapshots, loss_histories, param_history = run_simulation(
        M=M, 
        N_steps=N_steps, 
        delta_t=delta_t, 
        eps=eps, 
        momentum_refresh_interval=momentum_refresh_interval,
        make_T=make_T, 
        make_V=make_V, 
        lam_fn=lam_fn, 
        dot_lam_fn=dot_lam_fn, 
        A_ansatz=ansatz, 
        key=key
    )
    plot_results(snapshots, loss_histories, delta_t, make_V, lam_fn, param_history, A_ansatz)

if __name__ == '__main__':
    main()