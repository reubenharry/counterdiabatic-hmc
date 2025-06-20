import jax
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import equinox as eqx
import jax
import jax.numpy as jnp
import os

from .ansatze import PolynomialAnsatz, NeuralNetworkAnsatz, AnalyticAnsatz

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
    if ansatz.ansatz_type == 'polynomial':
        # Create a new ansatz object and update its parameters
        current_ansatz = eqx.tree_at(lambda m: m.params, ansatz, theta)
    elif ansatz.ansatz_type == 'neural':
        # Create a new ansatz and update its parameters layer by layer
        current_ansatz = ansatz
        param_idx = 0
        for i, layer in enumerate(ansatz.layers):
            if isinstance(layer, eqx.nn.Linear):
                current_ansatz = eqx.tree_at(lambda m: m.layers[i].weight, current_ansatz, theta[param_idx])
                current_ansatz = eqx.tree_at(lambda m: m.layers[i].bias, current_ansatz, theta[param_idx + 1])
                param_idx += 2
    elif ansatz.ansatz_type == 'analytic':
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

def plot_results(snapshots, loss_histories, delta_t, make_V, lam_fn, param_history=None, ansatz=None, potential_name="harmonic"):
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    # Determine ansatz type and create subdirectory
    if isinstance(ansatz, PolynomialAnsatz):
        ansatz_type = 'polynomial'
    elif isinstance(ansatz, NeuralNetworkAnsatz):
        ansatz_type = 'neural_network'
    elif isinstance(ansatz, AnalyticAnsatz):
        ansatz_type = 'analytic'
    else:
        raise ValueError(f"Unknown ansatz type")
    
    ansatz_dir = f"figures/{ansatz_type}"
    os.makedirs(ansatz_dir, exist_ok=True)
    
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

    # Save figures with potential information in filename
    plt.figure(fig1.number)
    plt.tight_layout()
    plt.savefig(f"{ansatz_dir}/distributions_{potential_name}.png", dpi=300, bbox_inches='tight')
    
    plt.figure(fig2.number)
    plt.tight_layout()
    plt.savefig(f"{ansatz_dir}/ansatz_{potential_name}.png", dpi=300, bbox_inches='tight') 