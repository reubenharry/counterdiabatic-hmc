import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import equinox as eqx
import os
from scipy.stats import gaussian_kde

from .ansatze import PolynomialAnsatz, NeuralNetworkAnsatz, AnalyticAnsatz

# Plotting constants
PLOTTING_CONSTANTS = {
    'HISTOGRAM_BINS': 25,
    'GRID_POINTS': 200,
    'DENSE_GRID_POINTS': 1000,
    'ALPHA': 0.6,
    'RIDGE_HEIGHT': 1.8,
    'FIGURE_SIZE_SINGLE': (8, 6),
    'FIGURE_SIZE_DOUBLE': (12, 6),
    'FIGURE_SIZE_DISTRIBUTIONS': (20, 5),
    'DISTRIBUTIONS_COLS': 4
}

def plot_learned_ansatz(ax, theta, ansatz, q_range=(-3, 3), p_range=(-3, 3), n_points=None, dim=1):
    if n_points is None:
        n_points = PLOTTING_CONSTANTS['HISTOGRAM_BINS']
    """Plot the learned ansatz function A(q,p) as a 2D surface.
    
    Args:
        ax: matplotlib axis to plot on
        theta: parameters for the ansatz. For analytic ansatz, this is None.
        ansatz: the ansatz object (either PolynomialAnsatz or NeuralNetworkAnsatz)
        q_range: tuple of (min_q, max_q)
        p_range: tuple of (min_p, max_p)
        n_points: number of points in each dimension for the grid
        dim: dimension of the system
    """
    if dim == 1:
        # 1D case: plot A(q,p) as 2D surface
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
            # For the analytic ansatz, theta contains the lambda value for this timestep.
            current_ansatz = eqx.tree_at(lambda m: m.params, ansatz, theta)
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz.ansatz_type}")

        # Evaluate A(q,p) at each point - vectorized for speed
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
    
    elif dim == 2:
        # 2D case: plot A(q,p) as a function of q_0, q_1 with fixed p
        q0 = np.linspace(q_range[0], q_range[1], n_points)
        q1 = np.linspace(q_range[0], q_range[1], n_points)
        Q0, Q1 = np.meshgrid(q0, q1)
        
        # Use fixed p = [0, 0] for visualization
        p_fixed = jnp.array([0.0, 0.0])
        
        # Create an ansatz instance with the parameters for the current timestep
        if ansatz.ansatz_type == 'polynomial':
            current_ansatz = eqx.tree_at(lambda m: m.params, ansatz, theta)
        elif ansatz.ansatz_type == 'neural':
            current_ansatz = ansatz
            param_idx = 0
            for i, layer in enumerate(ansatz.layers):
                if isinstance(layer, eqx.nn.Linear):
                    current_ansatz = eqx.tree_at(lambda m: m.layers[i].weight, current_ansatz, theta[param_idx])
                    current_ansatz = eqx.tree_at(lambda m: m.layers[i].bias, current_ansatz, theta[param_idx + 1])
                    param_idx += 2
        elif ansatz.ansatz_type == 'analytic':
            current_ansatz = eqx.tree_at(lambda m: m.params, ansatz, theta)
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz.ansatz_type}")

        # Evaluate A(q,p) at each point
        A_values = np.zeros_like(Q0)
        for i in range(n_points):
            for j in range(n_points):
                q = jnp.array([Q0[i,j], Q1[i,j]])
                A_values[i,j] = float(current_ansatz(q, p_fixed))
        
        # Plot the surface
        im = ax.imshow(A_values, extent=[q_range[0], q_range[1], q_range[0], q_range[1]], 
                       origin='lower', aspect='auto', cmap='RdBu')
        ax.set_xlabel('q_0')
        ax.set_ylabel('q_1')
        plt.colorbar(im, ax=ax, label='A(q,p=0)')

def plot_2d_distribution(ax, samples, title, color='blue', alpha=0.6):
    """Plot 2D samples as a scatter plot."""
    ax.scatter(samples[:, 0], samples[:, 1], alpha=alpha, s=1, color=color)
    ax.set_xlabel('q_0')
    ax.set_ylabel('q_1')
    ax.set_title(title)
    ax.set_aspect('equal')

def compute_weighted_kde(samples, weights=None, x_grid=None):
    """Compute weighted KDE for 1D samples.
    
    Args:
        samples: 1D array of samples
        weights: Optional array of weights (log weights)
        x_grid: Optional grid for evaluation
    
    Returns:
        density: KDE values on x_grid
    """
    if x_grid is None:
        x_grid = np.linspace(np.min(samples) - 0.5, np.max(samples) + 0.5, 200)
    
    if weights is not None:
        # Convert log weights to regular weights
        weights = np.exp(weights - np.max(weights))  # Subtract max for numerical stability
        weights = weights / np.sum(weights)  # Normalize
        
        # Check if weights are effectively uniform (all equal)
        if np.allclose(weights, weights[0]):
            # Use unweighted KDE for uniform weights to ensure consistency
            try:
                kde = gaussian_kde(samples.flatten())
                density = kde(x_grid)
            except:
                # Fallback to histogram if KDE fails
                hist, bin_edges = np.histogram(samples, bins=25, density=True, 
                                              range=(np.min(x_grid), np.max(x_grid)))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                density = np.interp(x_grid, bin_centers, hist)
        else:
            # Use weighted KDE for non-uniform weights
            try:
                kde = gaussian_kde(samples.flatten(), weights=weights)
                density = kde(x_grid)
            except:
                # Fallback to weighted histogram if KDE fails
                hist, bin_edges = np.histogram(samples, bins=25, weights=weights, density=True, 
                                              range=(np.min(x_grid), np.max(x_grid)))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                density = np.interp(x_grid, bin_centers, hist)
    else:
        # Unweighted KDE
        try:
            kde = gaussian_kde(samples.flatten())
            density = kde(x_grid)
        except:
            # Fallback to histogram if KDE fails
            hist, bin_edges = np.histogram(samples, bins=25, density=True, 
                                          range=(np.min(x_grid), np.max(x_grid)))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            density = np.interp(x_grid, bin_centers, hist)
    
    return density

def create_ridge_plot(snapshots, delta_t, make_V, potential_name="harmonic", ansatz_type="polynomial"):
    """Create a ridge plot showing the evolution of 1D distributions over time.
    
    Args:
        snapshots: Dictionary containing 'naive', 'naive_weighted', 'cd', 'cd_post_equil', 'lam' arrays
        delta_t: Time step
        make_V: Function to create potential energy
        lam_fn: Function to compute lambda at a given time
        potential_name: Name of the potential for filename
        ansatz_type: Type of ansatz for directory structure
    """
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    ansatz_dir = f"figures/{ansatz_type}"
    os.makedirs(ansatz_dir, exist_ok=True)
    
    # Check if re-equilibration was used (simplified - no post-equilibration for now)
    has_re_equil = False
    
    # Check if weights are available
    has_weights = 'weights' in snapshots and any(w is not None for w in snapshots['weights'])
    
    # Get time points from snapshots
    if 'times' in snapshots and len(snapshots['times']) > 0:
        times = snapshots['times']
    else:
        times = np.arange(len(snapshots['particles'])) * delta_t  # Fallback
    
    # For post-equilibration snapshots, the timing is different
    # They represent the state after CD step + re-equilibration, so they should be plotted
    # at the time after the CD step (i.e., at the next timestep)
    if has_re_equil:
        # Post-equilibration snapshots are stored at the end of timesteps
        # So they should be plotted at the next timestep's time
        post_equil_times = np.arange(1, len(snapshots['cd_post_equil']) + 1) * delta_t
    else:
        post_equil_times = np.array([])
    
    # Create figure with appropriate layout (simplified - only CD-HMC)
    if has_weights:
        # Two columns: CD unweighted, CD weighted
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Create secondary y-axes for lambda values
        ax1_lambda = ax1.twinx()
        ax2_lambda = ax2.twinx()
    else:
        # One column: CD only
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        
        # Create secondary y-axes for lambda values
        ax1_lambda = ax1.twinx()
    
    # Set up axis titles based on layout (simplified - only CD-HMC)
    if has_weights:
        # Two columns: CD unweighted, CD weighted
        ax1.set_title("Counterdiabatic HMC Evolution (Unweighted)", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Position q", fontsize=12)
        ax1.set_ylabel("Time t", fontsize=12)
        ax1_lambda.set_ylabel("λ", fontsize=12, color='red')
        ax1_lambda.tick_params(axis='y', labelcolor='red')
        
        ax2.set_title("Counterdiabatic HMC Evolution (Weighted)", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Position q", fontsize=12)
        ax2.set_ylabel("Time t", fontsize=12)
        ax2_lambda.set_ylabel("λ", fontsize=12, color='red')
        ax2_lambda.tick_params(axis='y', labelcolor='red')
    else:
        # One column: CD only
        ax1.set_title("Counterdiabatic HMC Evolution", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Position q", fontsize=12)
        ax1.set_ylabel("Time t", fontsize=12)
        ax1_lambda.set_ylabel("λ", fontsize=12, color='red')
        ax1_lambda.tick_params(axis='y', labelcolor='red')
    
    # Find global range for consistent x-axis
    all_qs = np.concatenate(snapshots['particles']).flatten()
    x_min = np.min(all_qs) - 0.5
    x_max = np.max(all_qs) + 0.5
    
    # Create x grid for smooth curves
    x_grid = np.linspace(x_min, x_max, 200)
    
    # Note: Naive HMC plotting removed since naive HMC is not performed in run_simulation
    
    # Note: Naive HMC weighted plotting removed since naive HMC is not performed in run_simulation
    
    # Plot CD HMC distributions
    cd_ax = ax2 if has_weights else ax1  # Use appropriate axis based on layout
    for i, (t, cd_snap, lam_val) in enumerate(zip(times, snapshots['particles'], snapshots['lam'])):
        # Compute KDE for smooth curve
        density = compute_weighted_kde(cd_snap.flatten(), weights=None, x_grid=x_grid)
        
        # Normalize and offset for ridge plot - increased height for more overlap
        density = density / np.max(density) * 1.8  # Increased from 1.2 to 1.8 for more overlap
        offset = t * 2.0  # Increased spacing between plots to reduce overlap
        
        # Plot the ridge with transparency for overlap
        cd_ax.fill_between(x_grid, offset, offset + density, 
                        color='red', alpha=0.4, edgecolor='red', linewidth=0.5)
        
        # Add true distribution at each time step
        potential_fn = make_V(lam_val)
        rho = np.array(jax.vmap(lambda x: jnp.exp(-potential_fn(x)))(x_grid))
        rho = rho / np.max(rho) * 1.8  # Scale to match
        cd_ax.plot(x_grid, offset + rho, 'k--', linewidth=1.5, alpha=0.8)
    
    # Note: Duplicate CD plotting section removed
    
    # Set consistent limits (simplified - only CD-HMC)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(times[0] * 2.0 - 0.1, times[-1] * 2.0 + 2.0)  # Adjusted for increased spacing
    
    if has_weights:
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(times[0] * 2.0 - 0.1, times[-1] * 2.0 + 2.0)  # Adjusted for increased spacing
    
    # Set y-axis ticks only at the time points where distributions are plotted (simplified)
    ax1.set_yticks(times * 2.0)
    
    if has_weights:
        ax2.set_yticks(times * 2.0)
    
    # Configure lambda axes (secondary y-axes)
    # Get lambda values for the time points
    lambda_values = snapshots['lam']
    
    # Configure lambda axes based on layout
    ax1_lambda.set_ylim(ax1.get_ylim())  # Same limits as time axis
    ax1_lambda.set_yticks(times * 2.0)
    lambda_tick_labels = [f"{lam:.3f}" for lam in lambda_values]
    ax1_lambda.set_yticklabels(lambda_tick_labels)
    
    if has_weights:
        ax2_lambda.set_ylim(ax2.get_ylim())  # Same limits as time axis
        ax2_lambda.set_yticks(times * 2.0)
        ax2_lambda.set_yticklabels(lambda_tick_labels)
    
    # Note: Removed ax3_lambda configuration since we only have 2 axes now
    
    # Add legends with true distribution reference
    ax1.plot([], [], 'k--', linewidth=1.5, label='True distribution')
    ax1.legend(loc='upper right')
    
    if has_weights:
        ax2.plot([], [], 'k--', linewidth=1.5, label='True distribution')
        ax2.legend(loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{ansatz_dir}/ridge_plot_{potential_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_overlay_ridge_plot(snapshots, delta_t, make_V, potential_name="harmonic", ansatz_type="polynomial"):
    """Create a ridge plot showing both CD weighted and unweighted distributions overlaid.
    
    Args:
        snapshots: Dictionary containing 'particles', 'weights', 'lam' arrays
        delta_t: Time step
        make_V: Function to create potential energy
        potential_name: Name of the potential for filename
        ansatz_type: Type of ansatz for directory structure
    """
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    ansatz_dir = f"figures/{ansatz_type}"
    os.makedirs(ansatz_dir, exist_ok=True)
    
    # Check if weights are available
    has_weights = 'weights' in snapshots and any(w is not None for w in snapshots['weights'])
    
    if not has_weights:
        print("Warning: No weights found in snapshots. Creating regular ridge plot instead.")
        create_ridge_plot(snapshots, delta_t, make_V, potential_name, ansatz_type)
        return
    
    # Get time points from snapshots
    if 'times' in snapshots and len(snapshots['times']) > 0:
        times = np.array(snapshots['times'])  # Convert to numpy array
    else:
        times = np.arange(len(snapshots['particles'])) * delta_t  # Fallback
    
    # Create single figure for overlay
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create secondary y-axis for lambda values
    ax_lambda = ax.twinx()
    
    # Set up axis titles
    ax.set_title("Counterdiabatic HMC Evolution (Weighted vs Unweighted)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Position q", fontsize=12)
    ax.set_ylabel("Time t", fontsize=12)
    ax_lambda.set_ylabel("λ", fontsize=12, color='red')
    ax_lambda.tick_params(axis='y', labelcolor='red')
    
    # Find global range for consistent x-axis
    all_qs = np.concatenate(snapshots['particles']).flatten()
    x_min = np.min(all_qs) - 0.5
    x_max = np.max(all_qs) + 0.5
    
    # Create x grid for smooth curves
    x_grid = np.linspace(x_min, x_max, 200)
    
    # Plot both weighted and unweighted distributions
    for i, (t, cd_snap, lam_val, weights) in enumerate(zip(times, snapshots['particles'], snapshots['lam'], snapshots['weights'])):
        offset = t * 2.0  # Spacing between plots
        
        # Plot unweighted distribution (red, more transparent)
        density_unweighted = compute_weighted_kde(cd_snap.flatten(), weights=None, x_grid=x_grid)
        density_unweighted = density_unweighted / np.max(density_unweighted) * 1.5
        ax.fill_between(x_grid, offset, offset + density_unweighted, 
                       color='red', alpha=0.4, edgecolor='red', linewidth=1.0, label='Unweighted' if i == 0 else "")
        
        # Plot weighted distribution (blue, more opaque)
        if weights is not None:
            # Convert log weights to regular weights
            weights_array = np.exp(weights - np.max(weights))  # Normalize to prevent overflow
            density_weighted = compute_weighted_kde(cd_snap.flatten(), weights=weights_array, x_grid=x_grid)
            density_weighted = density_weighted / np.max(density_weighted) * 1.5
            ax.fill_between(x_grid, offset, offset + density_weighted, 
                           color='blue', alpha=0.8, edgecolor='blue', linewidth=1.5, label='Weighted' if i == 0 else "")
            
            # Also plot the difference as a separate line
            diff = density_weighted - density_unweighted
            ax.plot(x_grid, offset + diff + 0.2, 'g-', linewidth=1.0, alpha=0.7, label='Difference' if i == 0 else "")
        
        # Add true distribution at each time step (black dashed)
        potential_fn = make_V(lam_val)
        rho = np.array(jax.vmap(lambda x: jnp.exp(-potential_fn(x)))(x_grid))
        rho = rho / np.max(rho) * 1.5  # Scale to match
        ax.plot(x_grid, offset + rho, 'k--', linewidth=1.5, alpha=0.8, label='True distribution' if i == 0 else "")
    
    # Set consistent limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(times[0] * 2.0 - 0.1, times[-1] * 2.0 + 2.0)
    
    # Set y-axis ticks
    ax.set_yticks(times * 2.0)
    
    # Configure lambda axis (secondary y-axis)
    lambda_values = snapshots['lam']
    ax_lambda.set_ylim(ax.get_ylim())
    ax_lambda.set_yticks(times * 2.0)
    lambda_tick_labels = [f"{lam:.3f}" for lam in lambda_values]
    ax_lambda.set_yticklabels(lambda_tick_labels)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{ansatz_dir}/ridge_plot_{potential_name}_overlay.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_results(snapshots, loss_histories, delta_t, make_V, param_history=None, ansatz=None, potential_name="harmonic", dim=1, plot_ansatz=False, make_T=None, naive_snapshots=None):
    """Simplified plotting function that only creates two figures:
    1. Comparison ridge plot (handled in main.py)
    2. Distributions plot (this function)
    """
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
    
    # Debug information
    print(f"Plotting for ansatz type: {ansatz_type}")
    print(f"Number of snapshots: particles={len(snapshots.get('particles', []))}")
    print(f"Number of post-equilibration snapshots: 0")
    
    # Check if re-equilibration was used
    has_re_equil = 'cd_post_equil' in snapshots and len(snapshots['cd_post_equil']) > 0
    
    # Create distributions plot with diagnostic information
    create_distributions_plot(snapshots, delta_t, make_V, ansatz_dir, potential_name, dim, has_re_equil, 
                            loss_histories=loss_histories, param_history=param_history, make_T=make_T, naive_snapshots=naive_snapshots)
    
    print(f"Saved distributions plot to {ansatz_dir}/distributions_{potential_name}.png")


def create_distributions_plot(snapshots, delta_t, make_V, ansatz_dir, potential_name, dim, has_re_equil, loss_histories=None, param_history=None, make_T=None, naive_snapshots=None):
    """Create the distributions plot showing histograms and diagnostic plots."""
    # Collect all times from both CD and naive snapshots
    cd_times = snapshots['times']
    naive_times = []
    if naive_snapshots and 'times' in naive_snapshots:
        naive_times = naive_snapshots['times']
    elif naive_snapshots and 'particles' in naive_snapshots:
        # Use saved times from naive snapshots, or fail if not available
        if 'times' in naive_snapshots:
            naive_times = naive_snapshots['times']
        else:
            raise ValueError("Naive snapshots missing 'times' key - cannot plot without saved times")
    
    # Create union of all times and corresponding lambda values
    all_times = []
    all_lambda_values = []
    
    # Add CD times and lambda values
    for t, lam in zip(cd_times, snapshots['lam']):
        all_times.append(float(t))
        all_lambda_values.append(float(lam))
    
    # Add naive times and lambda values
    if naive_snapshots and 'lam' in naive_snapshots:
        for t, lam in zip(naive_times, naive_snapshots['lam']):
            all_times.append(float(t))
            all_lambda_values.append(float(lam))
    
    # Sort by time, keeping all unique time points
    sorted_pairs = sorted(zip(all_times, all_lambda_values))
    times = []
    lambda_values = []
    seen_times = set()
    
    for t, lam in sorted_pairs:
        if t not in seen_times:
            seen_times.add(t)
            times.append(t)
            lambda_values.append(lam)
    
    # Create figure with subplots for distributions and diagnostics
    num_snapshots = len(times)  # Use the union of times instead of just CD snapshots
    cols = 4
    rows_histograms = int(np.ceil(num_snapshots / cols))
    rows_diagnostics = 2  # For loss, energy stats, and parameter history
    total_rows = rows_histograms + rows_diagnostics
    
    fig = plt.figure(figsize=(20, 5*total_rows))
    
    # Create grid layout: histograms first, then diagnostics
    gs = fig.add_gridspec(total_rows, 4, height_ratios=[1]*rows_histograms + [1]*rows_diagnostics, width_ratios=[1, 1, 1, 1])
    
    # Check if this is a weighted simulation
    has_weights = 'weights' in snapshots and len(snapshots['weights']) > 0
    
    # FIRST: Plot all histograms at different time points
    for i, (time, lam_val) in enumerate(zip(times, lambda_values)):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        
        # Find CD data for this time
        cd_idx = None
        for j, cd_t in enumerate(cd_times):
            if abs(float(cd_t) - time) < 0.01:  # Exact match for CD
                cd_idx = j
                break
        
        # Plot CD particles distribution if available
        if cd_idx is not None and 'particles' in snapshots and cd_idx < len(snapshots['particles']):
            particles_snap = snapshots['particles'][cd_idx]
            if len(particles_snap) > 0:
                if has_weights and cd_idx is not None and cd_idx < len(snapshots['weights']):
                    # Use weighted histogram
                    weights = snapshots['weights'][cd_idx]
                    if weights is not None and not np.allclose(weights, 0.0):
                        # Convert log weights to regular weights
                        weights = np.exp(weights - np.max(weights))
                        weights = weights / np.sum(weights)
                        ax.hist(particles_snap.flatten(), bins=25, alpha=0.6, label='Particles (Weighted)', density=True, color='red', weights=weights)
                    else:
                        ax.hist(particles_snap.flatten(), bins=25, alpha=0.6, label='Particles', density=True, color='red')
                else:
                    ax.hist(particles_snap.flatten(), bins=25, alpha=0.6, label='Particles', density=True, color='red')
        
        # Plot naive HMC distribution if available
        if naive_snapshots and 'particles' in naive_snapshots:
            # Find the naive snapshot that corresponds to the current time
            naive_idx = None
            min_diff = float('inf')
            for j, naive_t in enumerate(naive_times):
                if abs(float(naive_t) - time) < min_diff:
                    min_diff = abs(float(naive_t) - time)
                    naive_idx = j
            
            # Only plot if we found a close match (within 0.001 time units to handle floating point errors)
            if naive_idx is not None and min_diff < 0.001:
                naive_snap = naive_snapshots['particles'][naive_idx]
                if len(naive_snap) > 0:
                    # Check if naive simulation also has weights
                    naive_has_weights = 'weights' in naive_snapshots and len(naive_snapshots['weights']) > 0
                    if naive_has_weights and naive_idx is not None and naive_idx < len(naive_snapshots['weights']):
                        weights = naive_snapshots['weights'][naive_idx]
                        if weights is not None and not np.allclose(weights, 0.0):
                            weights = np.exp(weights - np.max(weights))
                            weights = weights / np.sum(weights)
                            ax.hist(naive_snap.flatten(), bins=25, alpha=0.6, label='Naive HMC (Weighted)', density=True, color='blue', weights=weights)
                        else:
                            ax.hist(naive_snap.flatten(), bins=25, alpha=0.6, label='Naive HMC', density=True, color='blue')
                    else:
                        ax.hist(naive_snap.flatten(), bins=25, alpha=0.6, label='Naive HMC', density=True, color='blue')
        
        # Plot true distribution using the appropriate lambda value
        x_grid = np.linspace(-10, 10, 1000)
        potential_fn = make_V(lam_val)
        rho = np.array([np.exp(-potential_fn(x)) for x in x_grid])
        rho = rho / np.trapz(rho, x_grid)  # Normalize
        ax.plot(x_grid, rho, 'k--', linewidth=2, label='True distribution')
        
        ax.set_title(f't = {time:.2f}, λ = {lam_val:.3f}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # SECOND: Plot diagnostic plots after all histograms
    diagnostic_row_start = rows_histograms
    
    # Plot loss curves (first diagnostic row, left)
    if loss_histories and len(loss_histories) > 0:
        ax_loss = fig.add_subplot(gs[diagnostic_row_start, 0])
        for i, loss_history in enumerate(loss_histories):
            if len(loss_history) > 0:
                ax_loss.plot(loss_history, label=f'Step {i}', alpha=0.7)
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Loss History')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
    
    # Plot energy statistics (first diagnostic row, middle and right)
    if 'detailed_energy_stats' in snapshots and len(snapshots['detailed_energy_stats']) > 0:
        energy_stats = snapshots['detailed_energy_stats']
        
        # Use saved times for energy statistics
        if 'times' in snapshots:
            energy_times = snapshots['times']
        else:
            raise ValueError("Snapshots missing 'times' key - cannot plot energy statistics without saved times")
        
        # Plot <∂H/∂λ> over time
        ax_dH_dlam = fig.add_subplot(gs[diagnostic_row_start, 1])
        dH_dlam_vals = [stats['avg_dH_dlam'] for stats in energy_stats]
        ax_dH_dlam.plot(energy_times, dH_dlam_vals, 'r-', label='Current Method', linewidth=2)
        
        # Add naive HMC energy derivative if available
        if naive_snapshots and 'detailed_energy_stats' in naive_snapshots and len(naive_snapshots['detailed_energy_stats']) > 0:
            naive_energy_stats = naive_snapshots['detailed_energy_stats']
            if 'times' in naive_snapshots:
                naive_energy_times = naive_snapshots['times']
            else:
                raise ValueError("Naive snapshots missing 'times' key - cannot plot energy statistics")
            naive_dH_dlam_vals = [stats['avg_dH_dlam'] for stats in naive_energy_stats]
            ax_dH_dlam.plot(naive_energy_times, naive_dH_dlam_vals, 'b-', label='Naive HMC', linewidth=2)
        
        ax_dH_dlam.set_xlabel('Time')
        ax_dH_dlam.set_ylabel('<∂H/∂λ>')
        ax_dH_dlam.set_title('Energy Derivative')
        ax_dH_dlam.legend()
        ax_dH_dlam.grid(True, alpha=0.3)
        
        # Plot <ΔH²> over time
        ax_dH2 = fig.add_subplot(gs[diagnostic_row_start, 2])
        dH2_vals = [stats['avg_delta_H_sq'] for stats in energy_stats]
        ax_dH2.plot(energy_times, dH2_vals, 'r-', label='Current Method', linewidth=2)
        
        # Add naive HMC energy variance if available
        if naive_snapshots and 'detailed_energy_stats' in naive_snapshots and len(naive_snapshots['detailed_energy_stats']) > 0:
            naive_energy_stats = naive_snapshots['detailed_energy_stats']
            if 'times' in naive_snapshots:
                naive_energy_times = naive_snapshots['times']
            else:
                raise ValueError("Naive snapshots missing 'times' key - cannot plot energy statistics")
            naive_dH2_vals = [stats['avg_delta_H_sq'] for stats in naive_energy_stats]
            ax_dH2.plot(naive_energy_times, naive_dH2_vals, 'b-', label='Naive HMC', linewidth=2)
        
        ax_dH2.set_xlabel('Time')
        ax_dH2.set_ylabel('<ΔH²>')
        ax_dH2.set_title('Energy Change Variance')
        ax_dH2.legend()
        ax_dH2.grid(True, alpha=0.3)
        

        
        # Plot Var[A] over time (only for counterdiabatic)
        ax_var_A = fig.add_subplot(gs[diagnostic_row_start, 3])
        var_A_vals = [stats['var_A'] for stats in energy_stats]
        ax_var_A.plot(energy_times, var_A_vals, 'g-', label='Var[A]', linewidth=2)
        ax_var_A.set_xlabel('Time')
        ax_var_A.set_ylabel('Var[A]')
        ax_var_A.set_title('Gauge Potential Variance')
        ax_var_A.legend()
        ax_var_A.grid(True, alpha=0.3)
    
    # Plot parameter history if available (second diagnostic row, spanning all columns)
    if param_history and len(param_history) > 0:
        ax_params = fig.add_subplot(gs[diagnostic_row_start + 1, :])
        
        # Handle different parameter types
        if isinstance(param_history[0], jnp.ndarray):
            # Polynomial ansatz - simple array of parameters
            param_history_array = np.array(param_history)
            if 'times' in snapshots:
                param_times = snapshots['times']
            else:
                raise ValueError("Snapshots missing 'times' key - cannot plot parameter history")
            for i in range(param_history_array.shape[1]):
                ax_params.plot(param_times, param_history_array[:, i], 
                              label=f'Param {i}', alpha=0.7)
        else:
            # Neural network ansatz - complex structure, plot norm of parameters
            param_norms = []
            if 'times' in snapshots:
                param_times = snapshots['times']
            else:
                raise ValueError("Snapshots missing 'times' key - cannot plot parameter history")
            for params in param_history:
                # Calculate L2 norm of all parameters
                param_arrays = eqx.filter(params, eqx.is_array)
                total_norm = 0.0
                for param in jax.tree_leaves(param_arrays):
                    total_norm += jnp.sum(param ** 2)
                param_norms.append(float(jnp.sqrt(total_norm)))
            
            ax_params.plot(param_times, param_norms, 
                          label='Parameter Norm', alpha=0.7, color='red')
        
        ax_params.set_xlabel('Time')
        ax_params.set_ylabel('Parameter Value')
        ax_params.set_title('Parameter History')
        ax_params.legend()
        ax_params.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{ansatz_dir}/distributions_{potential_name}.png", dpi=300, bbox_inches='tight')
    plt.close()



# Removed unnecessary plotting functions - only keeping the essential ones 

# Removed unnecessary plotting functions - only keeping the essential ones 

def create_comparison_plots(all_snapshots, delta_t, make_V, system_name, dim):
    """Create comparison plots for all four simulation methods."""
    # Create figures directory
    os.makedirs("figures", exist_ok=True)
    ansatz_dir = f"figures/polynomial"
    os.makedirs(ansatz_dir, exist_ok=True)
    
    # Create a comprehensive comparison ridge plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Get time points - use saved times if available, otherwise calculate from first snapshot
    times = None
    lambda_values = None
    
    # Try to find saved times and lambda_values
    for method, snapshots in all_snapshots.items():
        if isinstance(snapshots, dict) and 'particles' in snapshots:
            # Check if this method has saved times directly in snapshots
            if 'times' in snapshots:
                times = snapshots['times']
                lambda_values = snapshots['lam']
                break
            # Check if this method has separate times keys (old format)
            times_key = f'times_{method}'
            lambda_key = f'lambda_values_{method}'
            if times_key in all_snapshots:
                times = all_snapshots[times_key]
                lambda_values = all_snapshots[lambda_key]
                break
    
    # Fallback to calculating from first snapshot
    if times is None:
        first_snapshot = None
        for snapshots in all_snapshots.values():
            if isinstance(snapshots, dict) and 'particles' in snapshots:
                first_snapshot = snapshots['particles']
                break
        
        if not first_snapshot:
            print("No valid snapshots found for plotting")
            return
        
        # Cannot plot without saved times
        raise ValueError("No saved times found in snapshots - cannot create comparison plots")
    
    # Find global range for consistent x-axis
    # Exclude methods with extreme particle values that would dominate the range
    all_qs = []
    for method, snapshots in all_snapshots.items():
        if isinstance(snapshots, dict) and 'particles' in snapshots:
            particles = snapshots['particles']
            # Check if this method has reasonable particle values
            method_qs = np.concatenate(particles)
            method_range = np.max(method_qs) - np.min(method_qs)
            
            # Only include methods with reasonable ranges (exclude exploded naive HMC)
            if method_range < 1000:  # Reasonable range threshold
                all_qs.extend(particles)
                print(f"Including {method} in global range (range: {method_range:.2f})")
            else:
                print(f"Excluding {method} from global range (range: {method_range:.2f})")
    
    if not all_qs:
        # Fallback: use all methods if none pass the filter
        for method, snapshots in all_snapshots.items():
            if isinstance(snapshots, dict) and 'particles' in snapshots:
                all_qs.extend(snapshots['particles'])
    
    x_min = np.min(np.concatenate(all_qs)) - 0.5
    x_max = np.max(np.concatenate(all_qs)) + 0.5
    x_grid = np.linspace(x_min, x_max, 200)
    
    # Plot each method
    colors = {'naive_unweighted': 'blue', 'naive_weighted': 'green', 'cd_unweighted': 'red', 'cd_weighted': 'orange'}
    titles = {
        'naive_unweighted': 'Naive HMC (Unweighted)',
        'naive_weighted': 'Naive HMC (Weighted SMC)',
        'cd_unweighted': 'Counterdiabatic HMC (Unweighted)',
        'cd_weighted': 'Counterdiabatic HMC (Weighted)'
    }
    
    # Filter out non-snapshot keys (like loss_histories and param_history)
    snapshot_methods = {k: v for k, v in all_snapshots.items() if k in titles}
    
    for i, (method, snapshots) in enumerate(snapshot_methods.items()):
        ax = axes[i]
        ax.set_title(titles[method], fontsize=14, fontweight='bold')
        ax.set_xlabel("Position q", fontsize=12)
        ax.set_ylabel("Time t", fontsize=12)
        
        # Use unified keys for all methods
        snapshot_key = 'particles'
        weights_key = 'weights'
        lam_key = 'lam'
        
        # Use saved lambda values if available, otherwise use snapshots[lam_key]
        if lambda_values is not None:
            method_lambda_values = lambda_values
        else:
            method_lambda_values = snapshots[lam_key]
        
        # Plot distributions
        for j, (t, snap, lam_val) in enumerate(zip(times, snapshots[snapshot_key], method_lambda_values)):
            # Get weights if available
            weights = None
            if weights_key and snapshots[weights_key][j] is not None:
                log_weights = snapshots[weights_key][j]
                # Check if all log weights are zero (unit weights)
                if np.allclose(log_weights, 0.0):
                    weights = None  # Use unweighted histogram
                else:
                    weights = np.exp(log_weights - np.max(log_weights))
                    weights = weights / np.sum(weights)
            
            # Compute KDE
            try:
                from scipy.stats import gaussian_kde
                if weights is not None:
                    # Use weighted KDE
                    kde = gaussian_kde(snap.flatten(), weights=weights)
                else:
                    kde = gaussian_kde(snap.flatten())
                density = kde(x_grid)
            except:
                # Fallback to histogram
                if weights is not None:
                    hist, bin_edges = np.histogram(snap, bins=50, density=True, range=(x_min, x_max), weights=weights)
                else:
                    hist, bin_edges = np.histogram(snap, bins=50, density=True, range=(x_min, x_max))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                density = np.interp(x_grid, bin_centers, hist)
            
            # Normalize and offset for ridge plot
            density = density / np.max(density) * 1.8
            offset = t * 2.0  # Increased spacing between plots to reduce overlap
            
            # Plot the ridge
            ax.fill_between(x_grid, offset, offset + density, 
                           color=colors[method], alpha=0.4, edgecolor=colors[method], linewidth=0.5)
            
            # Add true distribution
            potential_fn = make_V(lam_val)
            rho = np.array([np.exp(-potential_fn(x)) for x in x_grid])
            rho = rho / np.max(rho) * 1.8
            ax.plot(x_grid, offset + rho, 'k--', linewidth=1.5, alpha=0.8)
        
        # Set limits
        ax.set_xlim(x_min, x_max)
        times_array = np.array(times)  # Convert to numpy array for arithmetic
        ax.set_ylim(times_array[0] * 2.0 - 0.1, times_array[-1] * 2.0 + 2.0)  # Adjusted for increased spacing
        ax.set_yticks(times_array * 2.0)
    
    plt.tight_layout()
    plt.savefig(f"{ansatz_dir}/comparison_ridge_plot_{system_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison ridge plot to {ansatz_dir}/comparison_ridge_plot_{system_name}.png")

def create_comparison_histogram_plots(all_snapshots, delta_t, make_V, system_name, dim, n_bins=50):
    """Create comparison plots using histograms instead of KDE for all four simulation methods."""
    # Create figures directory
    os.makedirs("figures", exist_ok=True)
    ansatz_dir = f"figures/polynomial"
    os.makedirs(ansatz_dir, exist_ok=True)
    
    # Create a comprehensive comparison histogram ridge plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Get time points - use saved times if available, otherwise calculate from first snapshot
    times = None
    lambda_values = None
    
    # Try to find saved times and lambda_values
    for method, snapshots in all_snapshots.items():
        if isinstance(snapshots, dict) and 'particles' in snapshots:
            # Check if this method has saved times directly in snapshots
            if 'times' in snapshots:
                times = snapshots['times']
                lambda_values = snapshots['lam']
                break
            # Check if this method has separate times keys (old format)
            times_key = f'times_{method}'
            lambda_key = f'lambda_values_{method}'
            if times_key in all_snapshots:
                times = all_snapshots[times_key]
                lambda_values = all_snapshots[lambda_key]
                break
    
    # Fallback to calculating from first snapshot
    if times is None:
        first_snapshot = None
        for snapshots in all_snapshots.values():
            if isinstance(snapshots, dict) and 'particles' in snapshots:
                first_snapshot = snapshots['particles']
                break
        
        if not first_snapshot:
            print("No valid snapshots found for plotting")
            return
        
        # Cannot plot without saved times
        raise ValueError("No saved times found in snapshots - cannot create comparison plots")
    
    # Find global range for consistent x-axis
    # Exclude methods with extreme particle values that would dominate the range
    all_qs = []
    for method, snapshots in all_snapshots.items():
        if isinstance(snapshots, dict) and 'particles' in snapshots:
            particles = snapshots['particles']
            # Check if this method has reasonable particle values
            method_qs = np.concatenate(particles)
            method_range = np.max(method_qs) - np.min(method_qs)
            
            # Only include methods with reasonable ranges (exclude exploded naive HMC)
            if method_range < 1000:  # Reasonable range threshold
                all_qs.extend(particles)
                print(f"Including {method} in global range (range: {method_range:.2f})")
            else:
                print(f"Excluding {method} from global range (range: {method_range:.2f})")
    
    if not all_qs:
        # Fallback: use all methods if none pass the filter
        for method, snapshots in all_snapshots.items():
            if isinstance(snapshots, dict) and 'particles' in snapshots:
                all_qs.extend(snapshots['particles'])
    
    x_min = np.min(np.concatenate(all_qs)) - 0.5
    x_max = np.max(np.concatenate(all_qs)) + 0.5
    
    # Create histogram bins
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot each method
    colors = {'naive_unweighted': 'blue', 'naive_weighted': 'green', 'cd_unweighted': 'red', 'cd_weighted': 'orange'}
    titles = {
        'naive_unweighted': 'Naive HMC (Unweighted)',
        'naive_weighted': 'Naive HMC (Weighted SMC)',
        'cd_unweighted': 'Counterdiabatic HMC (Unweighted)',
        'cd_weighted': 'Counterdiabatic HMC (Weighted)'
    }
    
    # Filter out non-snapshot keys (like loss_histories and param_history)
    snapshot_methods = {k: v for k, v in all_snapshots.items() if k in titles}
    
    for i, (method, snapshots) in enumerate(snapshot_methods.items()):
        ax = axes[i]
        ax.set_title(titles[method], fontsize=14, fontweight='bold')
        ax.set_xlabel("Position q", fontsize=12)
        ax.set_ylabel("Time t", fontsize=12)
        
        # Use unified keys for all methods
        snapshot_key = 'particles'
        weights_key = 'weights'
        lam_key = 'lam'
        
        # Use saved lambda values if available, otherwise use snapshots[lam_key]
        if lambda_values is not None:
            method_lambda_values = lambda_values
        else:
            method_lambda_values = snapshots[lam_key]
        
        # Plot distributions
        for j, (t, snap, lam_val) in enumerate(zip(times, snapshots[snapshot_key], method_lambda_values)):
            # Get weights if available
            weights = None
            if weights_key and snapshots[weights_key][j] is not None:
                log_weights = snapshots[weights_key][j]
                # Check if all log weights are zero (unit weights)
                if np.allclose(log_weights, 0.0):
                    weights = None  # Use unweighted histogram
                else:
                    weights = np.exp(log_weights - np.max(log_weights))
                    weights = weights / np.sum(weights)
            
            # Compute histogram
            if weights is not None:
                hist, _ = np.histogram(snap.flatten(), bins=bin_edges, density=True, weights=weights)
            else:
                hist, _ = np.histogram(snap.flatten(), bins=bin_edges, density=True)
            
            # Normalize and offset for ridge plot
            hist = hist / np.max(hist) * 1.8
            offset = t * 2.0  # Increased spacing between plots to reduce overlap
            
            # Plot the histogram as individual bars (true histogram, not filled area)
            for i, (bin_center, height) in enumerate(zip(bin_centers, hist)):
                if height > 0:  # Only plot non-zero bars
                    ax.bar(bin_center, height, width=bin_edges[1]-bin_edges[0], 
                          bottom=offset, color=colors[method], alpha=0.6, 
                          edgecolor=colors[method], linewidth=0.3)
            
            # Add true distribution
            potential_fn = make_V(lam_val)
            rho = np.array([np.exp(-potential_fn(x)) for x in bin_centers])
            rho = rho / np.max(rho) * 1.8
            ax.plot(bin_centers, offset + rho, 'k--', linewidth=1.5, alpha=0.8)
        
        # Set limits
        ax.set_xlim(x_min, x_max)
        times_array = np.array(times)  # Convert to numpy array for arithmetic
        ax.set_ylim(times_array[0] * 2.0 - 0.1, times_array[-1] * 2.0 + 2.0)  # Adjusted for increased spacing
        ax.set_yticks(times_array * 2.0)
    
    plt.tight_layout()
    plt.savefig(f"{ansatz_dir}/comparison_histogram_plot_{system_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison histogram plot to {ansatz_dir}/comparison_histogram_plot_{system_name}.png")

def create_all_plots(successful_simulations, system_name, ansatz, delta_t, make_V, make_T=None, dim=1):
    """
    Create all plots for the simulation results.
    
    Args:
        successful_simulations: Dictionary containing simulation results
        system_name: Name of the system
        ansatz: The ansatz object used
        delta_t: Time step
        make_V: Function to create potential energy
        make_T: Function to create kinetic energy (optional)
        dim: Dimension of the system (default=1)
    """
    if len(successful_simulations) > 0:
        if dim == 1:
            print(f"\nCreating comparison plots for {len(successful_simulations)} successful simulations...")
            create_comparison_plots(successful_simulations, delta_t, make_V, system_name, dim=dim)
            
            # Create histogram-based comparison plots
            print("Creating histogram-based comparison plots...")
            create_comparison_histogram_plots(successful_simulations, delta_t, make_V, system_name, dim=dim)
            
            # Create unified distributions plot showing all time points
            print("Creating unified distributions plot...")
            ansatz_dir = "figures/polynomial"  # Default directory
            create_unified_distributions_plot(successful_simulations, delta_t, make_V, ansatz_dir, system_name, dim=dim)
        else:
            print(f"\nSkipping 1D comparison plots for {dim}D system...")
        
        # Create detailed distribution plots for counterdiabatic methods
        cd_methods = ['cd_unweighted', 'cd_weighted']
        for method in cd_methods:
            if method in successful_simulations:
                print(f"Creating detailed distribution plot for {method.replace('_', ' ')} case...")
                loss_histories = successful_simulations.get(f'loss_histories_{method}', [])
                param_history = successful_simulations.get(f'param_history_{method}', None)
                naive_method = method.replace('cd_', 'naive_')
                naive_snapshots = successful_simulations.get(naive_method, None)
                
                # Use snapshots directly - times are already stored in snapshots['times']
                snapshots_with_timing = successful_simulations[method].copy()
                
                plot_results(snapshots_with_timing, loss_histories, delta_t, make_V, 
                            param_history=param_history, ansatz=ansatz, 
                            potential_name=f"{system_name}_{method}", dim=1, plot_ansatz=False, 
                            make_T=make_T, naive_snapshots=naive_snapshots)
        
        # Create overlay ridge plot for CD weighted vs unweighted
        if 'cd_weighted' in successful_simulations:
            print("Creating overlay ridge plot for CD weighted vs unweighted...")
            # Use the weighted snapshots which contain both particles and weights
            weighted_snapshots = successful_simulations['cd_weighted']
            
            # Determine ansatz type for directory
            if isinstance(ansatz, PolynomialAnsatz):
                ansatz_type = 'polynomial'
            elif isinstance(ansatz, NeuralNetworkAnsatz):
                ansatz_type = 'neural_network'
            elif isinstance(ansatz, AnalyticAnsatz):
                ansatz_type = 'analytic'
            else:
                ansatz_type = 'polynomial'  # Default
            
            create_overlay_ridge_plot(weighted_snapshots, delta_t, make_V, system_name, ansatz_type)
    else:
        print("No successful simulations to plot.") 

def create_unified_distributions_plot(successful_simulations, delta_t, make_V, ansatz_dir, potential_name, dim):
    """
    Create a unified distributions plot showing all time points from both CD and naive simulations.
    Shows subplots for every time in the union of both sets.
    
    Args:
        successful_simulations: Dictionary containing all simulation results
        delta_t: Base time step
        make_V: Function to create potential energy
        ansatz_dir: Directory to save plots
        potential_name: Name of the potential for filename
        dim: Dimension of the system
    """
    # Collect all time points from all simulations
    all_times = set()
    all_lambda_values = {}
    
    # Extract times from each method
    methods = ['naive_unweighted', 'naive_weighted', 'cd_unweighted', 'cd_weighted']
    for method in methods:
        if method in successful_simulations:
            # Get saved times if available
            # Get times and lambda values from snapshots
            snapshots = successful_simulations[method]
            if 'times' in snapshots and len(snapshots['times']) > 0:
                # Use saved times from snapshots
                times = snapshots['times']
                lambda_values = snapshots['lam']
            else:
                # Cannot plot without saved times
                raise ValueError(f"Method {method} missing 'times' key - cannot create unified distributions plot")
            
            # Add all times to the set (convert JAX arrays to Python types)
            for t, lam in zip(times, lambda_values):
                # Convert JAX arrays to Python types for hashing
                t_python = float(t) if hasattr(t, 'item') else t
                lam_python = float(lam) if hasattr(lam, 'item') else lam
                all_times.add(t_python)
                all_lambda_values[t_python] = lam_python
    
    # Sort times for consistent plotting
    all_times = sorted(list(all_times))
    
    if not all_times:
        print("No time points found for plotting")
        return
    
    # Calculate grid layout
    num_times = len(all_times)
    cols = 4
    rows = int(np.ceil(num_times / cols))
    
    # Create figure
    fig = plt.figure(figsize=(20, 5*rows))
    gs = fig.add_gridspec(rows, cols, height_ratios=[1]*rows, width_ratios=[1]*cols)
    
    # Find global range for consistent x-axis
    all_qs = []
    for method in methods:
        if method in successful_simulations:
            snapshots = successful_simulations[method]
            if 'particles' in snapshots:
                all_qs.extend(snapshots['particles'])
    
    if all_qs:
        x_min = np.min(np.concatenate(all_qs)) - 0.5
        x_max = np.max(np.concatenate(all_qs)) + 0.5
    else:
        x_min, x_max = -3, 3
    
    x_grid = np.linspace(x_min, x_max, 200)
    
    # Colors for different methods
    colors = {
        'naive_unweighted': 'blue',
        'naive_weighted': 'green', 
        'cd_unweighted': 'red',
        'cd_weighted': 'orange'
    }
    
    # Plot each time point
    for i, t in enumerate(all_times):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        
        lam_val = all_lambda_values.get(t, 0.0)
        
        # Plot data from each method that has data at this time
        for method in methods:
            if method in successful_simulations:
                snapshots = successful_simulations[method]
                
                # Find the closest time point in this method's data
                if 'times' in snapshots and len(snapshots['times']) > 0:
                    # Use saved times from snapshots
                    method_times = snapshots['times']
                else:
                    raise ValueError(f"Method {method} missing 'times' key - cannot create unified distributions plot")
                
                # Find the closest time point
                time_idx = None
                min_diff = float('inf')
                for j, method_t in enumerate(method_times):
                    if abs(method_t - t) < min_diff:
                        min_diff = abs(method_t - t)
                        time_idx = j
                
                # Only plot if we have data at this time (within tolerance)
                if time_idx is not None and min_diff < 0.1:  # Allow approximate matches for adaptive stepping
                    particles = snapshots['particles'][time_idx]
                    
                    # Get weights if available
                    weights = None
                    if 'weights' in snapshots and len(snapshots['weights']) > time_idx:
                        log_weights = snapshots['weights'][time_idx]
                        if not np.allclose(log_weights, 0.0):
                            weights = np.exp(log_weights - np.max(log_weights))
                            weights = weights / np.sum(weights)
                    
                    # Create histogram
                    if weights is not None:
                        ax.hist(particles.flatten(), bins=50, density=True, alpha=0.6,
                               color=colors[method], label=method.replace('_', ' ').title(),
                               weights=weights, range=(x_min, x_max))
                    else:
                        ax.hist(particles.flatten(), bins=50, density=True, alpha=0.6,
                               color=colors[method], label=method.replace('_', ' ').title(),
                               range=(x_min, x_max))
        
        # Add true distribution
        potential_fn = make_V(lam_val)
        rho = np.array([np.exp(-potential_fn(x)) for x in x_grid])
        rho = rho / np.trapz(rho, x_grid)  # Normalize
        ax.plot(x_grid, rho, 'k--', linewidth=2, alpha=0.8, label='True')
        
        # Show which methods have data at this time
        methods_with_data = []
        for method in methods:
            if method in successful_simulations:
                snapshots = successful_simulations[method]
                times_key = f'times_{method}'
                if times_key in successful_simulations:
                    method_times = successful_simulations[times_key]
                else:
                    method_times = np.arange(len(snapshots['particles'])) * delta_t
                
                # Check if this method has data near this time
                if 'times' in snapshots and len(snapshots['times']) > 0:
                    method_times = snapshots['times']
                else:
                    raise ValueError(f"Method {method} missing 'times' key - cannot create unified distributions plot")
                
                for method_t in method_times:
                    if abs(method_t - t) < 0.1:
                        methods_with_data.append(method.replace('_', ' ').title())
                        break
        
        methods_str = ', '.join(methods_with_data) if methods_with_data else 'None'
        ax.set_title(f't = {t:.3f}, λ = {lam_val:.3f}\nMethods: {methods_str}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{ansatz_dir}/unified_distributions_{potential_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved unified distributions plot to {ansatz_dir}/unified_distributions_{potential_name}.png")

def create_2d_ansatz_plot(successful_simulations, system_name, ansatz):
    """
    Create ansatz plots for 2D CD simulations showing A(q₁,p₁) and A(q₂,p₂).
    """
    print(f"Creating ansatz plots for 2D {system_name}...")
    
    # Only plot for CD methods
    cd_methods = ['cd_unweighted', 'cd_weighted']
    available_methods = [method for method in cd_methods if method in successful_simulations]
    
    if not available_methods:
        print("No CD methods found for ansatz plotting.")
        return
    
    # Create a figure with subplots for each method
    fig, axes = plt.subplots(len(available_methods), 2, figsize=(12, 5*len(available_methods)))
    if len(available_methods) == 1:
        axes = axes.reshape(1, -1)
    
    colors = {
        'cd_unweighted': 'red',
        'cd_weighted': 'orange'
    }
    
    for method_idx, method in enumerate(available_methods):
        ax1 = axes[method_idx, 0]  # q₁ vs p₁ plot
        ax2 = axes[method_idx, 1]  # q₂ vs p₂ plot
        
        snapshots = successful_simulations[method]
        
        # Get the final snapshot to see the range of values
        if 'particles' in snapshots and len(snapshots['particles']) > 0:
            final_particles = snapshots['particles'][-1]
            
            # For 2D, particles only contain q coordinates (shape: N, 2)
            # We need to generate reasonable p ranges for plotting
            q_min, q_max = np.min(final_particles), np.max(final_particles)
            q_range = np.linspace(q_min - 0.5, q_max + 0.5, 50)
            # Use a reasonable p range based on typical momentum values
            p_range = np.linspace(-2.0, 2.0, 50)
            
            # Create meshgrid for plotting
            Q1, P1 = np.meshgrid(q_range, p_range)
            Q2, P2 = np.meshgrid(q_range, p_range)
            
            # Evaluate ansatz A(q₁,p₁) and A(q₂,p₂)
            A_q1p1 = np.zeros_like(Q1)
            A_q2p2 = np.zeros_like(Q2)
            
            # For A(q₁,p₁), fix q₂=0, p₂=0
            for i in range(Q1.shape[0]):
                for j in range(Q1.shape[1]):
                    q = np.array([Q1[i,j], 0.0])  # q₁ varies, q₂=0
                    p = np.array([P1[i,j], 0.0])  # p₁ varies, p₂=0
                    A_q1p1[i,j] = ansatz(q, p)
            
            # For A(q₂,p₂), fix q₁=0, p₁=0
            for i in range(Q2.shape[0]):
                for j in range(Q2.shape[1]):
                    q = np.array([0.0, Q2[i,j]])  # q₁=0, q₂ varies
                    p = np.array([0.0, P2[i,j]])  # p₁=0, p₂ varies
                    A_q2p2[i,j] = ansatz(q, p)
            
            # Plot A(q₁,p₁)
            im1 = ax1.contourf(Q1, P1, A_q1p1, levels=20, cmap='RdBu_r')
            ax1.set_xlabel('q₁')
            ax1.set_ylabel('p₁')
            ax1.set_title(f'{method.replace("_", " ").title()}: A(q₁,p₁)')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(im1, ax=ax1, label='A(q₁,p₁)')
            
            # Plot A(q₂,p₂)
            im2 = ax2.contourf(Q2, P2, A_q2p2, levels=20, cmap='RdBu_r')
            ax2.set_xlabel('q₂')
            ax2.set_ylabel('p₂')
            ax2.set_title(f'{method.replace("_", " ").title()}: A(q₂,p₂)')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(im2, ax=ax2, label='A(q₂,p₂)')
    
    plt.tight_layout()
    plt.savefig(f'figures/polynomial/2d_{system_name}_ansatz.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created ansatz plots for {len(available_methods)} CD method(s)")

def create_2d_loss_plot(successful_simulations, system_name):
    """
    Create loss curve plots for 2D CD simulations.
    """
    print(f"Creating loss curves for 2D {system_name}...")
    
    # Only plot for CD methods that have loss histories
    cd_methods = ['cd_unweighted', 'cd_weighted']
    available_methods = [method for method in cd_methods if method in successful_simulations]
    
    if not available_methods:
        print("No CD methods with loss histories found.")
        return
    
    fig, axes = plt.subplots(1, len(available_methods), figsize=(6*len(available_methods), 5))
    if len(available_methods) == 1:
        axes = [axes]
    
    colors = {
        'cd_unweighted': 'red',
        'cd_weighted': 'orange'
    }
    
    for i, method in enumerate(available_methods):
        ax = axes[i]
        snapshots = successful_simulations[method]
        
        # Check if loss histories are available (they're stored with a different key)
        loss_histories_key = f'loss_histories_{method}'
        if loss_histories_key not in successful_simulations or not successful_simulations[loss_histories_key]:
            ax.text(0.5, 0.5, f'No loss data for {method}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{method.replace("_", " ").title()}: Loss History')
            continue
        
        loss_histories = successful_simulations[loss_histories_key]
        times = snapshots['times'] if 'times' in snapshots else list(range(len(loss_histories)))
        
        # Plot each fitting step's loss history
        for step_idx, (loss_history, time_val) in enumerate(zip(loss_histories, times)):
            if loss_history:  # Check if loss history is not empty
                iterations = range(len(loss_history))
                ax.semilogy(iterations, loss_history, 
                           alpha=0.7, linewidth=1, 
                           color=colors[method],
                           label=f'Step {step_idx} (t={time_val:.2f})' if step_idx < 3 else "")
        
        ax.set_xlabel('Optimization Iteration')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title(f'{method.replace("_", " ").title()}: Loss History')
        ax.grid(True, alpha=0.3)
        if len(loss_histories) <= 3:  # Only show legend if few steps
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'figures/polynomial/2d_{system_name}_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created loss curves for {len(available_methods)} CD method(s)")

def create_2d_plots(successful_simulations, system_name, ansatz, delta_t, make_V, make_T=None):
    """
    Create 2D plots for simulation results.
    This function handles 2D systems that can't use the 1D plotting functions.
    """
    print(f"Creating 2D plots for {len(successful_simulations)} successful simulations...")
    
    # Create a comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'2D {system_name}: Method Comparison', fontsize=16)
    
    colors = {
        'naive_unweighted': 'blue',
        'naive_weighted': 'green', 
        'cd_unweighted': 'red',
        'cd_weighted': 'orange'
    }
    
    methods = ['naive_unweighted', 'naive_weighted', 'cd_unweighted', 'cd_weighted']
    
    # Plot final distributions for each method
    for i, method in enumerate(methods):
        if method not in successful_simulations:
            continue
            
        ax = axes[i//2, i%2]
        snapshots = successful_simulations[method]  # Data is stored directly
        
        # Get the final snapshot
        if 'particles' in snapshots and len(snapshots['particles']) > 0:
            final_particles = snapshots['particles'][-1]
            final_time = snapshots['times'][-1] if 'times' in snapshots else 0.0
            final_lam = snapshots['lam'][-1] if 'lam' in snapshots else 0.0
            
            # Plot particles
            ax.scatter(final_particles[:, 0], final_particles[:, 1], 
                      alpha=0.6, s=10, color=colors[method], label=f'{method.replace("_", " ")}')
            
            # Add weights if available
            if 'weights' in snapshots and len(snapshots['weights']) > 0:
                weights = snapshots['weights'][-1]
                if weights is not None and not np.allclose(weights, 0.0):
                    # Normalize weights for visualization
                    weights_norm = np.exp(weights - np.max(weights))
                    weights_norm = weights_norm / np.sum(weights_norm)
                    # Use alpha based on weights
                    alphas = 0.3 + 0.7 * weights_norm
                    ax.scatter(final_particles[:, 0], final_particles[:, 1], 
                              alpha=alphas, s=10, color=colors[method], label=f'{method.replace("_", " ")} (weighted)')
            
            # Plot the interpolating potential contour at this timestep
            x = np.linspace(-2, 2, 100)
            y = np.linspace(-1, 3, 100)
            X, Y = np.meshgrid(x, y)
            
            # Create the interpolating potential: (1-λ)*normal + λ*rosenbrock
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    q = np.array([X[i,j], Y[i,j]])
                    # Use the same potential function as the simulation
                    potential_fn = make_V(final_lam)
                    Z[i,j] = potential_fn(q)
            
            # Plot contours
            levels = np.logspace(-1, 2, 10)
            ax.contour(X, Y, Z, levels=levels, colors='black', alpha=0.5, linewidths=0.5)
            
            ax.set_xlabel('q₀')
            ax.set_ylabel('q₁')
            ax.set_title(f'{method.replace("_", " ").title()}\nt={final_time:.2f}, λ={final_lam:.3f}')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'figures/polynomial/2d_{system_name}_final_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create evolution plot showing how particles move over time
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'2D {system_name}: Evolution Over Time', fontsize=16)
    
    # Use CD unweighted for evolution (has more detailed snapshots)
    if 'cd_unweighted' in successful_simulations:
        snapshots = successful_simulations['cd_unweighted']
        times = snapshots['times'] if 'times' in snapshots else []
        particles_list = snapshots['particles'] if 'particles' in snapshots else []
        
        # Plot every other snapshot to fit in 10 subplots
        step = max(1, len(particles_list) // 10)
        
        for i in range(min(10, len(particles_list))):
            idx = i * step
            if idx >= len(particles_list):
                break
                
            ax = axes[i//5, i%5]
            particles = particles_list[idx]
            time_val = times[idx] if idx < len(times) else idx * delta_t
            lam_val = snapshots['lam'][idx] if 'lam' in snapshots and idx < len(snapshots['lam']) else 0.0
            
            # Plot particles
            ax.scatter(particles[:, 0], particles[:, 1], alpha=0.6, s=5, color='red')
            
            # Plot the interpolating potential contour at this timestep
            x = np.linspace(-2, 2, 50)
            y = np.linspace(-1, 3, 50)
            X, Y = np.meshgrid(x, y)
            
            # Create the interpolating potential: (1-λ)*normal + λ*rosenbrock
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    q = np.array([X[i,j], Y[i,j]])
                    # Use the same potential function as the simulation
                    potential_fn = make_V(lam_val)
                    Z[i,j] = potential_fn(q)
            
            # Plot contours
            levels = np.logspace(-1, 2, 5)
            ax.contour(X, Y, Z, levels=levels, colors='black', alpha=0.5, linewidths=0.5)
            
            ax.set_xlabel('q₀')
            ax.set_ylabel('q₁')
            ax.set_title(f't={time_val:.2f}, λ={lam_val:.3f}')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'figures/polynomial/2d_{system_name}_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create method comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'2D {system_name}: Energy Statistics', fontsize=16)
    
    # Plot 1: Average energy over time
    ax1 = axes[0, 0]
    for method in methods:
        if method in successful_simulations:
            snapshots = successful_simulations[method]
            if 'detailed_energy_stats' in snapshots and 'times' in snapshots:
                energy_stats = snapshots['detailed_energy_stats']
                times = snapshots['times']
                H_vals = [stats.get('avg_H', 0) for stats in energy_stats]
                ax1.plot(times, H_vals, color=colors[method], label=method.replace('_', ' '), linewidth=2)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('<H>')
    ax1.set_title('Average Energy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy derivative over time
    ax2 = axes[0, 1]
    for method in methods:
        if method in successful_simulations:
            snapshots = successful_simulations[method]
            if 'detailed_energy_stats' in snapshots and 'times' in snapshots:
                energy_stats = snapshots['detailed_energy_stats']
                times = snapshots['times']
                dH_dlam_vals = [stats.get('avg_dH_dlam', 0) for stats in energy_stats]
                ax2.plot(times, dH_dlam_vals, color=colors[method], label=method.replace('_', ' '), linewidth=2)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('<∂H/∂λ>')
    ax2.set_title('Energy Derivative')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Energy variance over time
    ax3 = axes[1, 0]
    for method in methods:
        if method in successful_simulations:
            snapshots = successful_simulations[method]
            if 'detailed_energy_stats' in snapshots and 'times' in snapshots:
                energy_stats = snapshots['detailed_energy_stats']
                times = snapshots['times']
                dH2_vals = [stats.get('avg_delta_H_sq', 0) for stats in energy_stats]
                ax3.plot(times, dH2_vals, color=colors[method], label=method.replace('_', ' '), linewidth=2)
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('<ΔH²>')
    ax3.set_title('Energy Change Variance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Var[A] for CD methods
    ax4 = axes[1, 1]
    for method in ['cd_unweighted', 'cd_weighted']:
        if method in successful_simulations:
            snapshots = successful_simulations[method]
            if 'detailed_energy_stats' in snapshots and 'times' in snapshots:
                energy_stats = snapshots['detailed_energy_stats']
                times = snapshots['times']
                var_A_vals = [stats.get('var_A', 0) for stats in energy_stats]
                ax4.plot(times, var_A_vals, color=colors[method], label=method.replace('_', ' '), linewidth=2)
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Var[A]')
    ax4.set_title('Gauge Potential Variance (CD only)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'figures/polynomial/2d_{system_name}_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create loss curve plot for CD methods
    create_2d_loss_plot(successful_simulations, system_name)
    
    # Create ansatz plots for CD methods
    create_2d_ansatz_plot(successful_simulations, system_name, ansatz)
    
    print(f"✓ Created 2D {system_name} distribution plots")
    print(f"📁 Plots saved to: figures/polynomial/")
    print(f"   - 2d_{system_name}_final_distributions.png")
    print(f"   - 2d_{system_name}_evolution.png")
    print(f"   - 2d_{system_name}_method_comparison.png")
    print(f"   - 2d_{system_name}_loss_curves.png")
    print(f"   - 2d_{system_name}_ansatz.png") 