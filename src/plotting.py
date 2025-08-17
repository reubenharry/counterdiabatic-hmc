import jax
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import equinox as eqx
import jax
import jax.numpy as jnp
import os
from scipy.stats import gaussian_kde

from .ansatze import PolynomialAnsatz, NeuralNetworkAnsatz, AnalyticAnsatz

def plot_learned_ansatz(ax, theta, ansatz, q_range=(-3, 3), p_range=(-3, 3), n_points=25, dim=1):
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
        
        # Compute weighted histogram
        hist, bin_edges = np.histogram(samples, bins=50, weights=weights, density=True, 
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
            hist, bin_edges = np.histogram(samples, bins=50, density=True, 
                                          range=(np.min(x_grid), np.max(x_grid)))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            density = np.interp(x_grid, bin_centers, hist)
    
    return density

def create_ridge_plot(snapshots, delta_t, make_V, lam_fn, potential_name="harmonic", ansatz_type="polynomial"):
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
    
    # Check if re-equilibration was used
    has_re_equil = 'cd_post_equil' in snapshots and len(snapshots['cd_post_equil']) > 0
    
    # Check if weights are available
    has_weights = 'weights' in snapshots and any(w is not None for w in snapshots['weights'])
    
    # Get time points
    times = np.arange(len(snapshots['cd_pre_equil'])) * delta_t * 10  # *10 because we record every 10 steps
    
    # For post-equilibration snapshots, the timing is different
    # They represent the state after CD step + re-equilibration, so they should be plotted
    # at the time after the CD step (i.e., at the next timestep)
    if has_re_equil:
        # Post-equilibration snapshots are stored at the end of timesteps
        # So they should be plotted at the next timestep's time
        post_equil_times = np.arange(1, len(snapshots['cd_post_equil']) + 1) * delta_t * 10
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
    all_qs = np.concatenate(snapshots['cd_pre_equil'])
    if has_re_equil:
        # Flatten all arrays to 1D before concatenation
        all_qs_flat = all_qs.flatten()
        cd_post_equil_flat = np.concatenate(snapshots['cd_post_equil']).flatten()
        all_qs = np.concatenate([all_qs_flat, cd_post_equil_flat])
    else:
        all_qs = all_qs.flatten()
    x_min = np.min(all_qs) - 0.5
    x_max = np.max(all_qs) + 0.5
    
    # Create x grid for smooth curves
    x_grid = np.linspace(x_min, x_max, 200)
    
    # Note: Naive HMC plotting removed since naive HMC is not performed in run_simulation
    
    # Note: Naive HMC weighted plotting removed since naive HMC is not performed in run_simulation
    
    # Plot CD HMC distributions (pre-equilibration)
    cd_ax = ax2 if has_weights else ax1  # Use appropriate axis based on layout
    for i, (t, cd_snap, lam_val) in enumerate(zip(times, snapshots['cd_pre_equil'], snapshots['lam_pre_equil'])):
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
    lambda_values = snapshots['lam_pre_equil']
    
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

def plot_results(snapshots, loss_histories, delta_t, make_V, lam_fn, param_history=None, ansatz=None, potential_name="harmonic", dim=1, plot_ansatz=False, make_T=None):
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
    print(f"Number of snapshots: naive={len(snapshots.get('naive', []))}, cd_pre_equil={len(snapshots.get('cd_pre_equil', []))}")
    if 'cd_post_equil' in snapshots:
        print(f"Number of post-equilibration snapshots: {len(snapshots['cd_post_equil'])}")
    
    # Check if re-equilibration was used
    has_re_equil = 'cd_post_equil' in snapshots and len(snapshots['cd_post_equil']) > 0
    
    # Create distributions plot with diagnostic information
    create_distributions_plot(snapshots, delta_t, make_V, ansatz_dir, potential_name, dim, has_re_equil, 
                            loss_histories=loss_histories, param_history=param_history, make_T=make_T)
    
    print(f"Saved distributions plot to {ansatz_dir}/distributions_{potential_name}.png")


def create_distributions_plot(snapshots, delta_t, make_V, ansatz_dir, potential_name, dim, has_re_equil, loss_histories=None, param_history=None, make_T=None):
    """Create the distributions plot showing histograms and diagnostic plots."""
    # Create figure with subplots for distributions and diagnostics
    fig = plt.figure(figsize=(20, 18))
    
    # Create grid layout: 2 rows, 4 columns
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # Define time points to plot (every 10 steps)
    times = np.arange(len(snapshots['cd_pre_equil'])) * delta_t * 10
    
    # Plot distributions at different time points (top 2 rows)
    for i, (time, lam_val) in enumerate(zip(times, snapshots['lam_pre_equil'])):
        if i >= 8:  # Only plot first 8 distributions
            break
            
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        
        # Plot CD-HMC distribution (pre-equilibration)
        if 'cd_pre_equil' in snapshots and i < len(snapshots['cd_pre_equil']):
            cd_snap = snapshots['cd_pre_equil'][i]
            if len(cd_snap) > 0:
                ax.hist(cd_snap.flatten(), bins=50, alpha=0.6, label='CD-HMC', density=True, color='red')
        
        # Plot true distribution
        x_grid = np.linspace(-5, 5, 1000)
        potential_fn = make_V(lam_val)
        rho = np.array([np.exp(-potential_fn(x)) for x in x_grid])
        rho = rho / np.trapz(rho, x_grid)  # Normalize
        ax.plot(x_grid, rho, 'k--', linewidth=2, label='True distribution')
        
        ax.set_title(f't = {time:.2f}, λ = {lam_val:.3f}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot loss curves (bottom left)
    if loss_histories and len(loss_histories) > 0:
        ax_loss = fig.add_subplot(gs[2, 0])
        for i, loss_history in enumerate(loss_histories):
            if len(loss_history) > 0:
                ax_loss.plot(loss_history, label=f'Step {i}', alpha=0.7)
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Loss History')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
    
    # Plot energy statistics (bottom right 3 plots)
    if 'detailed_energy_stats' in snapshots and len(snapshots['detailed_energy_stats']) > 0:
        energy_stats = snapshots['detailed_energy_stats']
        
        # Create time array for energy statistics (every timestep, not every 10 steps)
        energy_times = np.arange(len(energy_stats)) * delta_t
        
        # Plot <H> over time
        ax_H = fig.add_subplot(gs[2, 1])
        H_vals = [stats['cd']['avg_H'] for stats in energy_stats]
        ax_H.plot(energy_times, H_vals, 'b-', label='CD-HMC')
        ax_H.set_xlabel('Time')
        ax_H.set_ylabel('<H>')
        ax_H.set_title('Average Energy')
        ax_H.legend()
        ax_H.grid(True, alpha=0.3)
        
        # Plot <ΔH²> over time
        ax_dH2 = fig.add_subplot(gs[2, 2])
        dH2_vals = [stats['cd']['avg_delta_H_sq'] for stats in energy_stats]
        ax_dH2.plot(energy_times, dH2_vals, 'r-', label='CD-HMC')
        ax_dH2.set_xlabel('Time')
        ax_dH2.set_ylabel('<ΔH²>')
        ax_dH2.set_title('Energy Variance')
        ax_dH2.legend()
        ax_dH2.grid(True, alpha=0.3)
        
        # Plot <∂H/∂λ> over time
        ax_dH_dlam = fig.add_subplot(gs[2, 3])
        dH_dlam_vals = [stats['cd']['avg_dH_dlam'] for stats in energy_stats]
        ax_dH_dlam.plot(energy_times, dH_dlam_vals, 'g-', label='CD-HMC')
        ax_dH_dlam.set_xlabel('Time')
        ax_dH_dlam.set_ylabel('<∂H/∂λ>')
        ax_dH_dlam.set_title('Energy Derivative')
        ax_dH_dlam.legend()
        ax_dH_dlam.grid(True, alpha=0.3)
        
        # Plot <{A,H}> over time (Poisson bracket)
        ax_A_H = fig.add_subplot(gs[3, 0])
        A_H_vals = [stats['cd']['avg_A_H'] for stats in energy_stats]
        ax_A_H.plot(energy_times, A_H_vals, 'm-', label='CD-HMC')
        ax_A_H.set_xlabel('Time')
        ax_A_H.set_ylabel('<{A,H}>')
        ax_A_H.set_title('Poisson Bracket')
        ax_A_H.legend()
        ax_A_H.grid(True, alpha=0.3)
    
    # Plot parameter history if available (bottom row, spanning 3 columns)
    if param_history and len(param_history) > 0:
        ax_params = fig.add_subplot(gs[3, 1:])
        param_history_array = np.array(param_history)
        # Create time array for parameter history (every 10 steps)
        param_times = np.arange(len(param_history_array)) * delta_t * 10
        for i in range(param_history_array.shape[1]):
            ax_params.plot(param_times, param_history_array[:, i], 
                          label=f'Param {i}', alpha=0.7)
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