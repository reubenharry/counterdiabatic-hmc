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

def create_ridge_plot(snapshots, delta_t, make_V, lam_fn, potential_name="harmonic", ansatz_type="polynomial"):
    """Create a ridge plot showing the evolution of 1D distributions over time.
    
    Args:
        snapshots: Dictionary containing 'naive', 'cd', 'cd_post_equil', 'lam' arrays
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
    
    # Get time points
    times = np.arange(len(snapshots['naive'])) * delta_t * 10  # *10 because we record every 10 steps
    
    # For post-equilibration snapshots, the timing is different
    # They represent the state after CD step + re-equilibration, so they should be plotted
    # at the time after the CD step (i.e., at the next timestep)
    if has_re_equil:
        # Post-equilibration snapshots are stored at the end of timesteps
        # So they should be plotted at the next timestep's time
        post_equil_times = np.arange(1, len(snapshots['cd_post_equil']) + 1) * delta_t * 10
    else:
        post_equil_times = np.array([])
    
    # Create figure with narrower side-by-side layout
    if has_re_equil:
        # Three columns: naive, CD pre-equil, CD post-equil
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Create secondary y-axes for lambda values
        ax1_lambda = ax1.twinx()
        ax2_lambda = ax2.twinx()
        ax3_lambda = ax3.twinx()
    else:
        # Two columns: naive and CD
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Create secondary y-axes for lambda values
        ax1_lambda = ax1.twinx()
        ax2_lambda = ax2.twinx()
    
    # Plot naive HMC ridge plot
    ax1.set_title("Naïve HMC Evolution", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Position q", fontsize=12)
    ax1.set_ylabel("Time t", fontsize=12)
    ax1_lambda.set_ylabel("λ", fontsize=12, color='red')
    ax1_lambda.tick_params(axis='y', labelcolor='red')
    
    # Plot CD HMC ridge plot (pre-equilibration)
    ax2.set_title("Counterdiabatic HMC Evolution (Pre-equilibration)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Position q", fontsize=12)
    ax2.set_ylabel("Time t", fontsize=12)
    ax2_lambda.set_ylabel("λ", fontsize=12, color='red')
    ax2_lambda.tick_params(axis='y', labelcolor='red')
    
    # Plot CD HMC ridge plot (post-equilibration) if available
    if has_re_equil:
        ax3.set_title("Counterdiabatic HMC Evolution (Post-equilibration)", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Position q", fontsize=12)
        ax3.set_ylabel("Time t", fontsize=12)
        ax3_lambda.set_ylabel("λ", fontsize=12, color='red')
        ax3_lambda.tick_params(axis='y', labelcolor='red')
    
    # Find global range for consistent x-axis
    all_qs = np.concatenate([snapshots['naive'] + snapshots['cd_pre_equil']])
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
    
    # Plot naive HMC distributions
    for i, (t, naive_snap, lam_val) in enumerate(zip(times, snapshots['naive'], snapshots['lam_pre_equil'])):
        # Compute KDE for smooth curve
        try:
            kde = gaussian_kde(naive_snap.flatten())
            density = kde(x_grid)
        except:
            # Fallback to histogram if KDE fails
            hist, bin_edges = np.histogram(naive_snap, bins=50, density=True, range=(x_min, x_max))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            density = np.interp(x_grid, bin_centers, hist)
        
        # Normalize and offset for ridge plot - increased height for more overlap
        density = density / np.max(density) * 1.8  # Increased from 1.2 to 1.8 for more overlap
        offset = t
        
        # Plot the ridge with transparency for overlap
        ax1.fill_between(x_grid, offset, offset + density, 
                        color='blue', alpha=0.4, edgecolor='blue', linewidth=0.5)
        
        # Add true distribution at each time step
        potential_fn = make_V(lam_val)
        rho = np.array(jax.vmap(lambda x: jnp.exp(-potential_fn(x)))(x_grid))
        rho = rho / np.max(rho) * 1.8  # Scale to match
        ax1.plot(x_grid, offset + rho, 'k--', linewidth=1.5, alpha=0.8)
    
    # Plot CD HMC distributions (pre-equilibration)
    for i, (t, cd_snap, lam_val) in enumerate(zip(times, snapshots['cd_pre_equil'], snapshots['lam_pre_equil'])):
        # Compute KDE for smooth curve
        try:
            kde = gaussian_kde(cd_snap.flatten())
            density = kde(x_grid)
        except:
            # Fallback to histogram if KDE fails
            hist, bin_edges = np.histogram(cd_snap, bins=50, density=True, range=(x_min, x_max))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            density = np.interp(x_grid, bin_centers, hist)
        
        # Normalize and offset for ridge plot - increased height for more overlap
        density = density / np.max(density) * 1.8  # Increased from 1.2 to 1.8 for more overlap
        offset = t
        
        # Plot the ridge with transparency for overlap
        ax2.fill_between(x_grid, offset, offset + density, 
                        color='red', alpha=0.4, edgecolor='red', linewidth=0.5)
        
        # Add true distribution at each time step
        potential_fn = make_V(lam_val)
        rho = np.array(jax.vmap(lambda x: jnp.exp(-potential_fn(x)))(x_grid))
        rho = rho / np.max(rho) * 1.8  # Scale to match
        ax2.plot(x_grid, offset + rho, 'k--', linewidth=1.5, alpha=0.8)
    
    # Plot CD HMC distributions (post-equilibration) if available
    if has_re_equil:
        # Post-equilibration snapshots represent the state after CD step + re-equilibration
        # They should be plotted at the next timestep since re-equilibration happens at lam_k1
        for i, (cd_post_equil_snap, lam_val) in enumerate(zip(snapshots['cd_post_equil'], snapshots['lam_post_equil'])):
            # Compute KDE for smooth curve
            try:
                kde = gaussian_kde(cd_post_equil_snap.flatten())
                density = kde(x_grid)
            except:
                # Fallback to histogram if KDE fails
                hist, bin_edges = np.histogram(cd_post_equil_snap, bins=50, density=True, range=(x_min, x_max))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                density = np.interp(x_grid, bin_centers, hist)
            
            # Normalize and offset for ridge plot - increased height for more overlap
            density = density / np.max(density) * 1.8  # Increased from 1.2 to 1.8 for more overlap
            
            # Plot at the next timestep since re-equilibration happens after the CD step
            # The post-equilibration snapshot at index i corresponds to the state after the CD step
            # that was taken at snapshot time i, so it should be plotted at time (i+1) * delta_t * 10
            # But actually, it should be at the next timestep after the CD step, which is just delta_t later
            # So if the pre-equilibration snapshot is at time i * delta_t * 10, the post-equilibration
            # should be at time (i * delta_t * 10) + delta_t
            pre_equil_time = i * delta_t * 10
            next_timestep = pre_equil_time + delta_t
            offset = next_timestep
            
            # Plot the ridge with transparency for overlap
            ax3.fill_between(x_grid, offset, offset + density, 
                            color='orange', alpha=0.4, edgecolor='orange', linewidth=0.5)
            
            # Add true distribution at the next timestep
            # Use the stored lambda value from snapshots since it's now correct
            potential_fn = make_V(lam_val)
            rho = np.array(jax.vmap(lambda x: jnp.exp(-potential_fn(x)))(x_grid))
            rho = rho / np.max(rho) * 1.8  # Scale to match
            ax3.plot(x_grid, offset + rho, 'k--', linewidth=1.5, alpha=0.8)
    
    # Set consistent limits
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    ax1.set_ylim(times[0] - 0.1, times[-1] + 2.0)  # Increased upper limit for taller histograms
    ax2.set_ylim(times[0] - 0.1, times[-1] + 2.0)  # Increased upper limit for taller histograms
    
    if has_re_equil:
        # For post-equilibration, extend the y-axis to accommodate the next timestep
        # The last post-equilibration snapshot will be at time (len-1) * delta_t * 10 + delta_t
        max_post_equil_time = (len(snapshots['cd_post_equil']) - 1) * delta_t * 10 + delta_t
        max_time = max(times[-1], max_post_equil_time)
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(times[0] - 0.1, max_time + 2.0)  # Extended upper limit for post-equil
    
    # Set y-axis ticks only at the time points where distributions are plotted
    ax1.set_yticks(times)
    ax2.set_yticks(times)
    if has_re_equil:
        # Include both original times and the next timestep for post-equilibration
        post_equil_times = np.array([i * delta_t * 10 + delta_t for i in range(len(snapshots['cd_post_equil']))])
        all_times = np.concatenate([times, post_equil_times])
        ax3.set_yticks(all_times)
    
    # Configure lambda axes (secondary y-axes)
    # Get lambda values for the time points
    lambda_values = snapshots['lam_pre_equil']
    
    ax1_lambda.set_ylim(ax1.get_ylim())  # Same limits as time axis
    ax2_lambda.set_ylim(ax2.get_ylim())  # Same limits as time axis
    
    # Set lambda ticks at the same positions as time ticks
    ax1_lambda.set_yticks(times)
    ax2_lambda.set_yticks(times)
    
    # Map time positions to lambda values for tick labels
    lambda_tick_labels = [f"{lam:.3f}" for lam in lambda_values]
    ax1_lambda.set_yticklabels(lambda_tick_labels)
    ax2_lambda.set_yticklabels(lambda_tick_labels)
    
    if has_re_equil:
        ax3_lambda.set_ylim(ax3.get_ylim())  # Same limits as time axis
        ax3_lambda.set_yticks(all_times)
        
        # For post-equilibration, use the stored lambda values from snapshots
        post_equil_lambda_values = snapshots['lam_post_equil']
        
        # Combine lambda values for all time points
        all_lambda_values = lambda_values + post_equil_lambda_values
        all_lambda_tick_labels = [f"{lam:.3f}" for lam in all_lambda_values]
        ax3_lambda.set_yticklabels(all_lambda_tick_labels)
    
    # Add legends with true distribution reference
    ax1.plot([], [], 'k--', linewidth=1.5, label='True distribution')
    ax2.plot([], [], 'k--', linewidth=1.5, label='True distribution')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    if has_re_equil:
        ax3.plot([], [], 'k--', linewidth=1.5, label='True distribution')
        ax3.legend(loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{ansatz_dir}/ridge_plot_{potential_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_results(snapshots, loss_histories, delta_t, make_V, lam_fn, param_history=None, ansatz=None, potential_name="harmonic", dim=1, plot_ansatz=False):
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
    print(f"Available snapshot keys: {list(snapshots.keys())}")
    if 'detailed_energy_stats' in snapshots:
        print(f"Number of energy stats: {len(snapshots['detailed_energy_stats'])}")
    else:
        print("No detailed_energy_stats found in snapshots")
    
    # Check if re-equilibration was used
    has_re_equil = 'cd_post_equil' in snapshots and len(snapshots['cd_post_equil']) > 0
    
    # Diagnostic check: compare pre- and post-equilibration positions
    if has_re_equil:
        print("\n=== DIAGNOSTIC: Checking pre- vs post-equilibration positions ===")
        num_comparisons = min(len(snapshots['cd_pre_equil']), len(snapshots['cd_post_equil']))
        print(f"Number of snapshots to compare: {num_comparisons}")
        
        for i in range(num_comparisons):
            pre_pos = snapshots['cd_pre_equil'][i]
            post_pos = snapshots['cd_post_equil'][i]
            
            # Check if positions are identical
            if np.array_equal(pre_pos, post_pos):
                print(f"⚠️  WARNING: Pre- and post-equilibration positions are IDENTICAL at snapshot {i}")
                print(f"   Pre-equil lambda: {snapshots['lam_pre_equil'][i]:.4f}")
                print(f"   Post-equil lambda: {snapshots['lam_post_equil'][i]:.4f}")
                print(f"   Lambda difference: {abs(snapshots['lam_post_equil'][i] - snapshots['lam_pre_equil'][i]):.6f}")
                print(f"   Pre-equil positions range: [{np.min(pre_pos):.4f}, {np.max(pre_pos):.4f}]")
                print(f"   Post-equil positions range: [{np.min(post_pos):.4f}, {np.max(post_pos):.4f}]")
            else:
                # Calculate some statistics to show the difference
                diff = np.abs(post_pos - pre_pos)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"✓ Snapshot {i}: Positions differ - max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")
        
        print("=== END DIAGNOSTIC ===\n")

    # Create figures based on what we want to plot
    if plot_ansatz:
        fig1, axes1 = plt.subplots(4, 6, figsize=(32, 18))  # Increased width for legend space
        fig2, axes2 = plt.subplots(3, 6, figsize=(28, 14))
        axes1 = axes1.flatten()
        axes2 = axes2.flatten()
    else:
        # Only plot distributions, not ansatz visualization
        fig1, axes1 = plt.subplots(4, 6, figsize=(32, 18))  # Increased width for legend space
        axes1 = axes1.flatten()
    
    times = np.arange(len(snapshots['naive'])) * delta_t * 10  # *10 because we record every 10 steps
    
    # Plot loss histories
    if loss_histories:
        axes1[0].set_title("Loss during optimization")
        # maximum y value: 100
        # axes1[0].set_ylim(0, 100)
        axes1[0].set_xlabel("Optimization iteration")
        axes1[0].set_ylabel("Loss")
        for i, loss_history in enumerate(loss_histories):
            axes1[0].plot(loss_history, label=f'Fit {i+1}')
        axes1[0].legend()
    else:
        axes1[0].set_title("No Loss Data")
        axes1[0].text(0.5, 0.5, "Fitting not performed\n(e.g., Analytic Ansatz)", 
                      horizontalalignment='center', verticalalignment='center', 
                      transform=axes1[0].transAxes)

    # Plot parameter history if available
    if param_history is not None and len(param_history) > 0:
        param_times = np.arange(len(param_history)) * delta_t * 10
        axes1[1].set_title("Parameters over time")
        axes1[1].set_xlabel("t")
        axes1[1].set_ylabel("Value")

        if isinstance(ansatz, PolynomialAnsatz):
            # Get term descriptions for legend labels
            term_descriptions = ansatz.get_term_description()
            num_params = len(term_descriptions)
            
            # Create more compact legend labels
            if num_params <= 10:
                # For few parameters, use the full descriptions
                term_labels = [desc.split(": ")[1] for desc in term_descriptions]
            else:
                # For many parameters, use compact parameter numbers
                term_labels = [f"θ_{i+1}" for i in range(num_params)]
            
            # Plot each parameter with NaN/inf handling
            for i in range(param_history[0].shape[0]):
                param_values = [p[i] for p in param_history]
                
                # Check for NaN or infinite values
                param_values = np.array(param_values)
                if np.any(np.isnan(param_values)) or np.any(np.isinf(param_values)):
                    print(f"⚠️  Warning: Parameter θ_{i+1} contains NaN or infinite values")
                    # Replace NaN and inf with zeros for plotting
                    param_values = np.nan_to_num(param_values, nan=0.0, posinf=0.0, neginf=0.0)
                
                axes1[1].plot(param_times, param_values, label=term_labels[i])
            
            # Check if we have any valid data and set axis limits safely
            all_param_values = []
            for i in range(param_history[0].shape[0]):
                param_values = [p[i] for p in param_history]
                param_values = np.array(param_values)
                param_values = np.nan_to_num(param_values, nan=0.0, posinf=0.0, neginf=0.0)
                all_param_values.extend(param_values)
            
            if len(all_param_values) == 0 or all(np.array(all_param_values) == 0):
                axes1[1].text(0.5, 0.5, "No valid parameter data to plot\nAll parameters may be NaN/Inf", 
                             horizontalalignment='center', verticalalignment='center', 
                             transform=axes1[1].transAxes, fontsize=10)
                axes1[1].set_ylim(-1, 1)  # Set default limits
            else:
                # Set reasonable axis limits based on the data
                all_param_values = np.array(all_param_values)
                valid_values = all_param_values[np.isfinite(all_param_values)]
                if len(valid_values) > 0:
                    y_min, y_max = np.min(valid_values), np.max(valid_values)
                    y_range = y_max - y_min
                    if y_range == 0:
                        y_range = 1.0
                    axes1[1].set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                else:
                    axes1[1].set_ylim(-1, 1)  # Fallback limits
                
            # Improve legend readability
            if num_params <= 10:
                # For few parameters, use standard legend
                axes1[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            else:
                # For many parameters, use multi-column legend outside plot
                axes1[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                               fontsize=6, ncol=2, columnspacing=0.5)
                
                # Add a note about the compact notation
                axes1[1].text(0.02, 0.98, f"Legend shows {num_params} polynomial parameters\nUse θ₁, θ₂, ... for compact notation", 
                             transform=axes1[1].transAxes, fontsize=8, 
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
        elif isinstance(ansatz, AnalyticAnsatz):
            param_values = [p[0] for p in param_history]
            
            # Check for NaN or infinite values
            param_values = np.array(param_values)
            if np.any(np.isnan(param_values)) or np.any(np.isinf(param_values)):
                print(f"⚠️  Warning: Analytic parameter λ contains NaN or infinite values")
                # Replace NaN and inf with zeros for plotting
                param_values = np.nan_to_num(param_values, nan=0.0, posinf=0.0, neginf=0.0)
            
            axes1[1].plot(param_times, param_values, label="λ")
            
            # Check if we have valid data and set axis limits safely
            if len(param_values) == 0 or all(param_values == 0):
                axes1[1].text(0.5, 0.5, "No valid parameter data to plot\nParameter may be NaN/Inf", 
                             horizontalalignment='center', verticalalignment='center', 
                             transform=axes1[1].transAxes, fontsize=10)
                axes1[1].set_ylim(-1, 1)  # Set default limits
            else:
                # Set reasonable axis limits based on the data
                valid_values = param_values[np.isfinite(param_values)]
                if len(valid_values) > 0:
                    y_min, y_max = np.min(valid_values), np.max(valid_values)
                    y_range = y_max - y_min
                    if y_range == 0:
                        y_range = 1.0
                    axes1[1].set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                else:
                    axes1[1].set_ylim(-1, 1)  # Fallback limits
            
            axes1[1].legend()
        # Note: Plotting for NN params is not implemented due to complexity

    # Plot energy change statistics if available
    if 'detailed_energy_stats' in snapshots and len(snapshots['detailed_energy_stats']) > 0:
        energy_times = snapshots['detailed_times']
        
        # Extract energy change statistics
        naive_avg_delta_H = [stats['naive']['avg_delta_H'] for stats in snapshots['detailed_energy_stats']]
        naive_avg_delta_H_sq = [stats['naive']['avg_delta_H_sq'] for stats in snapshots['detailed_energy_stats']]
        cd_avg_delta_H = [stats['cd']['avg_delta_H'] for stats in snapshots['detailed_energy_stats']]
        cd_avg_delta_H_sq = [stats['cd']['avg_delta_H_sq'] for stats in snapshots['detailed_energy_stats']]
        
        # Plot energy change statistics
        axes1[2].set_title("Energy Change Statistics over Time")
        axes1[2].set_xlabel("t")
        axes1[2].set_ylabel("Value")
        
        # Plot average ΔH
        axes1[2].plot(energy_times, naive_avg_delta_H, 'b-', label='⟨ΔH⟩ (Naive)', alpha=0.7)
        axes1[2].plot(energy_times, cd_avg_delta_H, 'r-', label='⟨ΔH⟩ (CD)', alpha=0.7)
        
        # Plot average (ΔH)²
        axes1[2].plot(energy_times, naive_avg_delta_H_sq, 'b--', label='⟨(ΔH)²⟩ (Naive)', alpha=0.7)
        axes1[2].plot(energy_times, cd_avg_delta_H_sq, 'r--', label='⟨(ΔH)²⟩ (CD)', alpha=0.7)
        
        axes1[2].legend()
        axes1[2].grid(True, alpha=0.3)
    else:
        axes1[2].set_title("No Energy Change Statistics")
        axes1[2].text(0.5, 0.5, "Energy change statistics not available", 
                      horizontalalignment='center', verticalalignment='center', 
                      transform=axes1[2].transAxes)

    # Plot ∂H/∂λ statistics if available
    if 'detailed_energy_stats' in snapshots and len(snapshots['detailed_energy_stats']) > 0:
        energy_times = snapshots['detailed_times']
        
        # Extract ∂H/∂λ statistics
        naive_avg_dH_dlam = [stats['naive']['avg_dH_dlam'] for stats in snapshots['detailed_energy_stats']]
        naive_avg_dH_dlam_sq = [stats['naive']['avg_dH_dlam_sq'] for stats in snapshots['detailed_energy_stats']]
        cd_avg_dH_dlam = [stats['cd']['avg_dH_dlam'] for stats in snapshots['detailed_energy_stats']]
        cd_avg_dH_dlam_sq = [stats['cd']['avg_dH_dlam_sq'] for stats in snapshots['detailed_energy_stats']]
        
        # Plot ∂H/∂λ statistics
        axes1[3].set_title("∂H/∂λ Statistics over Time")
        axes1[3].set_xlabel("t")
        axes1[3].set_ylabel("Value")
        
        # Plot average ∂H/∂λ
        axes1[3].plot(energy_times, naive_avg_dH_dlam, 'b-', label='⟨∂H/∂λ⟩ (Naive)', alpha=0.7)
        axes1[3].plot(energy_times, cd_avg_dH_dlam, 'r-', label='⟨∂H/∂λ⟩ (CD)', alpha=0.7)
        
        # Plot average (∂H/∂λ)²
        axes1[3].plot(energy_times, naive_avg_dH_dlam_sq, 'b--', label='⟨(∂H/∂λ)²⟩ (Naive)', alpha=0.7)
        axes1[3].plot(energy_times, cd_avg_dH_dlam_sq, 'r--', label='⟨(∂H/∂λ)²⟩ (CD)', alpha=0.7)
        
        axes1[3].legend()
        axes1[3].grid(True, alpha=0.3)
    else:
        axes1[3].set_title("No ∂H/∂λ Statistics")
        axes1[3].text(0.5, 0.5, "∂H/∂λ statistics not available", 
                      horizontalalignment='center', verticalalignment='center', 
                      transform=axes1[3].transAxes)

    num_hist_axes = 13  # Reduced to make room for parameter plot
    num_snaps = len(snapshots['naive'])
    if num_snaps > num_hist_axes:
        selected_indices = np.linspace(0, num_snaps - 1, num_hist_axes, dtype=int)
    else:
        selected_indices = np.arange(num_snaps)

    # Find global min and max for consistent x-axis (for 1D case)
    if dim == 1:
        all_qs = np.concatenate([snapshots['naive'][i] for i in selected_indices] + 
                               [snapshots['cd_pre_equil'][i] for i in selected_indices])
        if has_re_equil:
            all_qs = np.concatenate([all_qs] + [snapshots['cd_post_equil'][i] for i in selected_indices if i < len(snapshots['cd_post_equil'])])
        x_min = np.min(all_qs) - 0.5
        x_max = np.max(all_qs) + 0.5

    for plot_idx, snap_idx in enumerate(selected_indices):
        # Plot distributions
        ax1 = axes1[plot_idx + 4]  # +4 because we have loss, parameter, energy change, and ∂H/∂λ plots at the start
        naive_snap = snapshots['naive'][snap_idx]
        cd_pre_equil_snap = snapshots['cd_pre_equil'][snap_idx]
        lam_val = snapshots['lam_pre_equil'][snap_idx]

        if dim == 1:
            # 1D case: filled histograms without transparency
            # Use dynamic binning to avoid "too many bins" error
            def safe_hist(data, bins=25):
                """Create histogram with dynamic binning to handle edge cases."""
                if len(data) == 0:
                    return np.zeros(bins), np.linspace(x_min, x_max, bins + 1)
                
                # Check if data has enough variation for the requested number of bins
                data_range = np.max(data) - np.min(data)
                if data_range < 1e-10:  # Very small range
                    # Use fewer bins for nearly constant data
                    actual_bins = min(5, len(data))
                else:
                    # Use adaptive binning based on data range
                    actual_bins = min(bins, max(5, int(data_range * 10)))
                
                return np.histogram(data, bins=actual_bins, density=True, range=(x_min, x_max))
            
            # Plot histograms with safe binning
            naive_hist, naive_bins = safe_hist(naive_snap)
            cd_pre_hist, cd_pre_bins = safe_hist(cd_pre_equil_snap)
            
            # Plot histograms using bar plot for more control
            bin_centers = (naive_bins[:-1] + naive_bins[1:]) / 2
            ax1.bar(bin_centers, naive_hist, width=naive_bins[1]-naive_bins[0], 
                   color='blue', alpha=0.5, label='Naïve HMC')
            
            bin_centers = (cd_pre_bins[:-1] + cd_pre_bins[1:]) / 2
            ax1.bar(bin_centers, cd_pre_hist, width=cd_pre_bins[1]-cd_pre_bins[0], 
                   color='red', alpha=0.5, label='CD HMC (pre-equil)')
            
            # Add post-equilibration CD distribution if available
            # Note: Post-equilibration snapshots are stored at the same index but represent the next timestep
            if has_re_equil and snap_idx < len(snapshots['cd_post_equil']):
                cd_post_equil_snap = snapshots['cd_post_equil'][snap_idx]
                lam_post_equil = snapshots['lam_post_equil'][snap_idx]
                
                cd_post_hist, cd_post_bins = safe_hist(cd_post_equil_snap)
                bin_centers = (cd_post_bins[:-1] + cd_post_bins[1:]) / 2
                ax1.bar(bin_centers, cd_post_hist, width=cd_post_bins[1]-cd_post_bins[0], 
                       color='orange', alpha=0.5, label='CD HMC (post-equil)')
                
                # Plot true distribution for post-equilibration state
                # Use the stored lambda value from snapshots since it's now correct
                lam_post_equil = snapshots['lam_post_equil'][snap_idx]
                xs_post_equil = np.linspace(x_min, x_max, 400)
                potential_fn_post_equil = make_V(lam_post_equil)
                rho_post_equil = np.array(jax.vmap(lambda x: jnp.exp(-potential_fn_post_equil(x)))(xs_post_equil))
                rho_post_equil /= np.trapezoid(rho_post_equil, xs_post_equil)
                ax1.plot(xs_post_equil, rho_post_equil, 'g-', lw=2, label=f'True (post-equil, λ={lam_post_equil:.2f})', alpha=0.7)
            
            # Plot true distribution for pre-equilibration state
            xs = np.linspace(x_min, x_max, 400)
            potential_fn = make_V(lam_val)
            rho = np.array(jax.vmap(lambda x: jnp.exp(-potential_fn(x)))(xs))
            rho /= np.trapezoid(rho, xs)
            ax1.plot(xs, rho, 'k-', lw=2, label=f'True (pre-equil, λ={lam_val:.2f})')
            ax1.set_title(f"t={snap_idx*10*delta_t:.2f}, lam={lam_val:.2f}")
            ax1.set_xlabel("q")
            ax1.set_ylabel("Density")
            ax1.set_xlim(x_min, x_max)
            ax1.legend()
        
        elif dim == 2:
            # 2D case: scatter plots - plot both on same axis, no transparency
            ax1.scatter(naive_snap[:, 0], naive_snap[:, 1], alpha=1.0, s=1, color='blue', label='Naïve')
            ax1.scatter(cd_pre_equil_snap[:, 0], cd_pre_equil_snap[:, 1], alpha=1.0, s=1, color='red', label='CD (pre-equil)')
            
            # Add post-equilibration CD distribution if available
            if has_re_equil and snap_idx < len(snapshots['cd_post_equil']):
                cd_post_equil_snap = snapshots['cd_post_equil'][snap_idx]
                ax1.scatter(cd_post_equil_snap[:, 0], cd_post_equil_snap[:, 1], alpha=1.0, s=1, color='orange', label='CD (post-equil)')
            
            ax1.set_xlabel('q_0')
            ax1.set_ylabel('q_1')
            ax1.set_title(f"t={snap_idx*10*delta_t:.2f}, λ={lam_val:.2f}")
            ax1.set_aspect('equal')
            ax1.legend()

        # Plot learned ansatz
        if plot_ansatz:
            ax2 = axes2[plot_idx + 3]  # +3 because we have loss, parameter, and energy plots at the start
            if param_history is not None and snap_idx < len(param_history):
                theta = param_history[snap_idx]
                if dim == 1:
                    plot_learned_ansatz(ax2, theta, ansatz, q_range=(x_min, x_max), p_range=(-3, 3), dim=dim)
                elif dim == 2:
                    plot_learned_ansatz(ax2, theta, ansatz, q_range=(-3, 3), p_range=(-3, 3), dim=dim)
                
                if isinstance(ansatz, AnalyticAnsatz):
                    ax2.set_title(f"Analytic A(q,p) at t={snap_idx*10*delta_t:.2f}")
                else:
                    ax2.set_title(f"Learned A(q,p) at t={snap_idx*10*delta_t:.2f}")

    # Save figures with potential information in filename
    try:
        plt.figure(fig1.number)
        plt.tight_layout()
        plt.savefig(f"{ansatz_dir}/distributions_{potential_name}.png", dpi=300, bbox_inches='tight')
        print(f"Saved distributions plot to {ansatz_dir}/distributions_{potential_name}.png")
        
        # Create parameter legend for polynomial ansatz
        if isinstance(ansatz, PolynomialAnsatz):
            create_parameter_legend(ansatz, potential_name, ansatz_type)
            create_dedicated_parameter_plot(param_history, ansatz, delta_t, potential_name, ansatz_type)
        
        # Create ridge plot for 1D systems (publication quality)
        if dim == 1:
            create_ridge_plot(snapshots, delta_t, make_V, lam_fn, potential_name, ansatz_type)
            print(f"Saved ridge plot to {ansatz_dir}/ridge_plot_{potential_name}.png")
        
        if plot_ansatz:
            plt.figure(fig2.number)
            plt.tight_layout()
            plt.savefig(f"{ansatz_dir}/ansatz_{potential_name}.png", dpi=300, bbox_inches='tight')
            print(f"Saved ansatz plot to {ansatz_dir}/ansatz_{potential_name}.png")
    except Exception as e:
        print(f"Error saving plots: {e}")
        import traceback
        traceback.print_exc() 

def create_parameter_legend(ansatz, potential_name, ansatz_type):
    """Create a separate file showing the mapping between parameter numbers and polynomial terms."""
    if not isinstance(ansatz, PolynomialAnsatz):
        return
    
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    ansatz_dir = f"figures/{ansatz_type}"
    os.makedirs(ansatz_dir, exist_ok=True)
    
    term_descriptions = ansatz.get_term_description()
    num_params = len(term_descriptions)
    
    # Create a simple text file with the parameter mapping
    legend_file = f"{ansatz_dir}/parameter_legend_{potential_name}.txt"
    with open(legend_file, 'w') as f:
        f.write(f"Parameter Legend for {potential_name} (Polynomial Ansatz)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total parameters: {num_params}\n\n")
        f.write("Compact notation -> Full polynomial term:\n")
        f.write("-" * 40 + "\n")
        
        for i, desc in enumerate(term_descriptions):
            param_num = i + 1
            term_part = desc.split(": ")[1]
            f.write(f"θ_{param_num:2d} -> {term_part}\n")
    
    print(f"Parameter legend saved to {legend_file}") 

def create_dedicated_parameter_plot(param_history, ansatz, delta_t, potential_name, ansatz_type):
    """Create a separate dedicated plot for parameters with better legend handling."""
    if not isinstance(ansatz, PolynomialAnsatz) or param_history is None or len(param_history) == 0:
        return
    
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    ansatz_dir = f"figures/{ansatz_type}"
    os.makedirs(ansatz_dir, exist_ok=True)
    
    param_times = np.arange(len(param_history)) * delta_t * 10
    term_descriptions = ansatz.get_term_description()
    num_params = len(term_descriptions)
    
    # Create a larger figure specifically for parameters
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot each parameter with a different color
    colors = plt.cm.tab20(np.linspace(0, 1, num_params))
    
    for i in range(param_history[0].shape[0]):
        param_values = [p[i] for p in param_history]
        
        # Check for NaN or infinite values
        param_values = np.array(param_values)
        if np.any(np.isnan(param_values)) or np.any(np.isinf(param_values)):
            print(f"⚠️  Warning: Parameter θ_{i+1} contains NaN or infinite values in dedicated plot")
            # Replace NaN and inf with zeros for plotting
            param_values = np.nan_to_num(param_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        ax.plot(param_times, param_values, 
                color=colors[i], linewidth=1.5, alpha=0.8, label=f"θ_{i+1}")
    
    ax.set_title(f"Polynomial Parameters over Time - {potential_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("Parameter Value", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Check if we have any valid data to plot
    all_param_values = []
    for i in range(param_history[0].shape[0]):
        param_values = [p[i] for p in param_history]
        param_values = np.array(param_values)
        param_values = np.nan_to_num(param_values, nan=0.0, posinf=0.0, neginf=0.0)
        all_param_values.extend(param_values)
    
    if len(all_param_values) == 0 or all(np.array(all_param_values) == 0):
        ax.text(0.5, 0.5, "No valid parameter data to plot\nAll parameters may be NaN/Inf", 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_ylim(-1, 1)  # Set default limits
    else:
        # Set reasonable axis limits based on the data
        all_param_values = np.array(all_param_values)
        valid_values = all_param_values[np.isfinite(all_param_values)]
        if len(valid_values) > 0:
            y_min, y_max = np.min(valid_values), np.max(valid_values)
            y_range = y_max - y_min
            if y_range == 0:
                y_range = 1.0
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        else:
            ax.set_ylim(-1, 1)  # Fallback limits
    
    # Create a comprehensive legend
    if num_params <= 15:
        # For moderate number of parameters, show legend with full terms
        legend_labels = []
        for i, desc in enumerate(term_descriptions):
            term_part = desc.split(": ")[1]
            legend_labels.append(f"θ_{i+1}: {term_part}")
        
        # Place legend outside the plot
        ax.legend(legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', 
                 fontsize=8, ncol=1, columnspacing=0.5)
    else:
        # For many parameters, use compact legend with reference to separate file
        legend_labels = [f"θ_{i+1}" for i in range(num_params)]
        ax.legend(legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', 
                 fontsize=6, ncol=2, columnspacing=0.5)
        
        # Add note about parameter mapping
        ax.text(0.02, 0.98, f"Showing {num_params} polynomial parameters\nSee parameter_legend_{potential_name}.txt for full terms", 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Adjust layout to accommodate legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for legend
    
    # Save the dedicated parameter plot
    param_plot_file = f"{ansatz_dir}/parameters_{potential_name}.png"
    plt.savefig(param_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved dedicated parameter plot to {param_plot_file}") 