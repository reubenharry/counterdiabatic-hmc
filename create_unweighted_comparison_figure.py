#!/usr/bin/env python3
"""
Script to create a comprehensive figure comparing CD-HMC vs naive HMC
for three systems using unweighted simulations.

The figure will have:
- 3 columns: moving mean, annealing, double well
- 2 rows: naive HMC, CD-HMC
- Each subplot shows KDE ridgeplots of snapshots over time
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.stats import gaussian_kde
from src.systems import get_system

def compute_kde(samples, x_grid):
    """Compute KDE for 1D samples."""
    try:
        kde = gaussian_kde(samples.flatten())
        density = kde(x_grid)
        return density
    except:
        # Fallback to histogram if KDE fails
        hist, _ = np.histogram(samples.flatten(), bins=50, density=True)
        bin_centers = (_[:-1] + _[1:]) / 2
        # Interpolate to x_grid
        density = np.interp(x_grid, bin_centers, hist)
        return density

def create_ridge_subplot(ax, snapshots, delta_t, make_V, title, color='blue', show_true_dist=True, x_range=None, time_range=None):
    """Create a single ridge subplot for one method and system."""
    
    # Get time points and select only 3 steps (initial, middle, final)
    total_steps = len(snapshots['particles'])
    step_indices = [0, total_steps // 2, total_steps - 1]  # Initial, middle, final
    
    # Get actual times and lambda values from the snapshots
    if 'times' not in snapshots:
        raise ValueError(f"No 'times' key found in snapshots for {title}")
    
    # Use the saved times and lambda values directly
    times = [snapshots['times'][i] for i in step_indices]
    selected_particles = [snapshots['particles'][i] for i in step_indices]
    selected_lams = [snapshots['lam'][i] for i in step_indices]
    
    # Use provided x_range or find range for this subplot
    if x_range is not None:
        x_min, x_max = x_range
    else:
        all_qs = np.concatenate(selected_particles).flatten()
        x_min = np.min(all_qs) - 0.5
        x_max = np.max(all_qs) + 0.5
    
    # Create x grid for smooth curves
    x_grid = np.linspace(x_min, x_max, 200)
    
    # Plot distributions at each time step with more overlap
    for i, (t, snap, lam_val) in enumerate(zip(times, selected_particles, selected_lams)):
        # Compute KDE for smooth curve
        density = compute_kde(snap, x_grid)
        
        # Normalize and offset for ridge plot with more overlap and flatter histograms
        density = density / np.max(density) * 0.08  # Much flatter histograms
        
        # Use compressed visual spacing but keep original time values for labels
        offset = t * 0.05  # Even more compressed spacing for maximum overlap
        
        # Plot the ridge with transparency
        ax.fill_between(x_grid, offset, offset + density, 
                       color=color, alpha=0.4, edgecolor=color, linewidth=0.5)
        
        # Add true distribution if requested
        if show_true_dist:
            potential_fn = make_V(lam_val)
            rho = np.array(jax.vmap(lambda x: jnp.exp(-potential_fn(x)))(x_grid))
            rho = rho / np.max(rho) * 0.08  # Scale to match the much flatter histograms
            ax.plot(x_grid, offset + rho, 'k--', linewidth=1.0, alpha=0.7)
    
    # Set axis properties
    ax.set_xlim(x_min, x_max)
    
    # Compress visual spacing but keep original time values for labels
    y_min, y_max = min(times) * 0.05, max(times) * 0.05
    ax.set_ylim(y_min, y_max + 0.1)  # Remove space below t=0.0
    ax.set_yticks([t * 0.05 for t in times])  # Visual positions are compressed
    ax.set_yticklabels([f"{t:.1f}" for t in times])  # But show original time values
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add lambda values on secondary y-axis with clear labels
    lambda_tick_labels = [f"λ={lam:.1f}" for lam in selected_lams]
    ax_lambda = ax.twinx()
    ax_lambda.set_ylim(ax.get_ylim())
    ax_lambda.set_yticks([t * 0.05 for t in times])  # Visual positions are compressed
    ax_lambda.set_yticklabels(lambda_tick_labels)  # Show original lambda values
    ax_lambda.tick_params(axis='y', labelcolor='red')

def load_simulation_data(system_name):
    """Load unweighted simulation data for a given system."""
    
    # Load naive HMC data
    naive_file = f"data/{system_name}_naive_unweighted.pkl"
    with open(naive_file, 'rb') as f:
        naive_data = pickle.load(f)
    
    # Load CD-HMC data
    cd_file = f"data/{system_name}_cd_unweighted.pkl"
    with open(cd_file, 'rb') as f:
        cd_data = pickle.load(f)
    
    return naive_data, cd_data

def create_comparison_figure():
    """Create the main comparison figure."""
    
    # Systems to compare with their colors
    systems = [
        ('gaussian_moving_mean', 'Gaussian Moving Mean', 'blue'),
        ('gaussian_annealing', 'Gaussian Annealing', 'green'),
        ('double_well', 'Double Well', 'red')
    ]
    
    # Create figure with subplots: 2 rows (naive, CD) x 3 columns (systems)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))  # Slightly increased height to 9
    
    # Set up lambda function for timing
    v = 0.5
    max_lam = 1.0
    lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
    
    for col, (system_name, system_display_name, color) in enumerate(systems):
        print(f"Processing {system_display_name}...")
        
        # Load data
        naive_data, cd_data = load_simulation_data(system_name)
        
        # Get system functions
        make_T, make_V, description, dim = get_system(system_name)
        
        # Get simulation parameters
        delta_t = naive_data.get('delta_t', 0.2)
        
        # Calculate x-axis range for this column (system)
        # Get all particles from both naive and CD simulations for this system
        all_naive_particles = np.concatenate(naive_data['snapshots']['particles']).flatten()
        all_cd_particles = np.concatenate(cd_data['snapshots']['particles']).flatten()
        all_particles = np.concatenate([all_naive_particles, all_cd_particles])
        
        # Calculate x-axis range for this column
        x_min = np.min(all_particles) - 0.5
        x_max = np.max(all_particles) + 0.5
        x_range = (x_min, x_max)
        
        # Get time ranges for this column to ensure consistent y-axis scaling
        if 'times' not in naive_data['snapshots'] or 'times' not in cd_data['snapshots']:
            raise ValueError(f"No 'times' key found in snapshots for {system_name}")
        
        naive_times = naive_data['snapshots']['times']
        cd_times = cd_data['snapshots']['times']
        
        # Use the time range from this specific system for y-axis scaling
        all_times = naive_times + cd_times
        time_range = (min(all_times), max(all_times))
        
        # Create subplots for this system
        naive_ax = axes[0, col]
        cd_ax = axes[1, col]
        
        # Create ridge plots with system-specific colors and aligned x-axis
        create_ridge_subplot(naive_ax, naive_data['snapshots'], delta_t, make_V, 
                           f"Naive HMC - {system_display_name}", color=color, x_range=x_range, time_range=time_range)
        create_ridge_subplot(cd_ax, cd_data['snapshots'], delta_t, make_V, 
                           f"CD-HMC - {system_display_name}", color=color, x_range=x_range, time_range=time_range)
        
        # Set x-axis labels only for bottom row
        if col == 0:  # First column
            naive_ax.set_ylabel("Time t", fontsize=12)
            cd_ax.set_ylabel("Time t", fontsize=12)
        
        cd_ax.set_xlabel("Position q", fontsize=12)
    
    # Add overall title
    fig.suptitle("Comparison of Naive HMC vs CD-HMC: Unweighted Simulations", 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add legend for true distribution
    fig.legend([plt.Line2D([], [], color='k', linestyle='--', linewidth=1.0)], 
              ['True distribution'], loc='upper right', bbox_to_anchor=(0.98, 0.95))
    
    # Adjust layout
    plt.tight_layout()
    
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    # Save figure
    plt.savefig("figures/unweighted_comparison_ridgeplots.png", dpi=300, bbox_inches='tight')
    plt.savefig("figures/unweighted_comparison_ridgeplots.pdf", bbox_inches='tight')
    
    print("Figure saved as:")
    print("  - figures/unweighted_comparison_ridgeplots.png")
    print("  - figures/unweighted_comparison_ridgeplots.pdf")
    
    # plt.show()

if __name__ == "__main__":
    create_comparison_figure()
