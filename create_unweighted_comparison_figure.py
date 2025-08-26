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

def create_ridge_subplot(ax, snapshots, delta_t, make_V, title, color='blue', show_true_dist=True):
    """Create a single ridge subplot for one method and system."""
    
    # Get time points and select only 3 steps (initial, middle, final)
    total_steps = len(snapshots['particles'])
    step_indices = [0, total_steps // 2, total_steps - 1]  # Initial, middle, final
    
    times = np.array([snapshots['lam'][i] for i in step_indices])  # Use lambda values as time
    selected_particles = [snapshots['particles'][i] for i in step_indices]
    selected_lams = [snapshots['lam'][i] for i in step_indices]
    
    # Find global range for consistent x-axis
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
        density = density / np.max(density) * 0.4  # Further reduced height to flatten
        offset = i * 0.2  # Further reduced spacing for more overlap
        
        # Plot the ridge with transparency
        ax.fill_between(x_grid, offset, offset + density, 
                       color=color, alpha=0.4, edgecolor=color, linewidth=0.5)
        
        # Add true distribution if requested
        if show_true_dist:
            potential_fn = make_V(lam_val)
            rho = np.array(jax.vmap(lambda x: jnp.exp(-potential_fn(x)))(x_grid))
            rho = rho / np.max(rho) * 0.4  # Scale to match
            ax.plot(x_grid, offset + rho, 'k--', linewidth=1.0, alpha=0.7)
    
    # Set axis properties
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.1, 0.8)  # Reduced range for flatter histograms
    ax.set_yticks([0, 0.2, 0.4])  # Adjusted tick positions for new spacing
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add lambda values on secondary y-axis with clear labels
    ax_lambda = ax.twinx()
    ax_lambda.set_ylim(ax.get_ylim())
    ax_lambda.set_yticks([0, 0.2, 0.4])
    lambda_tick_labels = [f"λ={lam:.1f}" for lam in selected_lams]
    ax_lambda.set_yticklabels(lambda_tick_labels)
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
        
        # Create subplots for this system
        naive_ax = axes[0, col]
        cd_ax = axes[1, col]
        
        # Create ridge plots with system-specific colors
        create_ridge_subplot(naive_ax, naive_data['snapshots'], delta_t, make_V, 
                           f"Naive HMC - {system_display_name}", color=color)
        create_ridge_subplot(cd_ax, cd_data['snapshots'], delta_t, make_V, 
                           f"CD-HMC - {system_display_name}", color=color)
        
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
