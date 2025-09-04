#!/usr/bin/env python3
"""
Standalone script to create histogram-based comparison ridge plots.
This script can be run independently to generate histogram plots from existing data.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def create_comparison_histogram_plots_standalone(all_snapshots, delta_t, make_V, system_name, dim, n_bins=50):
    """Create comparison plots using histograms instead of KDE for all simulation methods."""
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

def make_V_gaussian_annealing(lam):
    """Gaussian potential with annealing temperature: V(q) = 0.5 * k(λ) * q^2 where k interpolates from 1 (var=1) to 10 (var=0.1)"""
    # k = 1 at λ=0 (var=1), k = 10 at λ=1 (var=0.1)
    k = 1.0 + 9.0 * lam
    return lambda q: 0.5 * np.sum(k * (q ** 2))

def main():
    """Main function to create histogram plots from existing data."""
    
    print("Creating histogram-based comparison ridge plots...")
    
    # Load all existing data for gaussian annealing
    try:
        with open('data/gaussian_annealing_naive_unweighted.pkl', 'rb') as f:
            naive_unweighted = pickle.load(f)
        
        with open('data/gaussian_annealing_naive_weighted.pkl', 'rb') as f:
            naive_weighted = pickle.load(f)
        
        with open('data/gaussian_annealing_cd_unweighted.pkl', 'rb') as f:
            cd_unweighted = pickle.load(f)
        
        with open('data/gaussian_annealing_cd_weighted.pkl', 'rb') as f:
            cd_weighted = pickle.load(f)
        
        print("Data loaded successfully!")
        
        # Create the data structure expected by the plotting function
        successful_simulations = {
            'naive_unweighted': naive_unweighted['snapshots'],
            'naive_weighted': naive_weighted['snapshots'],
            'cd_unweighted': cd_unweighted['snapshots'],
            'cd_weighted': cd_weighted['snapshots']
        }
        
        # Get delta_t from the data
        delta_t = cd_unweighted.get('delta_t', 0.2)
        
        print(f"Delta t: {delta_t}")
        print(f"Number of methods: {len(successful_simulations)}")
        
        # Create the histogram plots
        create_comparison_histogram_plots_standalone(
            successful_simulations, 
            delta_t, 
            make_V_gaussian_annealing, 
            'gaussian_annealing', 
            dim=1,
            n_bins=50
        )
        
        print("Histogram plots created successfully!")
        print("Check figures/polynomial/comparison_histogram_plot_gaussian_annealing.png")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data file: {e}")
        print("Please run the main simulation first to generate the data files.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
