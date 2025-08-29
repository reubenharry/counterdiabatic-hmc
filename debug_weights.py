#!/usr/bin/env python3
"""
Debug script to examine weight values and their effect on distributions.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def compute_weighted_kde(data, weights=None, x_grid=None):
    """Compute weighted KDE."""
    if weights is None:
        # Unweighted KDE
        kde = gaussian_kde(data)
        if x_grid is not None:
            return kde(x_grid)
        else:
            return kde
    else:
        # Weighted KDE
        kde = gaussian_kde(data, weights=weights)
        if x_grid is not None:
            return kde(x_grid)
        else:
            return kde

def main():
    # Load the weighted simulation data
    with open('data/gaussian_annealing_cd_weighted.pkl', 'rb') as f:
        weighted_data = pickle.load(f)
    
    print("=== WEIGHT DEBUGGING ===")
    print(f"Keys in weighted data: {list(weighted_data.keys())}")
    
    # Get snapshots from the data
    snapshots = weighted_data['snapshots']
    print(f"Keys in snapshots: {list(snapshots.keys())}")
    
    # Examine weights
    weights = snapshots['weights']
    particles = snapshots['particles']
    times = snapshots['times']
    lam_values = snapshots['lam']
    
    print(f"\nNumber of time steps: {len(weights)}")
    print(f"Number of particles: {len(particles[0])}")
    
    # Check weight statistics at each time step
    for i, (t, lam, w, p) in enumerate(zip(times, lam_values, weights, particles)):
        if w is not None:
            # Convert log weights to regular weights
            w_array = np.exp(w - np.max(w))
            
            print(f"\nTime step {i} (t={t:.1f}, λ={lam:.3f}):")
            print(f"  Log weights - min: {np.min(w):.6f}, max: {np.max(w):.6f}, mean: {np.mean(w):.6f}, std: {np.std(w):.6f}")
            print(f"  Regular weights - min: {np.min(w_array):.6f}, max: {np.max(w_array):.6f}, mean: {np.mean(w_array):.6f}, std: {np.std(w_array):.6f}")
            print(f"  Weight sum: {np.sum(w_array):.6f}")
            print(f"  Effective sample size: {np.sum(w_array)**2 / np.sum(w_array**2):.2f}")
            
            # Check if weights are all the same (which would make weighted = unweighted)
            if np.allclose(w_array, w_array[0]):
                print(f"  WARNING: All weights are identical! Weighted = unweighted")
            else:
                print(f"  Weights vary - weighted should differ from unweighted")
        else:
            print(f"\nTime step {i} (t={t:.1f}, λ={lam:.3f}): No weights")
    
    # Compare distributions at a specific time step
    time_idx = 2  # Middle time step
    t = times[time_idx]
    lam = lam_values[time_idx]
    w = weights[time_idx]
    p = particles[time_idx]
    
    print(f"\n=== DISTRIBUTION COMPARISON AT TIME {t} (λ={lam:.3f}) ===")
    
    if w is not None:
        w_array = np.exp(w - np.max(w))
        
        # Create x grid
        x_min = np.min(p) - 0.5
        x_max = np.max(p) + 0.5
        x_grid = np.linspace(x_min, x_max, 200)
        
        # Compute both distributions
        density_unweighted = compute_weighted_kde(p.flatten(), weights=None, x_grid=x_grid)
        density_weighted = compute_weighted_kde(p.flatten(), weights=w_array, x_grid=x_grid)
        
        # Normalize for comparison
        density_unweighted = density_unweighted / np.max(density_unweighted)
        density_weighted = density_weighted / np.max(density_weighted)
        
        # Check if they're identical
        if np.allclose(density_unweighted, density_weighted, rtol=1e-6):
            print("  WARNING: Weighted and unweighted distributions are identical!")
        else:
            print("  Weighted and unweighted distributions differ")
            diff = np.abs(density_weighted - density_unweighted)
            print(f"  Max difference: {np.max(diff):.6f}")
            print(f"  Mean difference: {np.mean(diff):.6f}")
        
        # Plot comparison
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(p.flatten(), bins=30, density=True, alpha=0.7, label='Histogram')
        plt.plot(x_grid, density_unweighted, 'r-', linewidth=2, label='Unweighted KDE')
        plt.plot(x_grid, density_weighted, 'b-', linewidth=2, label='Weighted KDE')
        plt.title(f'Distributions at t={t:.1f}, λ={lam:.3f}')
        plt.xlabel('Position q')
        plt.ylabel('Density')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(x_grid, density_unweighted, 'r-', linewidth=2, label='Unweighted')
        plt.plot(x_grid, density_weighted, 'b-', linewidth=2, label='Weighted')
        plt.title('KDE Comparison')
        plt.xlabel('Position q')
        plt.ylabel('Density')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(x_grid, np.abs(density_weighted - density_unweighted), 'g-', linewidth=2)
        plt.title('Absolute Difference')
        plt.xlabel('Position q')
        plt.ylabel('|Weighted - Unweighted|')
        
        plt.tight_layout()
        plt.savefig('debug_weight_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved comparison plot to debug_weight_comparison.png")
    
    # Also check the weight values themselves
    print(f"\n=== WEIGHT VALUE EXAMINATION ===")
    for i, (t, lam, w) in enumerate(zip(times, lam_values, weights)):
        if w is not None:
            w_array = np.exp(w - np.max(w))
            print(f"Time {i} (t={t:.1f}, λ={lam:.3f}):")
            print(f"  First 5 log weights: {w[:5]}")
            print(f"  First 5 regular weights: {w_array[:5]}")
            print(f"  Weight range: {np.min(w_array):.6f} to {np.max(w_array):.6f}")
            print()

if __name__ == "__main__":
    main()
