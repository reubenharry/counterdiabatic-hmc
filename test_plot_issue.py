#!/usr/bin/env python3
"""
Test script to debug the plotting issue with missing histograms.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def test_plot_method(method_name):
    """Test plotting for a specific method."""
    print(f"\n=== TESTING {method_name} ===")
    
    # Load data
    with open(f'data/double_well_{method_name}.pkl', 'rb') as f:
        data = pickle.load(f)
    
    snapshots = data['snapshots']
    particles = snapshots['particles']
    times = snapshots['times']
    lam_values = snapshots['lam']
    
    print(f"Number of time steps: {len(particles)}")
    
    # Test each time step
    for i, (t, p, lam) in enumerate(zip(times, particles, lam_values)):
        print(f"\nTime step {i} (t={t:.1f}, λ={lam:.3f}):")
        
        # Check particle data
        p_flat = p.flatten()
        print(f"  Particles - min: {np.min(p_flat):.6f}, max: {np.max(p_flat):.6f}, mean: {np.mean(p_flat):.6f}")
        print(f"  Particles - any NaN: {np.any(np.isnan(p_flat))}, any Inf: {np.any(np.isinf(p_flat))}")
        
        # Create x grid
        x_min = np.min(p_flat) - 0.5
        x_max = np.max(p_flat) + 0.5
        x_grid = np.linspace(x_min, x_max, 200)
        
        # Test KDE computation
        try:
            kde = gaussian_kde(p_flat)
            density = kde(x_grid)
            print(f"  KDE - min: {np.min(density):.6f}, max: {np.max(density):.6f}, mean: {np.mean(density):.6f}")
            print(f"  KDE - any NaN: {np.any(np.isnan(density))}, any Inf: {np.any(np.isinf(density))}")
            
            # Test normalization
            if np.max(density) > 0:
                density_norm = density / np.max(density) * 1.8
                print(f"  Normalized - min: {np.min(density_norm):.6f}, max: {np.max(density_norm):.6f}")
            else:
                print(f"  WARNING: Max density is zero!")
                
        except Exception as e:
            print(f"  ERROR in KDE computation: {e}")
        
        # Test potential computation
        try:
            # Simple double well potential for testing
            def make_V(lam):
                def V(q):
                    return (1-lam)*0.5*q**2 + lam*(q**2 - 3)**2
                return V
            
            potential_fn = make_V(lam)
            rho = np.array([np.exp(-potential_fn(x)) for x in x_grid])
            print(f"  Potential - min: {np.min(rho):.6f}, max: {np.max(rho):.6f}")
            print(f"  Potential - any NaN: {np.any(np.isnan(rho))}, any Inf: {np.any(np.isinf(rho))}")
            
            if np.max(rho) > 0:
                rho_norm = rho / np.max(rho) * 1.8
                print(f"  Potential normalized - min: {np.min(rho_norm):.6f}, max: {np.max(rho_norm):.6f}")
            else:
                print(f"  WARNING: Max potential is zero!")
                
        except Exception as e:
            print(f"  ERROR in potential computation: {e}")

def main():
    methods = ['naive_unweighted', 'naive_weighted', 'cd_unweighted', 'cd_weighted']
    
    for method in methods:
        test_plot_method(method)

if __name__ == "__main__":
    main()
