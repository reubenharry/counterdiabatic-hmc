#!/usr/bin/env python3
"""
Debug script to investigate ridge plot differences between ansatz types.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from src.simulation import run_simulation, run_naive_hmc_simulation
from src.ansatze import PolynomialAnsatz, NeuralNetworkAnsatz
from src.systems import get_system

def debug_ridge_plot_differences():
    """Compare naive HMC results between polynomial and neural network ansatzes."""
    
    # Set up the system
    system_name = "double_well"
    make_T, make_V, system_description, dim = get_system(system_name)
    
    # Define lambda functions
    v = 0.5
    max_lam = 1.0
    lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
    dot_lam_fn = jax.grad(lam_fn)
    
    # Parameters
    M = 1000
    N_steps = 40
    delta_t = 0.05
    eps = 0.05
    momentum_refresh_interval = 10.0
    
    # Test with polynomial ansatz
    print("Testing with Polynomial Ansatz...")
    ansatz_poly = PolynomialAnsatz(max_degree=5, dim=dim)
    
    # Run naive HMC
    key = jax.random.PRNGKey(0)
    snapshots_poly = run_naive_hmc_simulation(
        M=M, N_steps=N_steps, delta_t=delta_t, eps=eps,
        momentum_refresh_interval=momentum_refresh_interval,
        make_T=make_T, make_V=make_V, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
        key=key, dim=dim, use_weights=False
    )
    
    # Test with neural network ansatz
    print("Testing with Neural Network Ansatz...")
    ansatz_nn = NeuralNetworkAnsatz(dims=[2, 32, 64, 1], dim=dim, key=jax.random.PRNGKey(1))
    
    # Run naive HMC with same seed
    key = jax.random.PRNGKey(0)  # Same seed as polynomial
    snapshots_nn = run_naive_hmc_simulation(
        M=M, N_steps=N_steps, delta_t=delta_t, eps=eps,
        momentum_refresh_interval=momentum_refresh_interval,
        make_T=make_T, make_V=make_V, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
        key=key, dim=dim, use_weights=False
    )
    
    # Compare the data
    print("\n=== COMPARISON ===")
    print(f"Polynomial naive snapshots shape: {len(snapshots_poly['naive'])}")
    print(f"Neural network naive snapshots shape: {len(snapshots_nn['naive'])}")
    
    # Check first snapshot
    poly_first = snapshots_poly['naive'][0]
    nn_first = snapshots_nn['naive'][0]
    print(f"Polynomial first snapshot shape: {poly_first.shape}")
    print(f"Neural network first snapshot shape: {nn_first.shape}")
    
    # Check ranges
    poly_all = np.concatenate(snapshots_poly['naive'])
    nn_all = np.concatenate(snapshots_nn['naive'])
    print(f"Polynomial range: [{np.min(poly_all):.3f}, {np.max(poly_all):.3f}]")
    print(f"Neural network range: [{np.min(nn_all):.3f}, {np.max(nn_all):.3f}]")
    
    # Check if they're identical
    if np.allclose(poly_all, nn_all):
        print("✓ Snapshots are identical!")
    else:
        print("✗ Snapshots are different!")
        print(f"Max difference: {np.max(np.abs(poly_all - nn_all))}")
    
    # Check lambda values
    poly_lam = snapshots_poly['lam']
    nn_lam = snapshots_nn['lam']
    print(f"Polynomial lambda values: {poly_lam[:5]}")
    print(f"Neural network lambda values: {nn_lam[:5]}")
    
    if np.allclose(poly_lam, nn_lam):
        print("✓ Lambda values are identical!")
    else:
        print("✗ Lambda values are different!")
    
    # Now test the x-axis range calculation issue
    print("\n=== TESTING X-AXIS RANGE ISSUE ===")
    
    # Simulate what happens in create_comparison_plots
    def calculate_x_range(all_snapshots):
        all_qs = []
        for method, snapshots in all_snapshots.items():
            if method == 'naive_unweighted':
                all_qs.extend(snapshots['naive'])
            elif method == 'naive_weighted':
                all_qs.extend(snapshots['naive_weighted'])
            elif method == 'cd_unweighted':
                all_qs.extend(snapshots['cd_pre_equil'])
            elif method == 'cd_weighted':
                all_qs.extend(snapshots['cd_weighted'])
        
        x_min = np.min(np.concatenate(all_qs)) - 0.5
        x_max = np.max(np.concatenate(all_qs)) + 0.5
        return x_min, x_max
    
    # Test with only naive HMC data
    naive_only_poly = {'naive_unweighted': snapshots_poly}
    naive_only_nn = {'naive_unweighted': snapshots_nn}
    
    x_min_poly_naive, x_max_poly_naive = calculate_x_range(naive_only_poly)
    x_min_nn_naive, x_max_nn_naive = calculate_x_range(naive_only_nn)
    
    print(f"X-range with only naive HMC (polynomial): [{x_min_poly_naive:.3f}, {x_max_poly_naive:.3f}]")
    print(f"X-range with only naive HMC (neural): [{x_min_nn_naive:.3f}, {x_max_nn_naive:.3f}]")
    
    # Now simulate what happens when CD-HMC data is included
    # We need to run CD-HMC to get the actual data
    print("\nRunning CD-HMC to test x-axis range calculation...")
    
    # Run CD-HMC with polynomial ansatz
    key = jax.random.PRNGKey(0)
    _, snapshots_cd_poly, _, _ = run_simulation(
        M=M, N_steps=N_steps, delta_t=delta_t, eps=eps,
        momentum_refresh_interval=momentum_refresh_interval,
        fit_every=1, num_initial_iterations=10, num_iterations=10,
        make_T=make_T, make_V=make_V, A_ansatz=ansatz_poly, 
        lam_fn=lam_fn, dot_lam_fn=dot_lam_fn, key=key, dim=dim, 
        learning_rate=1e-5, re_equil_steps=0, use_weights=False
    )
    
    # Run CD-HMC with neural network ansatz
    key = jax.random.PRNGKey(0)
    _, snapshots_cd_nn, _, _ = run_simulation(
        M=M, N_steps=N_steps, delta_t=delta_t, eps=eps,
        momentum_refresh_interval=momentum_refresh_interval,
        fit_every=1, num_initial_iterations=10, num_iterations=10,
        make_T=make_T, make_V=make_V, A_ansatz=ansatz_nn, 
        lam_fn=lam_fn, dot_lam_fn=dot_lam_fn, key=key, dim=dim, 
        learning_rate=1e-5, re_equil_steps=0, use_weights=False
    )
    
    # Test x-range with CD-HMC included
    with_cd_poly = {
        'naive_unweighted': snapshots_poly,
        'cd_unweighted': snapshots_cd_poly
    }
    with_cd_nn = {
        'naive_unweighted': snapshots_nn,
        'cd_unweighted': snapshots_cd_nn
    }
    
    x_min_poly_with_cd, x_max_poly_with_cd = calculate_x_range(with_cd_poly)
    x_min_nn_with_cd, x_max_nn_with_cd = calculate_x_range(with_cd_nn)
    
    print(f"X-range with CD-HMC (polynomial): [{x_min_poly_with_cd:.3f}, {x_max_poly_with_cd:.3f}]")
    print(f"X-range with CD-HMC (neural): [{x_min_nn_with_cd:.3f}, {x_max_nn_with_cd:.3f}]")
    
    # Check CD-HMC ranges
    cd_poly_all = np.concatenate(snapshots_cd_poly['cd_pre_equil'])
    cd_nn_all = np.concatenate(snapshots_cd_nn['cd_pre_equil'])
    print(f"CD-HMC range (polynomial): [{np.min(cd_poly_all):.3f}, {np.max(cd_poly_all):.3f}]")
    print(f"CD-HMC range (neural): [{np.min(cd_nn_all):.3f}, {np.max(cd_nn_all):.3f}]")
    
    # Check if CD-HMC data is different
    if np.allclose(cd_poly_all, cd_nn_all):
        print("✓ CD-HMC snapshots are identical!")
    else:
        print("✗ CD-HMC snapshots are different!")
        print(f"Max difference: {np.max(np.abs(cd_poly_all - cd_nn_all))}")
        print(f"Mean difference: {np.mean(np.abs(cd_poly_all - cd_nn_all))}")
    
    # Check individual CD-HMC snapshots
    print(f"CD-HMC polynomial snapshots shape: {len(snapshots_cd_poly['cd_pre_equil'])}")
    print(f"CD-HMC neural snapshots shape: {len(snapshots_cd_nn['cd_pre_equil'])}")
    
    for i in range(min(len(snapshots_cd_poly['cd_pre_equil']), len(snapshots_cd_nn['cd_pre_equil']))):
        poly_snap = snapshots_cd_poly['cd_pre_equil'][i]
        nn_snap = snapshots_cd_nn['cd_pre_equil'][i]
        if np.allclose(poly_snap, nn_snap):
            print(f"✓ CD-HMC snapshot {i} identical")
        else:
            print(f"✗ CD-HMC snapshot {i} different, max diff: {np.max(np.abs(poly_snap - nn_snap))}")
    
    # Now let's test the actual KDE computation
    print("\n=== TESTING KDE COMPUTATION ===")
    
    # Test KDE on the same data
    test_snap = snapshots_poly['naive'][0]  # First snapshot
    x_grid = np.linspace(-4, 4, 200)
    
    try:
        kde_poly = gaussian_kde(test_snap.flatten())
        y_kde_poly = kde_poly(x_grid)
        print(f"KDE computation successful for polynomial data")
        print(f"KDE range: [{np.min(y_kde_poly):.6f}, {np.max(y_kde_poly):.6f}]")
    except Exception as e:
        print(f"KDE computation failed for polynomial data: {e}")
    
    try:
        kde_nn = gaussian_kde(test_snap.flatten())  # Same data
        y_kde_nn = kde_nn(x_grid)
        print(f"KDE computation successful for neural data (same as polynomial)")
        print(f"KDE range: [{np.min(y_kde_nn):.6f}, {np.max(y_kde_nn):.6f}]")
    except Exception as e:
        print(f"KDE computation failed for neural data: {e}")
    
    # Check if KDE results are identical
    if np.allclose(y_kde_poly, y_kde_nn):
        print("✓ KDE results are identical!")
    else:
        print("✗ KDE results are different!")
        print(f"Max difference: {np.max(np.abs(y_kde_poly - y_kde_nn))}")
    
    # Test the actual plotting function
    print("\n=== TESTING ACTUAL PLOTTING ===")
    
    # Import the plotting function from main.py
    import sys
    sys.path.append('.')
    
    # Create a simple test plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot polynomial data
    times = np.arange(len(snapshots_poly['naive'])) * 0.05 * 10
    x_grid = np.linspace(-4, 4, 200)
    
    for j, (t, snap) in enumerate(zip(times, snapshots_poly['naive'])):
        try:
            kde = gaussian_kde(snap.flatten())
            y_kde = kde(x_grid)
            y_kde = y_kde / np.max(y_kde) * 0.3
            ax1.fill_between(x_grid, t * 2.0 - y_kde, t * 2.0 + y_kde, 
                           alpha=0.6, color='blue')
        except Exception as e:
            print(f"Error plotting polynomial data at time {t}: {e}")
    
    ax1.set_title("Polynomial Ansatz (Naive HMC)")
    ax1.set_xlabel("Position q")
    ax1.set_ylabel("Time t")
    
    # Plot neural network data (should be identical)
    for j, (t, snap) in enumerate(zip(times, snapshots_nn['naive'])):
        try:
            kde = gaussian_kde(snap.flatten())
            y_kde = kde(x_grid)
            y_kde = y_kde / np.max(y_kde) * 0.3
            ax2.fill_between(x_grid, t * 2.0 - y_kde, t * 2.0 + y_kde, 
                           alpha=0.6, color='red')
        except Exception as e:
            print(f"Error plotting neural data at time {t}: {e}")
    
    ax2.set_title("Neural Network Ansatz (Naive HMC)")
    ax2.set_xlabel("Position q")
    ax2.set_ylabel("Time t")
    
    plt.tight_layout()
    plt.savefig("debug_ridge_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved debug comparison plot to debug_ridge_comparison.png")
    
    return snapshots_poly, snapshots_nn

if __name__ == "__main__":
    snapshots_poly, snapshots_nn = debug_ridge_plot_differences()
