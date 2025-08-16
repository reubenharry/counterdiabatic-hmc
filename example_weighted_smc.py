#!/usr/bin/env python3
"""
Simple example demonstrating weighted Sequential Monte Carlo (SMC) functionality.

This example shows how to run a naive HMC simulation with importance weights
to track the true distribution as the potential energy changes over time.
"""

import jax
import jax.numpy as jnp
import numpy as np
from src.simulation import run_simulation
from src.systems import get_system
from src.ansatze import PolynomialAnsatz
from src.plotting import create_ridge_plot

def main():
    # Set random seed
    key = jax.random.PRNGKey(42)
    
    # Get system (1D Gaussian with moving mean)
    make_T, make_V, description, dim = get_system('gaussian_moving_mean')
    
    # Simulation parameters
    M = 500  # Number of particles
    N_steps = 50  # Number of time steps
    delta_t = 0.02  # Time step size
    
    # Lambda function: mean moves from 0 to 1 over time
    def lam_fn(t):
        return t  # Linear increase from 0 to 1
    
    def dot_lam_fn(t):
        return 1.0  # Constant rate of change
    
    # Create dummy ansatz (not used for naive HMC)
    dummy_ansatz = PolynomialAnsatz(max_degree=2, dim=dim)
    
    print("Running weighted SMC simulation...")
    print(f"System: {description}")
    print(f"Particles: {M}, Steps: {N_steps}")
    
    # Run simulation with weights and resampling enabled
    A_ansatz, snapshots, loss_histories, param_history = run_simulation(
        M=M, N_steps=N_steps, delta_t=delta_t, eps=0.05,
        momentum_refresh_interval=4.0, fit_every=10,
        num_initial_iterations=500, num_iterations=50,
        make_T=make_T, make_V=make_V, A_ansatz=dummy_ansatz,
        lam_fn=lam_fn, dot_lam_fn=dot_lam_fn, key=key, dim=dim,
        learning_rate=1e-4, use_weights=True, ess_threshold=0.5  # Enable weights and resampling
    )
    
    # Create visualization
    create_ridge_plot(
        snapshots, delta_t, make_V, lam_fn,
        potential_name="example_weighted_smc", ansatz_type="polynomial"
    )
    
    print("\nSimulation completed!")
    print("The ridge plot shows:")
    print("- Left: Unweighted naive HMC (shows lag behind true distribution)")
    print("- Middle: Weighted SMC (should track true distribution better)")
    print("- Right: Counterdiabatic HMC (for comparison)")
    print("\nCheck figures/polynomial/ridge_plot_example_weighted_smc.png")
    
    # Print weight and resampling statistics
    if snapshots['weights'][-1] is not None:
        final_weights = np.exp(snapshots['weights'][-1] - np.max(snapshots['weights'][-1]))
        final_weights = final_weights / np.sum(final_weights)
        
        print(f"\nFinal weight statistics:")
        print(f"- Effective sample size: {1.0 / np.sum(final_weights**2):.1f}")
        print(f"- Weight variance: {np.var(final_weights):.6f}")
        
        # Print resampling statistics
        if 'resampling_events' in snapshots:
            resampling_events = snapshots['resampling_events']
            total_resampling = resampling_events[-1] if resampling_events else 0
            print(f"\nResampling statistics:")
            print(f"- Total resampling events: {total_resampling}")
            if total_resampling > 0:
                print(f"- Resampling frequency: {total_resampling / N_steps:.3f} events per step")

if __name__ == "__main__":
    main()
