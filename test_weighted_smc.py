#!/usr/bin/env python3
"""
Test script for weighted Sequential Monte Carlo (SMC) functionality.
This script runs both weighted and unweighted naive HMC simulations to demonstrate
the difference in tracking the true distribution over time.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import run_simulation
from src.systems import get_system
from src.ansatze import PolynomialAnsatz
from src.plotting import create_ridge_plot

def main():
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # System parameters
    system_name = 'gaussian_moving_mean'  # Simple 1D system for demonstration
    make_T, make_V, description, dim = get_system(system_name)
    
    print(f"Testing weighted SMC with system: {system_name}")
    print(f"Description: {description}")
    print(f"Dimension: {dim}")
    
    # Simulation parameters
    M = 1000  # Number of particles
    N_steps = 100  # Number of time steps
    delta_t = 0.01  # Time step size
    eps = 0.05  # HMC step size
    momentum_refresh_interval = 4.0  # Momentum refresh interval
    fit_every = 10  # How often to refit the ansatz
    num_initial_iterations = 1000  # Initial fitting iterations
    num_iterations = 100  # Subsequent fitting iterations
    learning_rate = 1e-4  # Learning rate for fitting
    
    # Lambda function (time-varying parameter)
    def lam_fn(t):
        return 0.5 * t  # Linear increase from 0 to 0.5
    
    def dot_lam_fn(t):
        return 0.5  # Constant rate of change
    
    # Create a dummy ansatz (not used for naive HMC, but required by interface)
    dummy_ansatz = PolynomialAnsatz(max_degree=2, dim=dim)
    
    print("\nRunning unweighted naive HMC simulation...")
    # Run simulation without weights
    key, subkey = jax.random.split(key)
    A_ansatz_unweighted, snapshots_unweighted, loss_histories_unweighted, param_history_unweighted = run_simulation(
        M=M, N_steps=N_steps, delta_t=delta_t, eps=eps, 
        momentum_refresh_interval=momentum_refresh_interval, fit_every=fit_every,
        num_initial_iterations=num_initial_iterations, num_iterations=num_iterations,
        make_T=make_T, make_V=make_V, A_ansatz=dummy_ansatz, lam_fn=lam_fn, 
        dot_lam_fn=dot_lam_fn, key=subkey, dim=dim, learning_rate=learning_rate,
        use_weights=False
    )
    
    print("\nRunning weighted SMC simulation with resampling...")
    # Run simulation with weights and resampling
    key, subkey = jax.random.split(key)
    A_ansatz_weighted, snapshots_weighted, loss_histories_weighted, param_history_weighted = run_simulation(
        M=M, N_steps=N_steps, delta_t=delta_t, eps=eps, 
        momentum_refresh_interval=momentum_refresh_interval, fit_every=fit_every,
        num_initial_iterations=num_initial_iterations, num_iterations=num_iterations,
        make_T=make_T, make_V=make_V, A_ansatz=dummy_ansatz, lam_fn=lam_fn, 
        dot_lam_fn=dot_lam_fn, key=subkey, dim=dim, learning_rate=learning_rate,
        use_weights=True, ess_threshold=0.5  # Enable resampling when ESS < 50%
    )
    
    print("\nCreating comparison plots...")
    
    # Create ridge plot for weighted simulation (shows both weighted and unweighted)
    create_ridge_plot(
        snapshots_weighted, delta_t, make_V, lam_fn, 
        potential_name=f"{system_name}_weighted_comparison", 
        ansatz_type="polynomial"
    )
    
    # Create a simple comparison plot showing the difference
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot final distributions
    final_time_idx = -1
    
    # Unweighted final distribution
    final_unweighted = snapshots_unweighted['naive'][final_time_idx]
    ax1.hist(final_unweighted.flatten(), bins=50, density=True, alpha=0.7, 
             label='Unweighted HMC', color='blue')
    
    # Weighted final distribution
    final_weighted = snapshots_weighted['naive_weighted'][final_time_idx]
    final_weights = snapshots_weighted['weights'][final_time_idx]
    
    if final_weights is not None:
        # Convert log weights to regular weights
        weights = np.exp(final_weights - np.max(final_weights))
        weights = weights / np.sum(weights)
        
        # Create weighted histogram
        ax2.hist(final_weighted.flatten(), bins=50, weights=weights, density=True, 
                 alpha=0.7, label='Weighted SMC', color='green')
    else:
        ax2.hist(final_weighted.flatten(), bins=50, density=True, alpha=0.7, 
                 label='Weighted SMC', color='green')
    
    # Add true distribution
    final_lam = snapshots_unweighted['lam'][final_time_idx]
    x_range = np.linspace(np.min(final_unweighted) - 1, np.max(final_unweighted) + 1, 200)
    potential_fn = make_V(final_lam)
    true_density = np.array([jnp.exp(-potential_fn(x)) for x in x_range])
    true_density = true_density / np.sum(true_density) * len(x_range) / (x_range[-1] - x_range[0])
    
    ax1.plot(x_range, true_density, 'r-', linewidth=2, label='True distribution')
    ax2.plot(x_range, true_density, 'r-', linewidth=2, label='True distribution')
    
    ax1.set_title(f'Unweighted HMC (λ = {final_lam:.3f})')
    ax1.set_xlabel('Position q')
    ax1.set_ylabel('Density')
    ax1.legend()
    
    ax2.set_title(f'Weighted SMC (λ = {final_lam:.3f})')
    ax2.set_xlabel('Position q')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'figures/polynomial/final_distribution_comparison_{system_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to figures/polynomial/")
    print(f"- Ridge plot: ridge_plot_{system_name}_weighted_comparison.png")
    print(f"- Final distribution comparison: final_distribution_comparison_{system_name}.png")
    
    # Print some statistics about the weights
    if snapshots_weighted['weights'][-1] is not None:
        final_log_weights = snapshots_weighted['weights'][-1]
        weights = np.exp(final_log_weights - np.max(final_log_weights))
        weights = weights / np.sum(weights)
        
        print(f"\nWeight statistics (final timestep):")
        print(f"- Effective sample size: {1.0 / np.sum(weights**2):.1f}")
        print(f"- Weight variance: {np.var(weights):.6f}")
        print(f"- Max weight: {np.max(weights):.6f}")
        print(f"- Min weight: {np.min(weights):.6f}")
        print(f"- Weight entropy: {-np.sum(weights * np.log(weights + 1e-10)):.3f}")
        
        # Print resampling statistics
        if 'resampling_events' in snapshots_weighted:
            resampling_events = snapshots_weighted['resampling_events']
            total_resampling = resampling_events[-1] if resampling_events else 0
            print(f"\nResampling statistics:")
            print(f"- Total resampling events: {total_resampling}")
            if total_resampling > 0:
                print(f"- Resampling frequency: {total_resampling / N_steps:.3f} events per step")
                print(f"- Average ESS before resampling: {M * ess_threshold:.1f}")

if __name__ == "__main__":
    main()
