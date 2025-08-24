#!/usr/bin/env python3
"""
Test script to demonstrate resampling functionality in weighted SMC.
This script uses a more challenging system that should trigger resampling.
"""

import jax
import jax.numpy as jnp
import numpy as np
from src.simulation import run_simulation
from src.systems import get_system
from src.ansatze import PolynomialAnsatz

def main():
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Use a more challenging system: double well potential
    system_name = 'double_well'
    make_T, make_V, description, dim = get_system(system_name)
    
    print(f"Testing resampling with system: {system_name}")
    print(f"Description: {description}")
    print(f"Dimension: {dim}")
    
    # Simulation parameters designed to trigger resampling
    M = 500  # Number of particles
    N_steps = 50  # Number of time steps
    delta_t = 0.05  # Larger time step to create more dramatic changes
    eps = 0.05  # HMC step size
    momentum_refresh_interval = 4.0
    fit_every = 10
    num_initial_iterations = 500
    num_iterations = 50
    learning_rate = 1e-4
    
    # Lambda function: rapid transition from single well to double well
    def lam_fn(t):
        return 0.8 * t  # Fast transition from 0 to 0.4
    
    def dot_lam_fn(t):
        return 0.8  # Constant rate of change
    
    # Create dummy ansatz
    dummy_ansatz = PolynomialAnsatz(max_degree=2, dim=dim)
    
    print(f"\nSimulation parameters:")
    print(f"- Particles: {M}")
    print(f"- Steps: {N_steps}")
    print(f"- Time step: {delta_t}")
    print(f"- Lambda range: {lam_fn(0):.1f} to {lam_fn(N_steps * delta_t):.1f}")
    print(f"- ESS threshold: 0.3 (30% of particles)")
    
    # Run simulation with resampling
    print(f"\nRunning weighted SMC with resampling...")
    A_ansatz, snapshots, loss_histories, param_history = run_simulation(
        M=M, N_steps=N_steps, delta_t=delta_t, eps=eps,
        momentum_refresh_interval=momentum_refresh_interval, fit_every=fit_every,
        num_initial_iterations=num_initial_iterations, num_iterations=num_iterations,
        make_T=make_T, make_V=make_V, A_ansatz=dummy_ansatz, lam_fn=lam_fn,
        dot_lam_fn=dot_lam_fn, key=key, dim=dim, learning_rate=learning_rate,
        use_weights=True, ess_threshold=0.3  # Lower threshold to trigger resampling
    )
    
    # Analyze results
    if snapshots['weights'][-1] is not None:
        final_log_weights = snapshots['weights'][-1]
        weights = np.exp(final_log_weights - np.max(final_log_weights))
        weights = weights / np.sum(weights)
        
        print(f"\nFinal statistics:")
        print(f"- Effective sample size: {1.0 / np.sum(weights**2):.1f}/{M}")
        print(f"- ESS ratio: {(1.0 / np.sum(weights**2)) / M:.3f}")
        print(f"- Weight variance: {np.var(weights):.6f}")
        print(f"- Max weight: {np.max(weights):.6f}")
        print(f"- Min weight: {np.min(weights):.6f}")
        
        # Resampling analysis
        if 'resampling_events' in snapshots:
            resampling_events = snapshots['resampling_events']
            total_resampling = resampling_events[-1] if resampling_events else 0
            
            print(f"\nResampling analysis:")
            print(f"- Total resampling events: {total_resampling}")
            if total_resampling > 0:
                print(f"- Resampling frequency: {total_resampling / N_steps:.3f} events per step")
                print(f"- Average ESS before resampling: {M * 0.3:.1f}")
                
                # Show resampling timeline
                print(f"\nResampling timeline:")
                for i, (t, resampling_count) in enumerate(zip(range(0, N_steps + 1, 1), resampling_events)):
                    if i > 0:
                        events_since_last = resampling_count - resampling_events[i-1]
                        if events_since_last > 0:
                            print(f"  Step {i*10}: {events_since_last} resampling events")
            else:
                print(f"- No resampling occurred (ESS stayed above threshold)")
        
        # Show weight evolution
        print(f"\nWeight evolution:")
        for i, (t, weights_snap) in enumerate(zip(range(0, N_steps + 1, 1), snapshots['weights'])):
            if weights_snap is not None:
                weights_np = np.exp(weights_snap - np.max(weights_snap))
                weights_np = weights_np / np.sum(weights_np)
                ess = 1.0 / np.sum(weights_np ** 2)
                ess_ratio = ess / M
                print(f"  Step {i*10}: ESS = {ess:.1f}/{M} ({ess_ratio:.3f})")

if __name__ == "__main__":
    main()
