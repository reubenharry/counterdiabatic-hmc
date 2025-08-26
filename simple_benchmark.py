#!/usr/bin/env python3
"""
Simple benchmarking script comparing naive HMC and CD-HMC without weights.
Focuses on the error metric: (fhat - E[f])^2/var[f] where f(x) = x^2.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from src.simulation import simulate
from src.ansatze import PolynomialAnsatz
from src.systems import get_system

def calculate_normalized_error(samples, true_expectation, true_variance, f_func=None):
    """
    Calculate the normalized error: (fhat - E[f])^2/var[f]
    
    Args:
        samples: Array of samples
        true_expectation: True expectation E[f]
        true_variance: True variance var[f]
        f_func: Function to apply to samples (default: f(x) = x^2)
    
    Returns:
        Normalized error value and estimate
    """
    if f_func is None:
        f_func = lambda x: x**2  # Default: f(x) = x^2
    
    # Apply function to samples
    f_vals = f_func(samples.flatten())
    
    # Calculate Monte Carlo estimate (unweighted)
    fhat = np.mean(f_vals)
    
    # Calculate normalized error
    normalized_error = (fhat - true_expectation)**2 / true_variance
    
    return normalized_error, fhat

def calculate_gaussian_moving_mean_expectations(lam_final):
    """
    Calculate true expectations for 1D Gaussian moving mean system.
    For V(q) = 0.5 * (q - λ)², the distribution is N(λ, 1).
    
    Args:
        lam_final: Final lambda value
    
    Returns:
        Dictionary with true expectations for different functions
    """
    mu = lam_final
    sigma = 1.0
    
    expectations = {
        'x': {
            'expectation': mu,  # E[x] = λ
            'variance': sigma**2  # Var[x] = 1
        },
        'x_squared': {
            'expectation': mu**2 + sigma**2,  # E[x²] = λ² + 1
            'variance': 2*sigma**4 + 4*mu**2*sigma**2  # Var[x²] = 2 + 4λ²
        },
        'x_cubed': {
            'expectation': mu**3 + 3*mu*sigma**2,  # E[x³] = λ³ + 3λ
            'variance': 6*sigma**6 + 9*mu**2*sigma**4 + 9*mu**4*sigma**2  # Var[x³] = 6 + 9λ² + 9λ⁴
        }
    }
    
    return expectations

def run_simple_benchmark():
    """Run simple benchmark comparing naive HMC and CD-HMC."""
    
    print("="*60)
    print("SIMPLE BENCHMARK: Naive HMC vs CD-HMC")
    print("="*60)
    print("System: Gaussian moving mean")
    print("Parameters:")
    print("  - delta_t = 0.2")
    print("  - N_steps = 10")
    print("  - Polynomial ansatz, degree 2")
    print("  - 100,000 training steps")
    print("  - No weights")
    print("="*60)
    
    # Set up the system
    make_T, make_V, system_description, dim = get_system('gaussian_moving_mean')
    print(f"System: {system_description}")
    print(f"Dimension: {dim}")
    
    # Define lambda functions for moving mean
    v = 0.5
    max_lam = 1.0
    lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
    dot_lam_fn = jax.grad(lam_fn)
    
    # Simulation parameters
    M = 500  # Number of particles
    N_steps = 10  # Number of steps
    delta_t = 0.2  # Time step
    momentum_refresh_interval = 5.0
    fit_every = 1
    num_initial_iterations = 100000  # Training steps
    num_iterations = 100000  # Training steps
    learning_rate = 1e-4
    use_weights = False  # No weights
    
    # Create polynomial ansatz
    ansatz = PolynomialAnsatz(max_degree=2, dim=dim)
    
    # Calculate true expectations for final lambda value
    lam_final = lam_fn(N_steps * delta_t)
    true_expectations = calculate_gaussian_moving_mean_expectations(lam_final)
    
    print(f"\nFinal lambda = {lam_final:.6f}")
    print(f"True expectations:")
    for f_name, exp_info in true_expectations.items():
        print(f"  {f_name}: E[f] = {exp_info['expectation']:.6f}, var[f] = {exp_info['variance']:.6f}")
    
    # Test functions
    functions = [
        (lambda x: x**2, 'x_squared'),
        (lambda x: x, 'x'),
        (lambda x: x**3, 'x_cubed')
    ]
    
    results = {}
    
    for f_func, f_name in functions:
        print(f"\n{'='*40}")
        print(f"TESTING FUNCTION: {f_name}")
        print(f"{'='*40}")
        
        true_exp = true_expectations[f_name]['expectation']
        true_var = true_expectations[f_name]['variance']
        
        print(f"True expectation E[f] = {true_exp:.6f}")
        print(f"True variance var[f] = {true_var:.6f}")
        
        # 1. Naive HMC
        print(f"\n1. Running Naive HMC...")
        key = jax.random.PRNGKey(42)
        start_time = time.time()
        
        try:
            snapshots_naive = simulate(
                simulation_type='naive',
                M=M, N_steps=N_steps, delta_t=delta_t, 
                momentum_refresh_interval=momentum_refresh_interval,
                make_T=make_T, make_V=make_V, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
                key=key, dim=dim, A_ansatz=ansatz, fit_every=fit_every,
                num_initial_iterations=num_initial_iterations, num_iterations=num_iterations,
                learning_rate=learning_rate, use_weights=use_weights
            )
            naive_time = time.time() - start_time
            
            # Get final samples
            final_samples_naive = snapshots_naive['particles'][-1]
            
            # Calculate error
            error_naive, estimate_naive = calculate_normalized_error(
                final_samples_naive, true_exp, true_var, f_func
            )
            
            print(f"   Time: {naive_time:.3f} s")
            print(f"   Estimate: {estimate_naive:.6f}")
            print(f"   Normalized Error: {error_naive:.6f}")
            
        except Exception as e:
            print(f"   Naive HMC failed: {e}")
            error_naive = float('inf')
            estimate_naive = float('nan')
            naive_time = float('inf')
        
        # 2. Counterdiabatic HMC
        print(f"\n2. Running CD-HMC...")
        key = jax.random.PRNGKey(42)  # Same seed for fair comparison
        start_time = time.time()
        
        try:
            _, snapshots_cd, _, _ = simulate(
                simulation_type='cd',
                M=M, N_steps=N_steps, delta_t=delta_t, 
                momentum_refresh_interval=momentum_refresh_interval,
                make_T=make_T, make_V=make_V, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
                key=key, dim=dim, A_ansatz=ansatz, fit_every=fit_every,
                num_initial_iterations=num_initial_iterations, num_iterations=num_iterations,
                learning_rate=learning_rate, use_weights=use_weights
            )
            cd_time = time.time() - start_time
            
            # Get final samples
            final_samples_cd = snapshots_cd['particles'][-1]
            
            # Calculate error
            error_cd, estimate_cd = calculate_normalized_error(
                final_samples_cd, true_exp, true_var, f_func
            )
            
            print(f"   Time: {cd_time:.3f} s")
            print(f"   Estimate: {estimate_cd:.6f}")
            print(f"   Normalized Error: {error_cd:.6f}")
            
        except Exception as e:
            print(f"   CD-HMC failed: {e}")
            error_cd = float('inf')
            estimate_cd = float('nan')
            cd_time = float('inf')
        
        # Store results
        results[f_name] = {
            'naive': {'error': error_naive, 'estimate': estimate_naive, 'time': naive_time},
            'cd': {'error': error_cd, 'estimate': estimate_cd, 'time': cd_time}
        }
        
        # Print comparison
        print(f"\n3. Comparison for {f_name}:")
        print(f"   Method    | Error        | Estimate     | Time")
        print(f"   ----------|--------------|--------------|-------")
        print(f"   Naive     | {error_naive:12.6f} | {estimate_naive:12.6f} | {naive_time:6.3f}s")
        print(f"   CD-HMC    | {error_cd:12.6f} | {estimate_cd:12.6f} | {cd_time:6.3f}s")
        
        if error_naive != float('inf') and error_cd != float('inf'):
            improvement = error_naive / error_cd if error_cd > 0 else float('inf')
            print(f"   CD-HMC improvement: {improvement:.2f}x better")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("Normalized Error: (fhat - E[f])^2/var[f]")
    print(f"{'='*60}")
    
    for f_name, result in results.items():
        print(f"\n{f_name.upper()}:")
        print(f"  Naive HMC:  {result['naive']['error']:.6f}")
        print(f"  CD-HMC:     {result['cd']['error']:.6f}")
        if result['naive']['error'] != float('inf') and result['cd']['error'] != float('inf'):
            improvement = result['naive']['error'] / result['cd']['error'] if result['cd']['error'] > 0 else float('inf')
            print(f"  Improvement: {improvement:.2f}x")

if __name__ == "__main__":
    run_simple_benchmark()
