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
    
    # Apply function to samples (don't flatten - let the function handle the shape)
    f_vals = f_func(samples)
    
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
            'variance': 2*sigma**2  # Var[x²] = 2 + 4λ²
        },
        'x_cubed': {
            'expectation': mu**3 + 3*mu*sigma**2,  # E[x³] = λ³ + 3λ
            'variance': 6*sigma**6 + 9*mu**2*sigma**4 + 9*mu**4*sigma**2  # Var[x³] = 6 + 9λ² + 9λ⁴
        }
    }
    
    return expectations

def calculate_gaussian_annealing_expectations(lam_final):
    """
    Calculate true expectations for 1D Gaussian annealing system.
    For V(q) = 0.5 * k(λ) * q², where k = 1 + 9*λ,
    the distribution is N(0, 1/sqrt(k)).
    
    Args:
        lam_final: Final lambda value
    
    Returns:
        Dictionary with true expectations for different functions
    """
    # k interpolates from 1 (var=1) to 10 (var=0.1)
    k = 1.0 + 9.0 * lam_final
    sigma = 1.0 / jnp.sqrt(k)  # Standard deviation
    
    expectations = {
        'x': {
            'expectation': 0.0,  # E[x] = 0 (mean is always 0)
            'variance': sigma**2  # Var[x] = 1/k
        },
        'x_squared': {
            'expectation': sigma**2,  # E[x²] = σ²
            'variance': 2*sigma**2  # Var[x²] = 2σ⁴
        },
        'x_cubed': {
            'expectation': 0.0,  # E[x³] = 0 (odd moment of symmetric distribution)
            'variance': 6*sigma**6  # Var[x³] = 6σ⁶
        }
    }
    
    return expectations

def calculate_double_well_expectations(lam_final):
    """
    Calculate true expectations for 1D double well system.
    For the double well at λ=1, we have exact values.
    
    Args:
        lam_final: Final lambda value
    
    Returns:
        Dictionary with true expectations for different functions
    """
    # For the double well at λ=1, we have the exact values you provided
    if lam_final == 1.0:
        expectations = {
            'x_squared': {
                'expectation': 7.3413954,  # E[x²] = 7.3413954
                'variance': 8.847134      # Var[x²] = 8.847134
            }
        }
    else:
        # For intermediate λ values, use linear interpolation
        expectations = {
            'x_squared': {
                'expectation': lam_final * 7.3413954,  # Linear interpolation
                'variance': lam_final * 8.847134       # Linear interpolation
            }
        }
    
    return expectations

def calculate_2d_normal_to_rosenbrock_expectations(lam_final):
    """
    Calculate true expectations for 2D normal to Rosenbrock system.
    For the Rosenbrock at λ=1, we have exact values for each component.
    
    Args:
        lam_final: Final lambda value
    
    Returns:
        Dictionary with true expectations for different functions
    """
    # For the Rosenbrock at λ=1, we have the exact values you provided
    if lam_final == 1.0:
        expectations = {
            'x0_squared': {
                'expectation': 2.0,      # E[x₀²] = 2.0
                'variance': 6.0          # Var[x₀²] = 6.0
            },
            'x1_squared': {
                'expectation': 1.0,      # E[x₁²] = 1.0
                'variance': 668.6        # Var[x₁²] = 668.6
            }
        }
    else:
        # For intermediate λ values, use linear interpolation
        expectations = {
            'x0_squared': {
                'expectation': lam_final * 2.0,      # Linear interpolation
                'variance': lam_final * 6.0          # Linear interpolation
            },
            'x1_squared': {
                'expectation': lam_final * 1.0,      # Linear interpolation
                'variance': lam_final * 668.6        # Linear interpolation
            }
        }
    
    return expectations

def run_simple_benchmark():
    """Run simple benchmark comparing naive HMC and CD-HMC for both systems using saved data."""
    
    # Test all systems
    systems = [
        ('gaussian_moving_mean', 'Gaussian Moving Mean'),
        ('gaussian_annealing', 'Gaussian Annealing'),
        ('double_well', 'Double Well'),
        ('2d_normal_to_rosenbrock', '2D Normal to Rosenbrock')
    ]
    
    results_summary = {}
    
    for system_name, system_display_name in systems:
        print("="*80)
        print(f"SIMPLE BENCHMARK: Naive HMC vs CD-HMC - {system_display_name}")
        print("="*80)
        print("Loading data from pickle files...")
        print("="*80)
        
        # Set up the system
        make_T, make_V, system_description, dim = get_system(system_name)
        print(f"System: {system_description}")
        print(f"Dimension: {dim}")
        
        # Define lambda functions
        v = 0.5
        max_lam = 1.0
        lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
        dot_lam_fn = jax.grad(lam_fn)
        
        # Load data from pickle files
        import pickle
        
        # Load naive HMC data
        naive_file = f"data/{system_name}_naive_unweighted.pkl"
        print(f"Loading naive HMC data from: {naive_file}")
        try:
            with open(naive_file, 'rb') as f:
                naive_data = pickle.load(f)
            naive_snapshots = naive_data['snapshots']
            naive_time = 0.0  # We don't have timing info from saved data
            print(f"   Loaded successfully. Final particles shape: {naive_snapshots['particles'][-1].shape}")
        except Exception as e:
            print(f"   Failed to load naive HMC data: {e}")
            continue
        
        # Load CD-HMC data
        cd_file = f"data/{system_name}_cd_unweighted.pkl"
        print(f"Loading CD-HMC data from: {cd_file}")
        try:
            with open(cd_file, 'rb') as f:
                cd_data = pickle.load(f)
            cd_snapshots = cd_data['snapshots']
            cd_time = 0.0  # We don't have timing info from saved data
            print(f"   Loaded successfully. Final particles shape: {cd_snapshots['particles'][-1].shape}")
        except Exception as e:
            print(f"   Failed to load CD-HMC data: {e}")
            continue
        
        # Get simulation parameters from saved data
        delta_t = naive_data.get('delta_t', 0.2)
        N_steps = len(naive_snapshots['particles']) - 1  # Number of steps
        
        print(f"\nSimulation parameters from saved data:")
        print(f"  - delta_t = {delta_t}")
        print(f"  - N_steps = {N_steps}")
        print(f"  - M = {naive_snapshots['particles'][-1].shape[0]}")
        
        # Calculate true expectations for final lambda value
        lam_final = lam_fn(N_steps * delta_t)
        
        if system_name == 'gaussian_moving_mean':
            true_expectations = calculate_gaussian_moving_mean_expectations(lam_final)
        elif system_name == 'gaussian_annealing':
            true_expectations = calculate_gaussian_annealing_expectations(lam_final)
        elif system_name == 'double_well':
            true_expectations = calculate_double_well_expectations(lam_final)
        elif system_name == '2d_normal_to_rosenbrock':
            true_expectations = calculate_2d_normal_to_rosenbrock_expectations(lam_final)
        else:
            raise ValueError(f"Unknown system: {system_name}")
        
        print(f"\nFinal lambda = {lam_final:.6f}")
        print(f"True expectations:")
        for f_name, exp_info in true_expectations.items():
            print(f"  {f_name}: E[f] = {exp_info['expectation']:.6f}, var[f] = {exp_info['variance']:.6f}")
        
        # Test functions (adjust based on system)
        if system_name == 'double_well':
            # For double well, we only have exact values for x_squared
            functions = [
                (lambda x: x**2, 'x_squared')
            ]
        elif system_name == '2d_normal_to_rosenbrock':
            # For 2D Rosenbrock, we have exact values for x0_squared and x1_squared
            functions = [
                (lambda x: x[:, 0]**2, 'x0_squared'),  # First component squared
                (lambda x: x[:, 1]**2, 'x1_squared')   # Second component squared
            ]
        else:
            # For gaussian systems, test all functions
            functions = [
                (lambda x: x**2, 'x_squared'),
                (lambda x: x, 'x'),
                (lambda x: x**3, 'x_cubed')
            ]
        
        results = {}
        
        for f_func, f_name in functions:
            print(f"\n{'='*50}")
            print(f"TESTING FUNCTION: {f_name}")
            print(f"{'='*50}")
            
            true_exp = true_expectations[f_name]['expectation']
            true_var = true_expectations[f_name]['variance']
            
            print(f"True expectation E[f] = {true_exp:.6f}")
            print(f"True variance var[f] = {true_var:.6f}")
            
            # 1. Naive HMC
            print(f"\n1. Analyzing Naive HMC data...")
            
            try:
                # Get final samples
                final_samples_naive = naive_snapshots['particles'][-1]
                
                # Calculate error
                error_naive, estimate_naive = calculate_normalized_error(
                    final_samples_naive, true_exp, true_var, f_func
                )
                
                print(f"   Estimate: {estimate_naive:.6f}")
                print(f"   Normalized Error: {error_naive:.6f}")
                
            except Exception as e:
                print(f"   Naive HMC analysis failed: {e}")
                error_naive = float('inf')
                estimate_naive = float('nan')
            
            # 2. Counterdiabatic HMC
            print(f"\n2. Analyzing CD-HMC data...")
            
            try:
                # Get final samples
                final_samples_cd = cd_snapshots['particles'][-1]
                
                # Calculate error
                error_cd, estimate_cd = calculate_normalized_error(
                    final_samples_cd, true_exp, true_var, f_func
                )
                
                print(f"   Estimate: {estimate_cd:.6f}")
                print(f"   Normalized Error: {error_cd:.6f}")
                
            except Exception as e:
                print(f"   CD-HMC analysis failed: {e}")
                error_cd = float('inf')
                estimate_cd = float('nan')
            
            # Store results
            results[f_name] = {
                'naive': {'error': error_naive, 'estimate': estimate_naive},
                'cd': {'error': error_cd, 'estimate': estimate_cd}
            }
            
            # Print comparison
            print(f"\n3. Comparison for {f_name}:")
            print(f"   Method    | Error        | Estimate")
            print(f"   ----------|--------------|--------------")
            print(f"   Naive     | {error_naive:12.6f} | {estimate_naive:12.6f}")
            print(f"   CD-HMC    | {error_cd:12.6f} | {estimate_cd:12.6f}")
            
            if error_naive != float('inf') and error_cd != float('inf'):
                improvement = error_naive / error_cd if error_cd > 0 else float('inf')
                print(f"   CD-HMC improvement: {improvement:.2f}x better")
        
        # Store results in summary
        results_summary[system_name] = {}
        for f_name, result in results.items():
            results_summary[system_name][f_name] = {
                'true_expectation': true_expectations[f_name]['expectation'],
                'naive': result['naive'],
                'cd': result['cd']
            }
        
        # Summary for this system
        print(f"\n{'='*60}")
        print(f"SUMMARY - {system_display_name}")
        print(f"{'='*60}")
        print("Monte Carlo Estimates and Normalized Errors")
        print(f"{'='*60}")
        
        for f_name, result in results.items():
            print(f"\n{f_name.upper()}:")
            print(f"  True E[f]:  {true_expectations[f_name]['expectation']:.6f}")
            print(f"  Naive HMC:  Estimate = {result['naive']['estimate']:.6f}, Error = {result['naive']['error']:.6f}")
            print(f"  CD-HMC:     Estimate = {result['cd']['estimate']:.6f}, Error = {result['cd']['error']:.6f}")
            if result['naive']['error'] != float('inf') and result['cd']['error'] != float('inf'):
                improvement = result['naive']['error'] / result['cd']['error'] if result['cd']['error'] > 0 else float('inf')
                print(f"  CD-HMC improvement: {improvement:.2f}x better")
        
        print(f"\n{'='*80}")
        print(f"END OF {system_display_name.upper()} BENCHMARK")
        print(f"{'='*80}\n")
    
    # Save LaTeX table with second moment results
    save_latex_table(results_summary)

def save_latex_table(results_summary):
    """Save a LaTeX table with second moment results for both systems."""
    
    # Create the LaTeX table
    latex_table = r"""\begin{table}[h]
\centering
\caption{Monte Carlo estimates of the second moment $E[x^2]$ for different systems and methods.}
\label{tab:second_moment_results}
\begin{tabular}{lccccc}
\toprule
System & True $E[x^2]$ & Naive HMC & CD-HMC & Error Ratio & Improvement \\
\midrule"""
    
    for system_name, system_results in results_summary.items():
        # Handle both 1D and 2D systems
        if 'x_squared' in system_results:
            # 1D system
            x_squared_data = system_results['x_squared']
            true_exp = x_squared_data['true_expectation']
            naive_est = x_squared_data['naive']['estimate']
            cd_est = x_squared_data['cd']['estimate']
            naive_error = x_squared_data['naive']['error']
            cd_error = x_squared_data['cd']['error']
            
            # Calculate error ratio and improvement
            if cd_error > 0:
                error_ratio = naive_error / cd_error
                improvement = f"{error_ratio:.1f}x"
            else:
                error_ratio = float('inf')
                improvement = "$\infty$"
            
            # Format the row
            system_display = system_name.replace('_', ' ').title()
            latex_table += f"\n{system_display} & {true_exp:.6f} & {naive_est:.6f} & {cd_est:.6f} & {error_ratio:.1f} & {improvement} \\\\"
        
        elif 'x0_squared' in system_results and 'x1_squared' in system_results:
            # 2D system - add both components
            for comp in ['x0_squared', 'x1_squared']:
                x_squared_data = system_results[comp]
                true_exp = x_squared_data['true_expectation']
                naive_est = x_squared_data['naive']['estimate']
                cd_est = x_squared_data['cd']['estimate']
                naive_error = x_squared_data['naive']['error']
                cd_error = x_squared_data['cd']['error']
                
                # Calculate error ratio and improvement
                if cd_error > 0:
                    error_ratio = naive_error / cd_error
                    improvement = f"{error_ratio:.1f}x"
                else:
                    error_ratio = float('inf')
                    improvement = "$\infty$"
                
                # Format the row with component indicator
                system_display = system_name.replace('_', ' ').title()
                comp_name = comp.replace('_', ' ').title()
                latex_table += f"\n{system_display} ({comp_name}) & {true_exp:.6f} & {naive_est:.6f} & {cd_est:.6f} & {error_ratio:.1f} & {improvement} \\\\"
    
    latex_table += r"""
\bottomrule
\end{tabular}
\end{table}"""
    
    # Save to file
    with open("benchmark_results/second_moment_latex_table.tex", "w") as f:
        f.write(latex_table)
    
    print("LaTeX table saved to: benchmark_results/second_moment_latex_table.tex")
    
    # Also save a simple text version
    text_table = "Second Moment Results (E[x²])\n"
    text_table += "=" * 60 + "\n"
    text_table += "System | True E[x²] | Naive HMC | CD-HMC | Error Ratio\n"
    text_table += "-" * 60 + "\n"
    
    for system_name, system_results in results_summary.items():
        # Handle both 1D and 2D systems
        if 'x_squared' in system_results:
            # 1D system
            x_squared_data = system_results['x_squared']
            true_exp = x_squared_data['true_expectation']
            naive_est = x_squared_data['naive']['estimate']
            cd_est = x_squared_data['cd']['estimate']
            naive_error = x_squared_data['naive']['error']
            cd_error = x_squared_data['cd']['error']
            
            if cd_error > 0:
                error_ratio = naive_error / cd_error
            else:
                error_ratio = float('inf')
            
            system_display = system_name.replace('_', ' ').title()
            text_table += f"{system_display} | {true_exp:.6f} | {naive_est:.6f} | {cd_est:.6f} | {error_ratio:.1f}x\n"
        
        elif 'x0_squared' in system_results and 'x1_squared' in system_results:
            # 2D system - add both components
            for comp in ['x0_squared', 'x1_squared']:
                x_squared_data = system_results[comp]
                true_exp = x_squared_data['true_expectation']
                naive_est = x_squared_data['naive']['estimate']
                cd_est = x_squared_data['cd']['estimate']
                naive_error = x_squared_data['naive']['error']
                cd_error = x_squared_data['cd']['error']
                
                if cd_error > 0:
                    error_ratio = naive_error / cd_error
                else:
                    error_ratio = float('inf')
                
                system_display = system_name.replace('_', ' ').title()
                comp_name = comp.replace('_', ' ').title()
                text_table += f"{system_display} ({comp_name}) | {true_exp:.6f} | {naive_est:.6f} | {cd_est:.6f} | {error_ratio:.1f}x\n"
    
    with open("benchmark_results/second_moment_results.txt", "w") as f:
        f.write(text_table)
    
    print("Text table saved to: benchmark_results/second_moment_results.txt")

if __name__ == "__main__":
    run_simple_benchmark()
