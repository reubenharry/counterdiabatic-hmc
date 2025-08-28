import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde
from src.simulation import run_simulation_and_save_data
from src.ansatze import PolynomialAnsatz
from src.systems import get_system

def calculate_expectation_error(true_expectation, estimated_expectation, weights=None):
    """Calculate error between true and estimated expectations."""
    if weights is not None:
        # For weighted samples, use weighted mean
        weights = np.exp(weights - np.max(weights))  # Numerical stability
        weights = weights / np.sum(weights)
        weighted_estimate = np.average(estimated_expectation, weights=weights)
    else:
        # For unweighted samples, use simple mean
        weighted_estimate = np.mean(estimated_expectation)
    
    absolute_error = abs(weighted_estimate - true_expectation)
    relative_error = absolute_error / abs(true_expectation) if true_expectation != 0 else absolute_error
    
    return {
        'absolute_error': absolute_error,
        'relative_error': relative_error,
        'weighted_estimate': weighted_estimate,
        'true_value': true_expectation
    }

def calculate_gaussian_expectations(lam_final, dim=1):
    """Calculate true expectations for Gaussian with mean λ."""
    # For V(q) = 0.5 * (q - λ)², the distribution is N(λ, 1)
    mu = lam_final
    sigma = 1.0
    
    expectations = {
        'mean': mu,  # E[q] = λ
        'variance': sigma**2,  # Var[q] = 1
        'second_moment': mu**2 + sigma**2,  # E[q²] = λ² + 1
        # 'third_moment': mu**3 + 3*mu*sigma**2,  # E[q³] = λ³ + 3λ
        # 'fourth_moment': mu**4 + 6*mu**2*sigma**2 + 3*sigma**4,  # E[q⁴] = λ⁴ + 6λ² + 3
        # 'potential_energy': 0.5,  # E[V(q)] = 0.5 * E[(q-λ)²] = 0.5 * Var[q] = 0.5
        # 'kinetic_energy': 0.5,  # E[T(p)] = 0.5 * E[p²] = 0.5 (assuming p ~ N(0,1))
        # 'total_energy': 1.0,  # E[H] = E[T] + E[V] = 0.5 + 0.5 = 1.0
    }
    
    return expectations

def calculate_double_well_expectations(lam_final, dim=1):
    """Calculate true expectations for double well potential at λ=1."""
    # For the double well at λ=1, we have the exact values you provided
    if lam_final == 1.0:
        expectations = {
            'second_moment': 7.3413954,  # E[x²] = 7.3413954
            'fourth_moment': 8.847134,   # Var[x²] = 8.847134 (this is actually Var[x²], not E[x⁴])
        }
    else:
        # For intermediate λ values, we could interpolate or use approximation
        # For now, let's use a simple interpolation
        expectations = {
            'second_moment': lam_final * 7.3413954,  # Linear interpolation
            'fourth_moment': lam_final * 8.847134,   # Linear interpolation
        }
    
    return expectations

def run_benchmark_comparison(system_name="gaussian_moving_mean", M=1000, N_steps=40, 
                           delta_t=0.05, eps=0.05, num_trials=10, ess_threshold=0.5):
    """Run comprehensive benchmark comparing CD-HMC and SMC methods."""
    
    print(f"Running benchmark comparison for {system_name}")
    print(f"Parameters: M={M}, N_steps={N_steps}, num_trials={num_trials}")
    
    # Set up the system
    make_T, make_V, system_description, dim = get_system(system_name)
    
    # Define lambda functions
    v = 0.5
    max_lam = 1.0
    lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
    dot_lam_fn = jax.grad(lam_fn)
    
    # Parameters
    momentum_refresh_interval = 5.0
    fit_every = 1
    num_initial_iterations = 100000
    num_iterations = 100000
    learning_rate = 1e-4
    re_equil_steps = 0
    
    # Create ansatz
    ansatz = PolynomialAnsatz(max_degree=2, dim=dim)
    
    # Storage for results
    results = {
        'cd_hmc': {'errors': [], 'times': [], 'final_samples': []},
        'smc_weighted': {'errors': [], 'times': [], 'final_samples': [], 'weights': []},
        'naive_unweighted': {'errors': [], 'times': [], 'final_samples': []}
    }
    
    # Expectations to test (will be set dynamically based on system)
    expectation_names = []
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # Use different random key for each trial
        key = jax.random.PRNGKey(trial)
        
        # 1. Counterdiabatic HMC
        print("  Running CD-HMC...")
        key, subkey = jax.random.split(key)
        start_time = time.time()
        
        try:
            _, snapshots_cd, _, _ = run_simulation(
                M=M, N_steps=N_steps, delta_t=delta_t, eps=eps,
                momentum_refresh_interval=momentum_refresh_interval,
                fit_every=fit_every, num_initial_iterations=num_initial_iterations,
                num_iterations=num_iterations, make_T=make_T, make_V=make_V,
                A_ansatz=ansatz, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
                key=subkey, dim=dim, learning_rate=learning_rate,
                re_equil_steps=re_equil_steps, use_weights=False
            )
            cd_time = time.time() - start_time
            results['cd_hmc']['times'].append(cd_time)
            
            # Get final samples (post-equilibration)
            if len(snapshots_cd['cd_post_equil']) > 0:
                final_samples_cd = snapshots_cd['cd_post_equil'][-1]
                results['cd_hmc']['final_samples'].append(final_samples_cd)
            else:
                final_samples_cd = snapshots_cd['cd_pre_equil'][-1]
                results['cd_hmc']['final_samples'].append(final_samples_cd)
                
        except Exception as e:
            print(f"    CD-HMC failed: {e}")
            continue
        
        # 2. SMC (Weighted)
        print("  Running SMC...")
        key, subkey = jax.random.split(key)
        start_time = time.time()
        
        try:
            snapshots_smc = run_naive_hmc_simulation(
                M=M, N_steps=N_steps, delta_t=delta_t, eps=eps,
                momentum_refresh_interval=momentum_refresh_interval,
                make_T=make_T, make_V=make_V, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
                key=subkey, dim=dim, use_weights=True, ess_threshold=ess_threshold
            )
            smc_time = time.time() - start_time
            results['smc_weighted']['times'].append(smc_time)
            
            final_samples_smc = snapshots_smc['naive_weighted'][-1]
            final_weights_smc = snapshots_smc['weights'][-1]
            results['smc_weighted']['final_samples'].append(final_samples_smc)
            results['smc_weighted']['weights'].append(final_weights_smc)
            
        except Exception as e:
            print(f"    SMC failed: {e}")
            continue
        
        # 3. Naive HMC (Unweighted)
        print("  Running Naive HMC...")
        key, subkey = jax.random.split(key)
        start_time = time.time()
        
        try:
            snapshots_naive = run_naive_hmc_simulation(
                M=M, N_steps=N_steps, delta_t=delta_t, eps=eps,
                momentum_refresh_interval=momentum_refresh_interval,
                make_T=make_T, make_V=make_V, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
                key=subkey, dim=dim, use_weights=False
            )
            naive_time = time.time() - start_time
            results['naive_unweighted']['times'].append(naive_time)
            
            final_samples_naive = snapshots_naive['naive'][-1]
            results['naive_unweighted']['final_samples'].append(final_samples_naive)
            
        except Exception as e:
            print(f"    Naive HMC failed: {e}")
            continue
        
        # Calculate true expectations for final lambda value
        lam_final = lam_fn(N_steps * delta_t)
        
        # Choose appropriate expectation calculation based on system
        if system_name == "double_well":
            true_expectations = calculate_double_well_expectations(lam_final, dim)
            # For double well, we focus on second_moment and fourth_moment
            expectation_names = ['second_moment', 'fourth_moment']
        else:
            true_expectations = calculate_gaussian_expectations(lam_final, dim)
            # For gaussian systems, we use the standard expectations
            expectation_names = ['mean', 'variance', 'second_moment']
        
        # Calculate errors for each method
        methods = [
            ('cd_hmc', final_samples_cd, None),
            ('smc_weighted', final_samples_smc, final_weights_smc),
            ('naive_unweighted', final_samples_naive, None)
        ]
        
        for method_name, samples, weights in methods:
            method_errors = {}
            
            for exp_name in expectation_names:
                if exp_name == 'mean':
                    estimated_exp = samples.flatten()
                elif exp_name == 'variance':
                    estimated_exp = (samples.flatten() - true_expectations['mean'])**2
                elif exp_name == 'second_moment':
                    estimated_exp = samples.flatten()**2
                elif exp_name == 'third_moment':
                    estimated_exp = samples.flatten()**3
                elif exp_name == 'fourth_moment':
                    # For double well, this is actually Var[x²], not E[x⁴]
                    # We calculate Var[x²] = E[x⁴] - (E[x²])²
                    x_squared = samples.flatten()**2
                    if weights is not None:
                        weights_norm = np.exp(weights - np.max(weights))
                        weights_norm = weights_norm / np.sum(weights_norm)
                        E_x2 = np.average(x_squared, weights=weights_norm)
                        E_x4 = np.average(samples.flatten()**4, weights=weights_norm)
                    else:
                        E_x2 = np.mean(x_squared)
                        E_x4 = np.mean(samples.flatten()**4)
                    estimated_exp = E_x4 - E_x2**2  # Var[x²]
                elif exp_name == 'potential_energy':
                    V_fn = make_V(lam_final)
                    estimated_exp = np.array([V_fn(q) for q in samples.flatten()])
                elif exp_name == 'kinetic_energy':
                    # Generate momentum samples for kinetic energy
                    key, subkey = jax.random.split(key)
                    p_samples = jax.random.normal(subkey, samples.shape)
                    T_fn = make_T(lam_final)
                    estimated_exp = np.array([T_fn(p) for p in p_samples.flatten()])
                elif exp_name == 'total_energy':
                    # Sum of potential and kinetic energy
                    V_fn = make_V(lam_final)
                    T_fn = make_T(lam_final)
                    key, subkey = jax.random.split(key)
                    p_samples = jax.random.normal(subkey, samples.shape)
                    V_vals = np.array([V_fn(q) for q in samples.flatten()])
                    T_vals = np.array([T_fn(p) for p in p_samples.flatten()])
                    estimated_exp = V_vals + T_vals
                
                error_info = calculate_expectation_error(
                    true_expectations[exp_name], estimated_exp, weights
                )
                method_errors[exp_name] = error_info
            
            results[method_name]['errors'].append(method_errors)
    
    # Analyze results
    print(f"\nBenchmark completed. Successful trials: {len(results['cd_hmc']['errors'])}")
    
    # Create results summary
    summary = create_benchmark_summary(results, expectation_names)
    
    # Save results
    save_benchmark_results(results, summary, system_name, M, N_steps, num_trials)
    
    # Create plots
    create_benchmark_plots(results, summary, system_name, expectation_names)
    
    return results, summary

def create_benchmark_summary(results, expectation_names):
    """Create summary statistics from benchmark results."""
    summary = {}
    
    for method_name in results.keys():
        if len(results[method_name]['errors']) == 0:
            continue
            
        summary[method_name] = {
            'mean_times': np.mean(results[method_name]['times']),
            'std_times': np.std(results[method_name]['times']),
            'expectations': {}
        }
        
        for exp_name in expectation_names:
            absolute_errors = [trial[exp_name]['absolute_error'] 
                             for trial in results[method_name]['errors']]
            relative_errors = [trial[exp_name]['relative_error'] 
                             for trial in results[method_name]['errors']]
            
            summary[method_name]['expectations'][exp_name] = {
                'mean_absolute_error': np.mean(absolute_errors),
                'std_absolute_error': np.std(absolute_errors),
                'mean_relative_error': np.mean(relative_errors),
                'std_relative_error': np.std(relative_errors),
                'median_absolute_error': np.median(absolute_errors),
                'median_relative_error': np.median(relative_errors)
            }
    
    return summary

def save_benchmark_results(results, summary, system_name, M, N_steps, num_trials):
    """Save benchmark results to files."""
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Save summary
    summary_file = f"benchmark_results/{system_name}_M{M}_N{N_steps}_trials{num_trials}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Benchmark Results for {system_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Parameters: M={M}, N_steps={N_steps}, num_trials={num_trials}\n\n")
        
        for method_name, method_summary in summary.items():
            f.write(f"{method_name.upper()}:\n")
            f.write(f"  Mean time: {method_summary['mean_times']:.3f} ± {method_summary['std_times']:.3f} s\n")
            f.write("  Expectation Errors:\n")
            
            for exp_name, exp_stats in method_summary['expectations'].items():
                f.write(f"    {exp_name}:\n")
                f.write(f"      Absolute: {exp_stats['mean_absolute_error']:.6f} ± {exp_stats['std_absolute_error']:.6f}\n")
                f.write(f"      Relative: {exp_stats['mean_relative_error']:.6f} ± {exp_stats['std_relative_error']:.6f}\n")
            f.write("\n")
    
    print(f"Results saved to {summary_file}")

def create_benchmark_plots(results, summary, system_name, expectation_names):
    """Create visualization plots for benchmark results."""
    os.makedirs("benchmark_results", exist_ok=True)
    
    # 1. Error comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    methods = ['cd_hmc', 'smc_weighted', 'naive_unweighted']
    colors = ['red', 'green', 'blue']
    labels = ['CD-HMC', 'SMC (Weighted)', 'Naive HMC']
    
    for i, exp_name in enumerate(['mean', 'variance', 'second_moment', 'potential_energy']):
        ax = axes[i]
        
        for j, method_name in enumerate(methods):
            if method_name not in summary:
                continue
                
            exp_stats = summary[method_name]['expectations'][exp_name]
            mean_error = exp_stats['mean_absolute_error']
            std_error = exp_stats['std_absolute_error']
            
            ax.bar(j, mean_error, yerr=std_error, color=colors[j], 
                   label=labels[j], alpha=0.7, capsize=5)
        
        ax.set_title(f'{exp_name.replace("_", " ").title()} Error')
        ax.set_ylabel('Absolute Error')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"benchmark_results/{system_name}_error_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Runtime comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    method_names = []
    mean_times = []
    std_times = []
    
    for method_name in methods:
        if method_name in summary:
            method_names.append(labels[methods.index(method_name)])
            mean_times.append(summary[method_name]['mean_times'])
            std_times.append(summary[method_name]['std_times'])
    
    bars = ax.bar(method_names, mean_times, yerr=std_times, capsize=5, alpha=0.7)
    ax.set_title('Runtime Comparison')
    ax.set_ylabel('Time (seconds)')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_time in zip(bars, mean_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean_time:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"benchmark_results/{system_name}_runtime_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to benchmark_results/{system_name}_*.png")

def run_quick_benchmark():
    """Run a quick benchmark for testing."""
    print("Running quick benchmark...")
    results, summary = run_benchmark_comparison(
        system_name="gaussian_moving_mean",
        M=500,  # Smaller number of particles
        N_steps=20,  # Fewer steps
        num_trials=3,  # Fewer trials
        ess_threshold=0.5
    )
    return results, summary

if __name__ == "__main__":
    # Run quick benchmark for testing
    results, summary = run_quick_benchmark()
    
    # Print key results
    print("\nKey Results:")
    for method_name, method_summary in summary.items():
        print(f"\n{method_name.upper()}:")
        print(f"  Mean time: {method_summary['mean_times']:.3f} ± {method_summary['std_times']:.3f} s")
        print(f"  Mean error: {method_summary['expectations']['mean']['mean_absolute_error']:.6f}")
        print(f"  Variance error: {method_summary['expectations']['variance']['mean_absolute_error']:.6f}")
