import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def load_simulation_data(system_name="gaussian_moving_mean"):
    """Load precomputed simulation data from data folder."""
    data = {}
    
    # Load CD weighted data
    try:
        with open(f'data/cd_weighted_snapshots_{system_name}.pkl', 'rb') as f:
            cd_weighted_data = pickle.load(f)
        data['cd_weighted'] = {
            'samples': cd_weighted_data['cd_weighted'][-1],  # Final samples
            'weights': cd_weighted_data['weights_cd'][-1],   # Final weights
            'lam_final': cd_weighted_data['lam'][-1]         # Final lambda
        }
        print(f"Loaded CD weighted data: {data['cd_weighted']['samples'].shape}")
    except Exception as e:
        print(f"Error loading CD weighted data: {e}")
        data['cd_weighted'] = None
    
    # Load CD unweighted data
    try:
        with open(f'data/cd_unweighted_snapshots_{system_name}.pkl', 'rb') as f:
            cd_unweighted_data = pickle.load(f)
        data['cd_unweighted'] = {
            'samples': cd_unweighted_data['cd_unweighted'][-1],  # Final samples
            'weights': None,                          # No weights for unweighted
            'lam_final': cd_unweighted_data['lam'][-1]  # Final lambda
        }
        print(f"Loaded CD unweighted data: {data['cd_unweighted']['samples'].shape}")
    except Exception as e:
        print(f"Error loading CD unweighted data: {e}")
        data['cd_unweighted'] = None
    
    # Load naive weighted data
    try:
        with open(f'data/naive_weighted_snapshots_{system_name}.pkl', 'rb') as f:
            naive_weighted_data = pickle.load(f)
        data['naive_weighted'] = {
            'samples': naive_weighted_data['naive_weighted'][-1],  # Final samples
            'weights': naive_weighted_data['weights_naive'][-1],   # Final weights
            'lam_final': naive_weighted_data['lam'][-1]            # Final lambda
        }
        print(f"Loaded naive weighted data: {data['naive_weighted']['samples'].shape}")
    except Exception as e:
        print(f"Error loading naive weighted data: {e}")
        data['naive_weighted'] = None
    
    # Load naive unweighted data
    try:
        with open(f'data/naive_unweighted_snapshots_{system_name}.pkl', 'rb') as f:
            naive_unweighted_data = pickle.load(f)
        data['naive_unweighted'] = {
            'samples': naive_unweighted_data['naive'][-1],  # Final samples
            'weights': None,                                # No weights for unweighted
            'lam_final': naive_unweighted_data['lam'][-1]   # Final lambda
        }
        print(f"Loaded naive unweighted data: {data['naive_unweighted']['samples'].shape}")
    except Exception as e:
        print(f"Error loading naive unweighted data: {e}")
        data['naive_unweighted'] = None
    
    return data

def calculate_gaussian_expectations(lam_final, dim=1):
    """Calculate true expectations for Gaussian with mean λ."""
    # For V(q) = 0.5 * (q - λ)², the distribution is N(λ, 1)
    mu = lam_final
    sigma = 1.0
    
    expectations = {
        'mean': mu,  # E[q] = λ
        'variance': sigma**2,  # Var[q] = 1
        'second_moment': mu**2 + sigma**2,  # E[q²] = λ² + 1
    }
    
    return expectations

def calculate_x2_error(samples, weights=None, lam_final=1.0):
    """
    Calculate error for x² estimate.
    Error = ((x2hat - E[x²])²) / var[x²] where var[x²] = 2*E[x²] for Gaussian
    """
    # Calculate x2hat (estimate of E[x²])
    if weights is not None:
        # For weighted samples, use weighted mean
        weights = np.exp(weights - np.max(weights))  # Numerical stability
        weights = weights / np.sum(weights)
        x2hat = np.average(samples.flatten()**2, weights=weights)
    else:
        # For unweighted samples, use simple mean
        x2hat = np.mean(samples.flatten()**2)
    
    # Calculate true E[x²] and var[x²]
    true_expectations = calculate_gaussian_expectations(lam_final)
    E_x2 = true_expectations['second_moment']
    var_x2 = 2 * E_x2  # For Gaussian: var[x²] = 2*E[x²]
    
    # Calculate error
    error = ((x2hat - E_x2)**2) / var_x2
    
    return {
        'error': error,
        'x2hat': x2hat,
        'E_x2': E_x2,
        'var_x2': var_x2
    }

def create_error_comparison_plot(data, system_name="gaussian_moving_mean"):
    """Create barplot comparing errors for the 4 different simulation methods."""
    
    # Define methods and their display names
    methods = ['cd_weighted', 'cd_unweighted', 'naive_weighted', 'naive_unweighted']
    method_labels = ['CD Weighted', 'CD Unweighted', 'Naive Weighted', 'Naive Unweighted']
    colors = ['red', 'orange', 'blue', 'lightblue']
    
    # Calculate errors for each method
    errors = []
    x2hats = []
    E_x2s = []
    var_x2s = []
    
    for method in methods:
        if data[method] is not None:
            result = calculate_x2_error(
                data[method]['samples'], 
                data[method]['weights'], 
                data[method]['lam_final']
            )
            errors.append(result['error'])
            x2hats.append(result['x2hat'])
            E_x2s.append(result['E_x2'])
            var_x2s.append(result['var_x2'])
        else:
            errors.append(np.nan)
            x2hats.append(np.nan)
            E_x2s.append(np.nan)
            var_x2s.append(np.nan)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Error comparison
    bars1 = ax1.bar(method_labels, errors, color=colors, alpha=0.7, capsize=5)
    ax1.set_title(f'{system_name.replace("_", " ").title()}: x² Error Comparison', fontsize=14)
    ax1.set_ylabel('Error = ((x2hat - E[x²])²) / var[x²]', fontsize=12)
    ax1.set_xlabel('Method', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bar, error in zip(bars1, errors):
        if not np.isnan(error):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                    f'{error:.4f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Plot 2: x2hat vs E[x²] comparison
    x_pos = np.arange(len(method_labels))
    width = 0.35
    
    bars2a = ax2.bar(x_pos - width/2, x2hats, width, label='x2hat (estimated)', 
                     color=colors, alpha=0.7)
    bars2b = ax2.bar(x_pos + width/2, E_x2s, width, label='E[x²] (true)', 
                     color='gray', alpha=0.7)
    
    ax2.set_title(f'{system_name.replace("_", " ").title()}: x² Estimates vs True Value', fontsize=14)
    ax2.set_ylabel('x² Value', fontsize=12)
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(method_labels, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2a, x2hats):
        if not np.isnan(value):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar, value in zip(bars2b, E_x2s):
        if not np.isnan(value):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("benchmark_results", exist_ok=True)
    plt.savefig(f"benchmark_results/{system_name}_x2_error_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\n{system_name.replace('_', ' ').title()} Error Summary:")
    print("=" * 60)
    for i, method in enumerate(methods):
        if not np.isnan(errors[i]):
            print(f"{method_labels[i]}:")
            print(f"  Error: {errors[i]:.6f}")
            print(f"  x2hat: {x2hats[i]:.6f}")
            print(f"  E[x²]: {E_x2s[i]:.6f}")
            print(f"  var[x²]: {var_x2s[i]:.6f}")
            print()
    
    return {
        'errors': errors,
        'x2hats': x2hats,
        'E_x2s': E_x2s,
        'var_x2s': var_x2s,
        'methods': methods,
        'method_labels': method_labels
    }

def run_data_benchmark(system_name="gaussian_moving_mean"):
    """Run benchmark using precomputed data from data folder."""
    print(f"Loading precomputed data for {system_name}...")
    
    # Load data
    data = load_simulation_data(system_name)
    
    # Check if we have all required data
    available_methods = [method for method, data_dict in data.items() if data_dict is not None]
    print(f"Available methods: {available_methods}")
    
    if len(available_methods) < 4:
        print(f"Warning: Only {len(available_methods)} methods available, expected 4")
    
    # Create comparison plot
    results = create_error_comparison_plot(data, system_name)
    
    return data, results

if __name__ == "__main__":
    # Run benchmark for gaussian_moving_mean
    data, results = run_data_benchmark("gaussian_moving_mean")
    
    print("\nBenchmark completed!")
    print(f"Results saved to: benchmark_results/gaussian_moving_mean_x2_error_comparison.png")
