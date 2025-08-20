import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the current directory to the path
sys.path.append('.')
from src.systems import get_system

def plot_final_distributions(system_name="double_well", ansatz_type="neural_network"):
    """Plot final distributions for all 4 simulation cases."""
    
    # Get system functions
    make_T, make_V, system_description, dim = get_system(system_name)
    
    # Load all simulation data
    simulation_types = ['naive_unweighted', 'naive_weighted', 'cd_unweighted', 'cd_weighted']
    data = {}
    
    for sim_type in simulation_types:
        filename = f"data/{sim_type}_snapshots_{system_name}.pkl"
        try:
            with open(filename, 'rb') as f:
                data[sim_type] = pickle.load(f)
            print(f"✓ Loaded {sim_type}")
        except FileNotFoundError:
            print(f"✗ {sim_type}: File not found")
            data[sim_type] = None
        except Exception as e:
            print(f"✗ {sim_type}: Error loading {e}")
            data[sim_type] = None
    
    # Get final lambda value
    final_lam = 1.0
    
    # Create true distribution
    x_grid = np.linspace(-5, 5, 1000)
    potential_fn = make_V(final_lam)
    rho = np.array([np.exp(-potential_fn(x)) for x in x_grid])
    rho = rho / np.trapz(rho, x_grid)  # Normalize
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    titles = {
        'naive_unweighted': 'Naive HMC (Unweighted)',
        'naive_weighted': 'Naive HMC (Weighted SMC)',
        'cd_unweighted': 'Counterdiabatic HMC (Unweighted)',
        'cd_weighted': 'Counterdiabatic HMC (Weighted)'
    }
    
    colors = {
        'naive_unweighted': 'blue',
        'naive_weighted': 'green',
        'cd_unweighted': 'red',
        'cd_weighted': 'orange'
    }
    
    # Plot each simulation type
    for i, sim_type in enumerate(simulation_types):
        ax = axes[i]
        
        if data[sim_type] is None:
            ax.text(0.5, 0.5, f'No data for {sim_type}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(titles[sim_type])
            continue
        
        # Get final samples based on simulation type
        if sim_type == 'naive_unweighted':
            final_samples = data[sim_type]['naive'][-1]
        elif sim_type == 'naive_weighted':
            final_samples = data[sim_type]['naive_weighted'][-1]
        elif sim_type in ['cd_unweighted', 'cd_weighted']:
            final_samples = data[sim_type]['cd'][-1]
        else:
            continue
        
        # Plot histogram
        ax.hist(final_samples.flatten(), bins=50, alpha=0.6, density=True, 
               color=colors[sim_type], label=titles[sim_type])
        
        # Plot true distribution
        ax.plot(x_grid, rho, 'k--', linewidth=2, label='True distribution')
        
        # Calculate and display statistics
        mean_val = np.mean(final_samples)
        std_val = np.std(final_samples)
        ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(titles[sim_type], fontsize=14, fontweight='bold')
        ax.set_xlabel('Position q')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create figures directory
    os.makedirs("figures", exist_ok=True)
    ansatz_dir = f"figures/{ansatz_type}"
    os.makedirs(ansatz_dir, exist_ok=True)
    
    # Save plot
    filename = f"{ansatz_dir}/final_distributions_{system_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved final distributions plot to {filename}")
    
    # Print summary statistics
    print("\nFinal Distribution Statistics:")
    print("-" * 80)
    print(f"{'Method':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 80)
    
    true_mean = np.average(x_grid, weights=rho)
    true_std = np.sqrt(np.average((x_grid - true_mean)**2, weights=rho))
    print(f"{'True distribution':<25} {true_mean:<10.3f} {true_std:<10.3f} {'N/A':<10} {'N/A':<10}")
    
    for sim_type in simulation_types:
        if data[sim_type] is None:
            continue
            
        # Get final samples
        if sim_type == 'naive_unweighted':
            final_samples = data[sim_type]['naive'][-1]
        elif sim_type == 'naive_weighted':
            final_samples = data[sim_type]['naive_weighted'][-1]
        elif sim_type in ['cd_unweighted', 'cd_weighted']:
            final_samples = data[sim_type]['cd'][-1]
        else:
            continue
        
        mean_val = np.mean(final_samples)
        std_val = np.std(final_samples)
        min_val = np.min(final_samples)
        max_val = np.max(final_samples)
        print(f"{titles[sim_type]:<25} {mean_val:<10.3f} {std_val:<10.3f} {min_val:<10.3f} {max_val:<10.3f}")

if __name__ == "__main__":
    # Plot final distributions for double well with neural network ansatz
    plot_final_distributions("double_well", "neural_network")
