import jax
import jax.numpy as jnp
import pickle
import os

from src.simulation import run_simulation, run_naive_hmc_simulation
from src.plotting import plot_results, create_ridge_plot
from src.ansatze import PolynomialAnsatz, NeuralNetworkAnsatz, AnalyticAnsatz
from src.systems import get_system
import matplotlib.pyplot as plt
import numpy as np

def save_simulation_data(snapshots, system_name, method_name, delta_t, lam_fn, ansatz_params=None, loss_histories=None, param_history=None):
    """Save simulation data to a pickle file."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Extract lambda values at each snapshot time
    times = jnp.arange(len(snapshots.get('naive', snapshots.get('cd_pre_equil', [])))) * delta_t
    lambda_values = [float(lam_fn(t)) for t in times]
    
    # Prepare data to save
    data = {
        'snapshots': snapshots,
        'system_name': system_name,
        'method_name': method_name,
        'delta_t': delta_t,
        'times': times,
        'lambda_values': lambda_values,  # Save actual lambda values
        'ansatz_params': ansatz_params,
        'loss_histories': loss_histories,
        'param_history': param_history
    }
    
    # Save to pickle file
    filename = f"data/{system_name}_{method_name}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved simulation data to {filename}")

def load_simulation_data(system_name, method_name):
    """Load simulation data from a pickle file."""
    filename = f"data/{system_name}_{method_name}.pkl"
    
    if not os.path.exists(filename):
        print(f"Data file {filename} not found.")
        return None
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded simulation data from {filename}")
    return data

def create_comparison_plots(all_snapshots, delta_t, make_V, system_name, dim):
    """Create comparison plots for all four simulation methods."""
    # Create figures directory
    os.makedirs("figures", exist_ok=True)
    ansatz_dir = f"figures/polynomial"
    os.makedirs(ansatz_dir, exist_ok=True)
    
    # Create a comprehensive comparison ridge plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Get time points - find the first available snapshot key
    first_snapshot = None
    for snapshots in all_snapshots.values():
        if isinstance(snapshots, dict) and any(key in snapshots for key in ['naive', 'naive_weighted', 'cd_pre_equil', 'cd_weighted']):
            for key in ['naive', 'naive_weighted', 'cd_pre_equil', 'cd_weighted']:
                if key in snapshots:
                    first_snapshot = snapshots[key]
                    break
            if first_snapshot:
                break
    
    if not first_snapshot:
        print("No valid snapshots found for plotting")
        return
    
    times = np.arange(len(first_snapshot)) * delta_t
    
    # Find global range for consistent x-axis
    all_qs = []
    for method, snapshots in all_snapshots.items():
        if method == 'naive_unweighted':
            all_qs.extend(snapshots['naive'])
        elif method == 'naive_weighted':
            all_qs.extend(snapshots['naive_weighted'])
        elif method == 'cd_unweighted':
            all_qs.extend(snapshots['cd_pre_equil'])  # Fixed: use cd_pre_equil instead of cd_post_equil
        elif method == 'cd_weighted':
            all_qs.extend(snapshots['cd_weighted'])
    x_min = np.min(np.concatenate(all_qs)) - 0.5
    x_max = np.max(np.concatenate(all_qs)) + 0.5
    x_grid = np.linspace(x_min, x_max, 200)
    
    # Plot each method
    colors = {'naive_unweighted': 'blue', 'naive_weighted': 'green', 'cd_unweighted': 'red', 'cd_weighted': 'orange'}
    titles = {
        'naive_unweighted': 'Naive HMC (Unweighted)',
        'naive_weighted': 'Naive HMC (Weighted SMC)',
        'cd_unweighted': 'Counterdiabatic HMC (Unweighted)',
        'cd_weighted': 'Counterdiabatic HMC (Weighted)'
    }
    
    # Filter out non-snapshot keys (like loss_histories and param_history)
    snapshot_methods = {k: v for k, v in all_snapshots.items() if k in titles}
    
    for i, (method, snapshots) in enumerate(snapshot_methods.items()):
        ax = axes[i]
        ax.set_title(titles[method], fontsize=14, fontweight='bold')
        ax.set_xlabel("Position q", fontsize=12)
        ax.set_ylabel("Time t", fontsize=12)
        
        # Choose the correct snapshot key for each method
        if method == 'naive_unweighted':
            snapshot_key = 'naive'
            weights_key = None
        elif method == 'naive_weighted':
            snapshot_key = 'naive_weighted'
            weights_key = 'weights_naive'
        elif method == 'cd_unweighted':
            snapshot_key = 'cd_pre_equil'
            weights_key = None
        elif method == 'cd_weighted':
            snapshot_key = 'cd_weighted'
            weights_key = 'weights_cd'
        else:
            snapshot_key = 'naive'  # fallback
            weights_key = None
        
        # Choose the correct lambda values for each method
        if method in ['cd_unweighted', 'cd_weighted']:
            lam_key = 'lam_pre_equil'
        else:
            lam_key = 'lam_pre_equil'
        
        # Plot distributions
        for j, (t, snap, lam_val) in enumerate(zip(times, snapshots[snapshot_key], snapshots[lam_key])):
            # Get weights if available
            weights = None
            if weights_key and snapshots[weights_key][j] is not None:
                log_weights = snapshots[weights_key][j]
                # Check if all log weights are zero (unit weights)
                if np.allclose(log_weights, 0.0):
                    weights = None  # Use unweighted histogram
                else:
                    weights = np.exp(log_weights - np.max(log_weights))
                    weights = weights / np.sum(weights)
            
            # Compute KDE
            try:
                from scipy.stats import gaussian_kde
                if weights is not None:
                    # Use weighted KDE
                    kde = gaussian_kde(snap.flatten(), weights=weights)
                else:
                    kde = gaussian_kde(snap.flatten())
                density = kde(x_grid)
            except:
                # Fallback to histogram
                if weights is not None:
                    hist, bin_edges = np.histogram(snap, bins=50, density=True, range=(x_min, x_max), weights=weights)
                else:
                    hist, bin_edges = np.histogram(snap, bins=50, density=True, range=(x_min, x_max))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                density = np.interp(x_grid, bin_centers, hist)
            
            # Normalize and offset for ridge plot
            density = density / np.max(density) * 1.8
            offset = t * 2.0  # Increased spacing between plots to reduce overlap
            
            # Plot the ridge
            ax.fill_between(x_grid, offset, offset + density, 
                           color=colors[method], alpha=0.4, edgecolor=colors[method], linewidth=0.5)
            
            # Add true distribution
            potential_fn = make_V(lam_val)
            rho = np.array([np.exp(-potential_fn(x)) for x in x_grid])
            rho = rho / np.max(rho) * 1.8
            ax.plot(x_grid, offset + rho, 'k--', linewidth=1.5, alpha=0.8)
        
        # Set limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(times[0] * 2.0 - 0.1, times[-1] * 2.0 + 2.0)  # Adjusted for increased spacing
        ax.set_yticks(times * 2.0)
    
    plt.tight_layout()
    plt.savefig(f"{ansatz_dir}/comparison_ridge_plot_{system_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison ridge plot to {ansatz_dir}/comparison_ridge_plot_{system_name}.png")

def main(run_simulations=True):
    # Set up the system
    # system_name = "gaussian_moving_mean"
    system_name = "gaussian_annealing"
    # system_name = "double_well"
    make_T, make_V, system_description, dim = get_system(system_name)
    
    # Flag to control whether to run simulations or load from data
    # run_simulations = True  # Set to False to load from saved data
    
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
    momentum_refresh_interval =  5.0
    fit_every = 1
    num_initial_iterations = 10000
    num_iterations = 10000
    learning_rate = 1e-4
    re_equil_steps = 0
    ess_threshold = 0.5
    snapshot_every = 1  # Record snapshots every N steps (1 = every step, 10 = every 10 steps, etc.)
    
    # Create ansatz
    key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
    if dim == 1:
        # For 1D: input [q, p] -> hidden layers -> output
        dims = [2*dim, 32, 16, 1]  # [2, 32, 16, 1]
    else:
        # For higher dimensions: adjust network size
        dims = [2*dim, 64, 32, 16, 1]  # [2*dim, 64, 32, 16, 1]
    
    ansatz = NeuralNetworkAnsatz(dims=dims, key=key, dim=dim)
    
    # Storage for all simulation results
    successful_simulations = {}

    if run_simulations:
        # Run simulations and save data
        key = jax.random.PRNGKey(0)
        
        # Define simulation configurations
        naive_configs = [
            {'name': 'naive_unweighted', 'use_weights': False, 'ess_threshold': None},
            {'name': 'naive_weighted', 'use_weights': True, 'ess_threshold': ess_threshold}
        ]
        
        cd_configs = [
            {'name': 'cd_unweighted', 'use_weights': False, 'ess_threshold': None},
            {'name': 'cd_weighted', 'use_weights': True, 'ess_threshold': ess_threshold}
        ]
        
        # Run naive HMC simulations
        for config in naive_configs:
            print(f"\n{'='*50}")
            print(f"Running Naive HMC ({config['name'].replace('_', ' ').title()})")
            print(f"{'='*50}")
            
            try:
                # Prepare parameters
                kwargs = {
                    'M': M, 'N_steps': N_steps, 'delta_t': delta_t, 'eps': eps,
                    'momentum_refresh_interval': momentum_refresh_interval,
                    'make_T': make_T, 'make_V': make_V, 'lam_fn': lam_fn, 'dot_lam_fn': dot_lam_fn,
                    'key': key, 'dim': dim, 'use_weights': config['use_weights'], 
                    'snapshot_every': snapshot_every
                }
                if config['ess_threshold'] is not None:
                    kwargs['ess_threshold'] = config['ess_threshold']
                
                snapshots = run_naive_hmc_simulation(**kwargs)
                successful_simulations[config['name']] = snapshots
                
                # Save data
                save_simulation_data(snapshots, system_name, config['name'], delta_t, lam_fn)
                print(f"✓ Naive HMC ({config['name'].replace('_', ' ').title()}) completed successfully")
                
            except Exception as e:
                print(f"✗ Naive HMC ({config['name'].replace('_', ' ').title()}) failed: {e}")
        
        # Run counterdiabatic HMC simulations
        for config in cd_configs:
            print(f"\n{'='*50}")
            print(f"Running Counterdiabatic HMC ({config['name'].replace('_', ' ').title()})")
            print(f"{'='*50}")
            
            try:
                # Prepare parameters
                kwargs = {
                    'M': M, 'N_steps': N_steps, 'delta_t': delta_t, 'eps': eps,
                    'momentum_refresh_interval': momentum_refresh_interval,
                    'fit_every': fit_every, 'num_initial_iterations': num_initial_iterations,
                    'num_iterations': num_iterations, 'make_T': make_T, 'make_V': make_V,
                    'A_ansatz': ansatz, 'lam_fn': lam_fn, 'dot_lam_fn': dot_lam_fn,
                    'key': key, 'dim': dim, 'learning_rate': learning_rate,
                    're_equil_steps': re_equil_steps, 'use_weights': config['use_weights'], 
                    'snapshot_every': snapshot_every
                }
                if config['ess_threshold'] is not None:
                    kwargs['ess_threshold'] = config['ess_threshold']
                
                _, snapshots, loss_histories, param_history = run_simulation(**kwargs)
                successful_simulations[config['name']] = snapshots
                successful_simulations[f'loss_histories_{config["name"]}'] = loss_histories
                successful_simulations[f'param_history_{config["name"]}'] = param_history
                
                # Save data
                save_simulation_data(snapshots, system_name, config['name'], delta_t, lam_fn, 
                                   ansatz_params=ansatz, loss_histories=loss_histories, 
                                   param_history=param_history)
                print(f"✓ Counterdiabatic HMC ({config['name'].replace('_', ' ').title()}) completed successfully")
                
            except Exception as e:
                print(f"✗ Counterdiabatic HMC ({config['name'].replace('_', ' ').title()}) failed: {e}")
    else:
        # Load data from saved files
        print("Loading simulation data from saved files...")
        
        # Try to load each method's data
        methods = ['naive_unweighted', 'naive_weighted', 'cd_unweighted', 'cd_weighted']
        for method in methods:
            data = load_simulation_data(system_name, method)
            if data is not None:
                successful_simulations[method] = data['snapshots']
                if 'loss_histories' in data and data['loss_histories'] is not None:
                    successful_simulations[f'loss_histories_{method}'] = data['loss_histories']
                if 'param_history' in data and data['param_history'] is not None:
                    successful_simulations[f'param_history_{method}'] = data['param_history']
                print(f"✓ Loaded {method} data")
            else:
                print(f"✗ Could not load {method} data")

    # Create comparison plots
    if len(successful_simulations) > 0:
        print(f"\nCreating comparison plots for {len(successful_simulations)} successful simulations...")
        create_comparison_plots(successful_simulations, delta_t, make_V, system_name, dim)
        
        # Create detailed distribution plots for counterdiabatic methods
        cd_methods = ['cd_unweighted', 'cd_weighted']
        for method in cd_methods:
            if method in successful_simulations:
                print(f"Creating detailed distribution plot for {method.replace('_', ' ')} case...")
                loss_histories = successful_simulations.get(f'loss_histories_{method}', [])
                param_history = successful_simulations.get(f'param_history_{method}', None)
                naive_method = method.replace('cd_', 'naive_')
                naive_snapshots = successful_simulations.get(naive_method, None)
                
                plot_results(successful_simulations[method], loss_histories, delta_t, make_V, 
                            param_history=param_history, ansatz=ansatz, 
                            potential_name=f"{system_name}_{method}", dim=dim, plot_ansatz=False, 
                            make_T=make_T, naive_snapshots=naive_snapshots)
    else:
        print("No successful simulations to plot.")

if __name__ == "__main__":
    main(run_simulations=True) 