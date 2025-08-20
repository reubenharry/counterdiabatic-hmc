import jax
import jax.numpy as jnp

from src.simulation import run_simulation, run_naive_hmc_simulation
from src.plotting import plot_results, create_ridge_plot
from src.ansatze import PolynomialAnsatz, NeuralNetworkAnsatz, AnalyticAnsatz
from src.systems import get_system
import matplotlib.pyplot as plt
import numpy as np
import os

def create_neural_network_ridge_plot(snapshots, delta_t, make_V, lam_fn, system_name, ansatz):
    """Create a ridge plot specifically for neural network ansatz results."""
    # Create figures directory
    os.makedirs("figures", exist_ok=True)
    ansatz_dir = f"figures/neural_network"
    os.makedirs(ansatz_dir, exist_ok=True)
    
    # Get time points
    times = np.arange(len(snapshots['cd_pre_equil'])) * delta_t * 10
    
    # Find global range for consistent x-axis
    all_qs = []
    all_qs.extend(snapshots['cd_pre_equil'])
    if 'cd_post_equil' in snapshots:
        all_qs.extend(snapshots['cd_post_equil'])
    
    x_min = np.min(np.concatenate(all_qs)) - 0.5
    x_max = np.max(np.concatenate(all_qs)) + 0.5
    x_grid = np.linspace(x_min, x_max, 200)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title(f"Neural Network Ansatz - {system_name.replace('_', ' ').title()}", fontsize=16, fontweight='bold')
    ax.set_xlabel("Position q", fontsize=12)
    ax.set_ylabel("Time t", fontsize=12)
    
    # Plot CD-HMC distributions
    for j, (t, snap, lam_val) in enumerate(zip(times, snapshots['cd_pre_equil'], snapshots['lam_pre_equil'])):
        # Compute KDE
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(snap.flatten())
            density = kde(x_grid)
        except:
            # Fallback to histogram
            hist, bin_edges = np.histogram(snap, bins=50, density=True, range=(x_min, x_max))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            density = np.interp(x_grid, bin_centers, hist)
        
        # Ensure density is non-negative
        density = np.maximum(density, 0.0)
        
        # Normalize and offset for ridge plot
        density = density / np.max(density) * 1.8
        offset = t * 2.0
        
        # Plot the ridge
        ax.fill_between(x_grid, offset, offset + density, 
                       color='red', alpha=0.4, edgecolor='red', linewidth=0.5)
        
        # Add true distribution
        potential_fn = make_V(lam_val)
        rho = np.array([np.exp(-potential_fn(x)) for x in x_grid])
        rho = rho / np.max(rho) * 1.8
        ax.plot(x_grid, offset + rho, 'k--', linewidth=1.5, alpha=0.8)
    
    # Add post-equilibration snapshots if available
    if 'cd_post_equil' in snapshots and len(snapshots['cd_post_equil']) > 0:
        post_equil_times = np.arange(1, len(snapshots['cd_post_equil']) + 1) * delta_t * 10
        for j, (t, snap, lam_val) in enumerate(zip(post_equil_times, snapshots['cd_post_equil'], snapshots['lam_post_equil'])):
            # Compute KDE
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(snap.flatten())
                density = kde(x_grid)
            except:
                # Fallback to histogram
                hist, bin_edges = np.histogram(snap, bins=50, density=True, range=(x_min, x_max))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                density = np.interp(x_grid, bin_centers, hist)
            
            # Ensure density is non-negative
            density = np.maximum(density, 0.0)
            
            # Normalize and offset for ridge plot
            density = density / np.max(density) * 1.8
            offset = t * 2.0
            
            # Plot the ridge (post-equilibration in different color)
            ax.fill_between(x_grid, offset, offset + density, 
                           color='blue', alpha=0.4, edgecolor='blue', linewidth=0.5)
            
            # Add true distribution
            potential_fn = make_V(lam_val)
            rho = np.array([np.exp(-potential_fn(x)) for x in x_grid])
            rho = rho / np.max(rho) * 1.8
            ax.plot(x_grid, offset + rho, 'k--', linewidth=1.5, alpha=0.8)
    
    # Set limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(times[0] * 2.0 - 0.1, times[-1] * 2.0 + 2.0)
    ax.set_yticks(times * 2.0)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.4, label='CD-HMC (Pre-equilibration)'),
        Patch(facecolor='blue', alpha=0.4, label='CD-HMC (Post-equilibration)'),
        plt.Line2D([0], [0], color='k', linestyle='--', linewidth=1.5, label='True Distribution')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{ansatz_dir}/ridge_plot_{system_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved neural network ridge plot to {ansatz_dir}/ridge_plot_{system_name}.png")

def create_comparison_plots(all_snapshots, delta_t, make_V, lam_fn, system_name, dim):
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
        
    times = np.arange(len(first_snapshot)) * delta_t * 10
    
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
            
            # Ensure density is non-negative (fix for interpolation issues)
            density = np.maximum(density, 0.0)
            
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

def main():
    # Set up the system
    # system_name = "gaussian_moving_mean"
    # system_name = "gaussian_annealing"
    system_name = "double_well"
    make_T, make_V, system_description, dim = get_system(system_name)
    
    # Define lambda functions
    v = 0.5
    max_lam = 1.0
    lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
    dot_lam_fn = jax.grad(lam_fn)
    
    # Parameters
    M = 1000
    N_steps = 10
    delta_t = 0.2
    eps = 0.2
    momentum_refresh_interval =  1.0
    fit_every = 1
    num_initial_iterations = 100000
    num_iterations = 100000
    learning_rate = 1e-4
    re_equil_steps = 0
    ess_threshold = 0.5
    
    # Create ansatz
    # ansatz = PolynomialAnsatz(max_degree=6, dim=dim)
    ansatz = NeuralNetworkAnsatz(dims=[2,32,32,1], dim=dim, key=jax.random.PRNGKey(0))
    
    # Storage for all simulation results
    successful_simulations = {}

    naive = False
    if naive:
    
        # 1. Naive HMC (Unweighted)
        print("\n" + "="*50)
        print("Running Naive HMC (Unweighted)")
        print("="*50)
        try:
            key = jax.random.PRNGKey(0)
            snapshots_naive_unweighted = run_naive_hmc_simulation(
                M=M, N_steps=N_steps, delta_t=delta_t, eps=eps,
                momentum_refresh_interval=momentum_refresh_interval,
                make_T=make_T, make_V=make_V, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
                key=key, dim=dim, use_weights=False
            )
            successful_simulations['naive_unweighted'] = snapshots_naive_unweighted
            print("✓ Naive HMC (Unweighted) completed successfully")
        except Exception as e:
            print(f"✗ Naive HMC (Unweighted) failed: {e}")
        
        # 2. Naive HMC (Weighted SMC)
        print("\n" + "="*50)
        print("Running Naive HMC (Weighted SMC)")
        print("="*50)
        key = jax.random.PRNGKey(0)
        snapshots_naive_weighted = run_naive_hmc_simulation(
            M=M, N_steps=N_steps, delta_t=delta_t, eps=eps,
            momentum_refresh_interval=momentum_refresh_interval,
            make_T=make_T, make_V=make_V, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
            key=key, dim=dim, use_weights=True, ess_threshold=ess_threshold
        )
        successful_simulations['naive_weighted'] = snapshots_naive_weighted
        print("✓ Naive HMC (Weighted SMC) completed successfully")
        
    counterdiabatic = True
    if counterdiabatic:
    # 3. Counterdiabatic HMC (Unweighted)
        print("\n" + "="*50)
        print("Running Counterdiabatic HMC (Unweighted)")
        print("="*50)
        try:
            key = jax.random.PRNGKey(0)
            _, snapshots_cd_unweighted, loss_histories_cd_unweighted, param_history_cd_unweighted = run_simulation(
                M=M, N_steps=N_steps, delta_t=delta_t, eps=eps,
                momentum_refresh_interval=momentum_refresh_interval,
                fit_every=fit_every, num_initial_iterations=num_initial_iterations,
                num_iterations=num_iterations, make_T=make_T, make_V=make_V,
                A_ansatz=ansatz, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
                key=key, dim=dim, learning_rate=learning_rate,
                re_equil_steps=re_equil_steps, use_weights=False
            )
            successful_simulations['cd_unweighted'] = snapshots_cd_unweighted
            successful_simulations['loss_histories_cd_unweighted'] = loss_histories_cd_unweighted
            successful_simulations['param_history_cd_unweighted'] = param_history_cd_unweighted
            print("✓ Counterdiabatic HMC (Unweighted) completed successfully")
        except Exception as e:
            print(f"✗ Counterdiabatic HMC (Unweighted) failed: {e}")
        
        # # 4. Counterdiabatic HMC (Weighted) - Using same seed as unweighted
        # print("\n" + "="*50)
        # print("Running Counterdiabatic HMC (Weighted) - Same seed as unweighted")
        # print("="*50)
        # try:
        #     key = jax.random.PRNGKey(0)  # Same seed as unweighted
        #     _, snapshots_cd_weighted, loss_histories_cd_weighted, param_history_cd_weighted = run_simulation(
        #         M=M, N_steps=N_steps, delta_t=delta_t, eps=eps,
        #         momentum_refresh_interval=momentum_refresh_interval,
        #         fit_every=fit_every, num_initial_iterations=num_initial_iterations,
        #         num_iterations=num_iterations, make_T=make_T, make_V=make_V,
        #         A_ansatz=ansatz, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
        #         key=key, dim=dim, learning_rate=learning_rate,
        #         re_equil_steps=re_equil_steps, use_weights=True, ess_threshold=ess_threshold
        #     )
        #     successful_simulations['cd_weighted'] = snapshots_cd_weighted
        #     successful_simulations['loss_histories_cd_weighted'] = loss_histories_cd_weighted
        #     successful_simulations['param_history_cd_weighted'] = param_history_cd_weighted
        #     print("✓ Counterdiabatic HMC (Weighted) completed successfully")
        # except Exception as e:
        #     print(f"✗ Counterdiabatic HMC (Weighted) failed: {e}")
    
    # Create plots
    if len(successful_simulations) > 0:
        print(f"\nCreating plots for {len(successful_simulations)} successful simulations...")
        
        # Create comparison plots if multiple simulation types are available
        if len(successful_simulations) > 1:
            create_comparison_plots(successful_simulations, delta_t, make_V, lam_fn, system_name, dim)
        
        # Create ridge plots for neural network ansatz
        if 'cd_unweighted' in successful_simulations:
            print("Creating ridge plot for neural network ansatz...")
            # Create a simple ridge plot for the neural network ansatz
            create_neural_network_ridge_plot(successful_simulations['cd_unweighted'], delta_t, make_V, lam_fn, system_name, ansatz)
            
            print("Creating detailed distribution plot...")
            loss_histories = successful_simulations.get('loss_histories_cd_unweighted', [])
            param_history = successful_simulations.get('param_history_cd_unweighted', None)
            naive_snapshots = successful_simulations.get('naive_unweighted', None)
            plot_results(successful_simulations['cd_unweighted'], loss_histories, delta_t, make_V, lam_fn, 
                        param_history=param_history, ansatz=ansatz, potential_name=system_name, dim=dim, plot_ansatz=False, make_T=make_T, naive_snapshots=naive_snapshots)
        elif 'cd_weighted' in successful_simulations:
            print("Creating ridge plot for neural network ansatz...")
            # Create a simple ridge plot for the neural network ansatz
            create_neural_network_ridge_plot(successful_simulations['cd_weighted'], delta_t, make_V, lam_fn, system_name, ansatz)
            
            print("Creating detailed distribution plot...")
            loss_histories = successful_simulations.get('loss_histories_cd_weighted', [])
            param_history = successful_simulations.get('param_history_cd_weighted', None)
            naive_snapshots = successful_simulations.get('naive_weighted', None)
            plot_results(successful_simulations['cd_weighted'], loss_histories, delta_t, make_V, lam_fn, 
                        param_history=param_history, ansatz=ansatz, potential_name=system_name, dim=dim, plot_ansatz=False, make_T=make_T, naive_snapshots=naive_snapshots)
        

    else:
        print("No successful simulations to plot.")

if __name__ == "__main__":
    main() 