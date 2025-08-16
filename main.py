import jax
import jax.numpy as jnp

from src.simulation import run_simulation, run_naive_hmc_simulation
from src.plotting import plot_results, create_ridge_plot
from src.ansatze import PolynomialAnsatz, NeuralNetworkAnsatz, AnalyticAnsatz
from src.systems import get_system
import matplotlib.pyplot as plt
import numpy as np
import os

def create_comparison_plots(all_snapshots, delta_t, make_V, lam_fn, system_name, dim):
    """Create comparison plots for all three simulation methods."""
    # Create figures directory
    os.makedirs("figures", exist_ok=True)
    ansatz_dir = f"figures/polynomial"
    os.makedirs(ansatz_dir, exist_ok=True)
    
    # Create a comprehensive comparison ridge plot
    fig, axes = plt.subplots(1, len(all_snapshots), figsize=(6*len(all_snapshots), 6))
    if len(all_snapshots) == 1:
        axes = [axes]
    
    # Get time points
    times = np.arange(len(list(all_snapshots.values())[0]['naive'])) * delta_t * 10
    
    # Find global range for consistent x-axis
    all_qs = []
    for method, snapshots in all_snapshots.items():
        if method == 'unweighted':
            all_qs.extend(snapshots['naive'])
        elif method == 'weighted':
            all_qs.extend(snapshots['naive_weighted'])
        elif method == 'cd':
            all_qs.extend(snapshots['cd_post_equil'])
    x_min = np.min(np.concatenate(all_qs)) - 0.5
    x_max = np.max(np.concatenate(all_qs)) + 0.5
    x_grid = np.linspace(x_min, x_max, 200)
    
    # Plot each method
    colors = {'unweighted': 'blue', 'weighted': 'green', 'cd': 'red'}
    titles = {
        'unweighted': 'Naive HMC (Unweighted)',
        'weighted': 'Naive HMC (Weighted SMC)',
        'cd': 'Counterdiabatic HMC'
    }
    
    for i, (method, snapshots) in enumerate(all_snapshots.items()):
        ax = axes[i]
        ax.set_title(titles[method], fontsize=14, fontweight='bold')
        ax.set_xlabel("Position q", fontsize=12)
        ax.set_ylabel("Time t", fontsize=12)
        
        # Choose the correct snapshot key for each method
        if method == 'unweighted':
            snapshot_key = 'naive'
        elif method == 'weighted':
            snapshot_key = 'naive_weighted'
        elif method == 'cd':
            snapshot_key = 'cd_post_equil'  # Use post-equilibration results
        else:
            snapshot_key = 'naive'  # fallback
        
        # Plot distributions
        for j, (t, snap, lam_val) in enumerate(zip(times, snapshots[snapshot_key], snapshots['lam_pre_equil'])):
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
            
            # Normalize and offset for ridge plot
            density = density / np.max(density) * 1.8
            offset = t
            
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
        ax.set_ylim(times[0] - 0.1, times[-1] + 2.0)
        ax.set_yticks(times)
    
    plt.tight_layout()
    plt.savefig(f"{ansatz_dir}/comparison_ridge_plot_{system_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison ridge plot to {ansatz_dir}/comparison_ridge_plot_{system_name}.png")

def main():
    # Define all routines and parameters here
    M = 2048  # Reduced from 3000
    N_steps = 40  # Reduced from 1000
    eps = 0.05  # Increased from 0.0001 for faster simulation
    delta_t = eps # should this even be a parameter?
    momentum_refresh_interval = 1/eps # jnp.sqrt(1)/eps
    fit_every = 1  # Fit the gauge potential every N steps
    num_initial_iterations = 100000  # Reduced from 10000
    num_iterations = 100000  # Reduced from 10000
    v = 0.5
    max_lam = 1.0
    # lam_fn = lambda t: 0.0
    lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
    dot_lam_fn = jax.grad(lam_fn)
    learning_rate = 5e-4
    
    # Re-equilibration parameters
    re_equil_steps = 1000  # Number of naive HMC steps after each CD step
    
    # Choose the system to simulate
    # system_name = "2d_gaussian_moving_mean"  # Try the 2D system
    # system_name = "2d_gaussian_annealing"  # Try the 2D system
    # system_name = "gaussian_moving_mean"
    system_name = "gaussian_annealing"
    # system_name = "double_well"
    # system_name = "2d_normal_to_rosenbrock"
    make_T, make_V, system_description, dim = get_system(system_name)
    print(f"Using system: {system_name}")
    print(f"Description: {system_description}")
    print(f"Dimension: {dim}")
    
    # Initialize ansatz (either neural network or polynomial)
    key = jax.random.PRNGKey(0)
    # For neural network:
    # ansatz = NeuralNetworkAnsatz([2*dim, 128, 256, 128, 1], key, dim=dim)
    # ansatz = NeuralNetworkAnsatz([2*dim, 16, 1], key, dim=dim)
    # For polynomial:
    ansatz = PolynomialAnsatz(max_degree=2, dim=dim)  
    # For analytic solution:
    # ansatz = AnalyticAnsatz()
    
    # Print polynomial terms if using polynomial ansatz
    if isinstance(ansatz, PolynomialAnsatz):
        print("Polynomial terms:")
        for desc in ansatz.get_term_description():
            print(f"  {desc}")
        print(f"Total number of parameters: {len(ansatz.params)}")
    
    import time
    start_time = time.time()
    
    # Run three separate simulations for comparison
    all_snapshots = {}
    all_loss_histories = {}
    all_param_histories = {}
    
    # 1. Naive HMC without weights
    print("\n" + "="*60)
    print("RUNNING NAIVE HMC WITHOUT WEIGHTS")
    print("="*60)
    key, subkey = jax.random.split(key)
    try:
        snapshots_unweighted = run_naive_hmc_simulation(
            M=M, 
            N_steps=N_steps, 
            delta_t=delta_t, 
            eps=eps, 
            momentum_refresh_interval=momentum_refresh_interval,
            make_T=make_T, 
            make_V=make_V, 
            lam_fn=lam_fn, 
            dot_lam_fn=dot_lam_fn, 
            key=subkey,
            dim=dim,
            use_weights=False
        )
        all_snapshots['unweighted'] = snapshots_unweighted
        all_loss_histories['unweighted'] = []  # No loss history for naive HMC
        all_param_histories['unweighted'] = []  # No parameter history for naive HMC
        print("✓ Naive HMC without weights completed successfully")
    except Exception as e:
        print(f"❌ Naive HMC without weights failed: {e}")
        all_snapshots['unweighted'] = None
    
    # 2. Naive HMC with weights and resampling
    print("\n" + "="*60)
    print("RUNNING NAIVE HMC WITH WEIGHTS AND RESAMPLING")
    print("="*60)
    key, subkey = jax.random.split(key)
    try:
        snapshots_weighted = run_naive_hmc_simulation(
            M=M, 
            N_steps=N_steps, 
            delta_t=delta_t, 
            eps=eps, 
            momentum_refresh_interval=momentum_refresh_interval,
            make_T=make_T, 
            make_V=make_V, 
            lam_fn=lam_fn, 
            dot_lam_fn=dot_lam_fn, 
            key=subkey,
            dim=dim,
            use_weights=True,
            ess_threshold=0.5
        )
        all_snapshots['weighted'] = snapshots_weighted
        all_loss_histories['weighted'] = []  # No loss history for naive HMC
        all_param_histories['weighted'] = []  # No parameter history for naive HMC
        print("✓ Naive HMC with weights completed successfully")
    except Exception as e:
        print(f"❌ Naive HMC with weights failed: {e}")
        all_snapshots['weighted'] = None
    
    # 3. Counterdiabatic HMC
    print("\n" + "="*60)
    print("RUNNING COUNTERDIABATIC HMC")
    print("="*60)
    key, subkey = jax.random.split(key)
    try:
        A_ansatz_cd, snapshots_cd, loss_histories_cd, param_history_cd = run_simulation(
            M=M, 
            N_steps=N_steps, 
            delta_t=delta_t, 
            eps=eps, 
            momentum_refresh_interval=momentum_refresh_interval,
            fit_every=fit_every,
            num_initial_iterations=num_initial_iterations,
            num_iterations=num_iterations,
            make_T=make_T, 
            make_V=make_V, 
            lam_fn=lam_fn, 
            dot_lam_fn=dot_lam_fn, 
            A_ansatz=ansatz, 
            key=subkey,
            dim=dim,
            learning_rate=learning_rate,
            re_equil_steps=re_equil_steps,  # Re-equilibration for CD HMC
            use_weights=False
        )
        all_snapshots['cd'] = snapshots_cd
        all_loss_histories['cd'] = loss_histories_cd
        all_param_histories['cd'] = param_history_cd
        print("✓ Counterdiabatic HMC completed successfully")
    except Exception as e:
        print(f"❌ Counterdiabatic HMC failed: {e}")
        all_snapshots['cd'] = None
    
    simulation_time = time.time() - start_time
    print(f"\nAll simulations completed in {simulation_time:.2f} seconds")
    
    # Create comparison plots
    print("\n" + "="*60)
    print("CREATING COMPARISON PLOTS")
    print("="*60)
    
    # Check which simulations succeeded
    successful_simulations = {k: v for k, v in all_snapshots.items() if v is not None}
    
    if len(successful_simulations) > 0:
        plot_start = time.time()
        create_comparison_plots(successful_simulations, delta_t, make_V, lam_fn, system_name, dim)
        plot_time = time.time() - plot_start
        print(f"Comparison plotting completed in {plot_time:.2f} seconds")
    else:
        print("⚠️  No simulations succeeded - cannot create plots")
    
    # Create original distributions plot using the counterdiabatic simulation
    print("\n" + "="*60)
    print("CREATING ORIGINAL DISTRIBUTIONS PLOT")
    print("="*60)
    
    if all_snapshots['cd'] is not None:
        plot_start = time.time()
        # Use the counterdiabatic simulation for the original plot_results
        plot_results(
            snapshots=all_snapshots['cd'],
            loss_histories=all_loss_histories['cd'],
            delta_t=delta_t,
            make_V=make_V,
            lam_fn=lam_fn,
            param_history=all_param_histories['cd'],
            ansatz=ansatz,
            potential_name=system_name,
            dim=dim,
            plot_ansatz=False,
            make_T=make_T
        )
        plot_time = time.time() - plot_start
        print(f"Original distributions plotting completed in {plot_time:.2f} seconds")
    else:
        print("⚠️  Counterdiabatic simulation failed - cannot create original distributions plot")

if __name__ == '__main__':
    main() 