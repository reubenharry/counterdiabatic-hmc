import jax
import jax.numpy as jnp

from src.simulation import run_simulation, run_naive_hmc_simulation
from src.plotting import plot_results, create_ridge_plot, create_comparison_plots
from src.ansatze import PolynomialAnsatz, NeuralNetworkAnsatz, AnalyticAnsatz
from src.systems import get_system
import matplotlib.pyplot as plt
import numpy as np
import os


def save_simulation_data_to_file(snapshots, simulation_name, system_name, save_final_samples=True, simulation_params=None):
    """Save complete simulation data including all snapshots and auxiliary data."""
    if not save_final_samples:
        return
        
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Add simulation parameters to the snapshots if provided
    if simulation_params is not None:
        snapshots['simulation_params'] = simulation_params
    
    # Save the complete snapshots dictionary as a pickle file
    filename = f"data/{simulation_name}_snapshots_{system_name}.pkl"
    
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(snapshots, f)
    
    print(f"Saved complete simulation data to {filename}")
    
    # Also save just the final samples as before (for backward compatibility)
    if simulation_name == 'naive_unweighted':
        if 'naive' in snapshots and len(snapshots['naive']) > 0:
            final_samples = snapshots['naive'][-1]
            final_filename = f"data/naive_unweighted_samples_{system_name}.npy"
            np.save(final_filename, final_samples)
    elif simulation_name == 'naive_weighted':
        if 'naive_weighted' in snapshots and len(snapshots['naive_weighted']) > 0:
            final_samples = snapshots['naive_weighted'][-1]
            final_filename = f"data/naive_weighted_samples_{system_name}.npy"
            np.save(final_filename, final_samples)
    elif simulation_name in ['cd_unweighted', 'cd_weighted']:
        if 'cd' in snapshots and len(snapshots['cd']) > 0:
            final_samples = snapshots['cd'][-1]
            final_filename = f"data/{simulation_name}_samples_{system_name}.npy"
            np.save(final_filename, final_samples)


def load_and_plot_precomputed_data(system_name, ansatz_type="neural_network"):
    """Load precomputed data and generate standard plots including ridge plots and <A²> analysis.
    
    Args:
        system_name: Name of the system (e.g., "double_well", "gaussian_moving_mean")
        ansatz_type: Type of ansatz used ("neural_network", "polynomial", "analytic")
    """
    print(f"Loading precomputed data for {system_name} with {ansatz_type} ansatz...")
    
    # Get system functions
    make_T, make_V, system_description, dim = get_system(system_name)
    
    # Define lambda functions (same as in main)
    v = 0.5
    max_lam = 1.0
    lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
    # dot_lam_fn = jax.grad(lam_fn)
    
    # Parameters will be loaded from saved data
    # (These are just placeholders, actual values come from simulation_params in saved files)
    
    # Create ansatz (needed for plotting)
    if ansatz_type == "neural_network":
        ansatz = NeuralNetworkAnsatz(dims=[2,32,32,1], dim=dim, key=jax.random.PRNGKey(0))
    elif ansatz_type == "polynomial":
        ansatz = PolynomialAnsatz(max_degree=2, dim=dim)
    elif ansatz_type == "analytic":
        ansatz = AnalyticAnsatz()
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")
    
    # Load saved complete snapshots
    successful_simulations = {}
    
    # Try to load each simulation type
    simulation_types = ['naive_unweighted', 'naive_weighted', 'cd_unweighted', 'cd_weighted']
    
    import pickle
    
    for sim_type in simulation_types:
        filename = f"data/{sim_type}_snapshots_{system_name}.pkl"
        try:
            with open(filename, 'rb') as f:
                snapshots = pickle.load(f)
            print(f"✓ Loaded {sim_type}: complete snapshots")
            successful_simulations[sim_type] = snapshots
            
        except FileNotFoundError:
            print(f"✗ {sim_type}: File not found ({filename})")
        except Exception as e:
            print(f"✗ {sim_type}: Error loading {e}")
    
    if len(successful_simulations) == 0:
        print("No precomputed data found!")
        return
        
    print(f"\nFound {len(successful_simulations)} simulation types. Generating full plots...")
    
    # Generate plots using the same logic as main()
    if len(successful_simulations) > 1:
        # Create comparison ridge plot with all simulation types
        print("Creating comparison ridge plot with all simulation types...")
        create_comparison_plots(system_name, make_V, lam_fn, dim, ansatz_type, subsample=2)  # Show every other snapshot
    else:
        # Fallback to single simulation ridge plot
        print("Creating single simulation ridge plot...")
        sim_type = list(successful_simulations.keys())[0]
        snapshots = successful_simulations[sim_type]
        # Get delta_t from snapshots or use fallback
        if 'simulation_params' in snapshots:
            saved_delta_t = snapshots['simulation_params'].get('delta_t', 0.02)
        else:
            saved_delta_t = 0.02
        create_ridge_plot(snapshots, saved_delta_t, make_V, lam_fn, system_name, ansatz_type, subsample=2)  # Show every other snapshot
    
    # Create detailed distribution plots for CD simulations (includes <A²> plots)
    for sim_type in ['cd_unweighted', 'cd_weighted']:
        if sim_type in successful_simulations:
            print(f"Creating detailed distribution plot for {sim_type}...")
            snapshots = successful_simulations[sim_type]
            
            # Get corresponding naive snapshots for comparison
            naive_snapshots = None
            if sim_type == 'cd_unweighted' and 'naive_unweighted' in successful_simulations:
                naive_snapshots = successful_simulations['naive_unweighted']
            elif sim_type == 'cd_weighted' and 'naive_weighted' in successful_simulations:
                naive_snapshots = successful_simulations['naive_weighted']
            
            # Load simulation parameters from saved data
            if 'simulation_params' in snapshots:
                saved_delta_t = snapshots['simulation_params'].get('delta_t', 0.02)
            else:
                saved_delta_t = 0.02  # fallback
            
            # Load loss histories and parameter history from saved data
            loss_histories = snapshots.get('loss_histories', [])
            param_history = snapshots.get('param_history', None)
            
            # Create a modified potential name to avoid overwriting
            modified_potential_name = f"{system_name}_{sim_type}"
            plot_results(snapshots, loss_histories, saved_delta_t, make_V, lam_fn, 
                        param_history=param_history, ansatz=ansatz, potential_name=modified_potential_name, 
                        dim=dim, plot_ansatz=False, make_T=make_T, naive_snapshots=naive_snapshots, sim_type=sim_type)
    
    # Create summary statistics table
    print("\nFinal Distribution Statistics:")
    print("-" * 80)
    print(f"{'Method':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 80)
    
    titles = {
        'naive_unweighted': 'Naive HMC (Unweighted)',
        'naive_weighted': 'Naive HMC (Weighted SMC)',
        'cd_unweighted': 'Counterdiabatic HMC (Unweighted)',
        'cd_weighted': 'Counterdiabatic HMC (Weighted)'
    }
    
    for sim_type, snapshots in successful_simulations.items():
        # Get final samples
        if sim_type.startswith('naive'):
            if 'naive' in snapshots and len(snapshots['naive']) > 0:
                final_samples = snapshots['naive'][-1]
            elif 'naive_weighted' in snapshots and len(snapshots['naive_weighted']) > 0:
                final_samples = snapshots['naive_weighted'][-1]
            else:
                continue
        else:  # cd simulations
            if sim_type == 'cd_unweighted' and 'cd_unweighted' in snapshots and len(snapshots['cd_unweighted']) > 0:
                final_samples = snapshots['cd_unweighted'][-1]
            elif sim_type == 'cd_weighted' and 'cd_weighted' in snapshots and len(snapshots['cd_weighted']) > 0:
                final_samples = snapshots['cd_weighted'][-1]
            else:
                continue
        
        mean_val = np.mean(final_samples)
        std_val = np.std(final_samples)
        min_val = np.min(final_samples)
        max_val = np.max(final_samples)
        print(f"{titles[sim_type]:<25} {mean_val:<10.3f} {std_val:<10.3f} {min_val:<10.3f} {max_val:<10.3f}")
    
    print("Plot generation completed!")

def run_simulations_and_save_data(system_name="double_well", ansatz_type="neural_network"):
    """Run all simulations and save data to files. Returns nothing - just saves data."""
    
    # Get system functions
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
    momentum_refresh_interval = 2.0
    fit_every = 1
    num_initial_iterations = 15000
    num_iterations = 15000
    learning_rate = 1e-4
    ess_threshold = 0.5
    snapshot_interval = 1  # Take snapshots every n steps
    
    # Save final particle populations
    save_final_samples = True
    
    # Create ansatz based on type
    if ansatz_type == "neural_network":
        ansatz = NeuralNetworkAnsatz(dims=[2,32,32,1], dim=dim, key=jax.random.PRNGKey(0))
    elif ansatz_type == "polynomial":
        ansatz = PolynomialAnsatz(max_degree=2, dim=dim)
    elif ansatz_type == "analytic":
        ansatz = AnalyticAnsatz()
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")
    
    print(f"\nRunning simulations for {system_name} with {ansatz_type} ansatz...")
    
    # Save simulation parameters
    simulation_params = {
        'delta_t': delta_t,
        'momentum_refresh_interval': momentum_refresh_interval,
        'M': M,
        'N_steps': N_steps,
        'snapshot_interval': snapshot_interval
    }
    
    # 1. Naive HMC (Unweighted)
    print("\n" + "="*50)
    print("Running Naive HMC (Unweighted)")
    print("="*50)
    try:
        key = jax.random.PRNGKey(0)
        snapshots_naive_unweighted = run_naive_hmc_simulation(
            M=M, N_steps=N_steps, delta_t=delta_t,
            momentum_refresh_interval=momentum_refresh_interval,
            make_T=make_T, make_V=make_V, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
            key=key, dim=dim, use_weights=False, snapshot_interval=snapshot_interval
        )
        save_simulation_data_to_file(snapshots_naive_unweighted, 'naive_unweighted', system_name, save_final_samples, simulation_params)
        print("✓ Naive HMC (Unweighted) completed successfully")
    except Exception as e:
        print(f"✗ Naive HMC (Unweighted) failed: {e}")
    
    # 2. Naive HMC (Weighted SMC)
    print("\n" + "="*50)
    print("Running Naive HMC (Weighted SMC)")
    print("="*50)
    try:
        key = jax.random.PRNGKey(0)
        snapshots_naive_weighted = run_naive_hmc_simulation(
            M=M, N_steps=N_steps, delta_t=delta_t,
            momentum_refresh_interval=momentum_refresh_interval,
            make_T=make_T, make_V=make_V, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
            key=key, dim=dim, use_weights=True, ess_threshold=ess_threshold, snapshot_interval=snapshot_interval
        )
        save_simulation_data_to_file(snapshots_naive_weighted, 'naive_weighted', system_name, save_final_samples, simulation_params)
        print("✓ Naive HMC (Weighted SMC) completed successfully")
    except Exception as e:
        print(f"✗ Naive HMC (Weighted SMC) failed: {e}")
    
    # 3. Counterdiabatic HMC (Unweighted)
    print("\n" + "="*50)
    print("Running Counterdiabatic HMC (Unweighted)")
    print("="*50)
    try:
        key = jax.random.PRNGKey(0)
        _, snapshots_cd_unweighted, loss_histories_cd_unweighted, param_history_cd_unweighted = run_simulation(
            M=M, N_steps=N_steps, delta_t=delta_t,
            momentum_refresh_interval=momentum_refresh_interval,
            fit_every=fit_every, num_initial_iterations=num_initial_iterations,
            num_iterations=num_iterations, make_T=make_T, make_V=make_V,
            A_ansatz=ansatz, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
            key=key, dim=dim, learning_rate=learning_rate,
            use_weights=False, snapshot_interval=snapshot_interval,
            adaptive_step_size=True, K=0.2
        )
        save_simulation_data_to_file(snapshots_cd_unweighted, 'cd_unweighted', system_name, save_final_samples, simulation_params)
        print("✓ Counterdiabatic HMC (Unweighted) completed successfully")
    except Exception as e:
        print(f"✗ Counterdiabatic HMC (Unweighted) failed: {e}")
    
    # 4. Counterdiabatic HMC (Weighted) - Using same seed as unweighted
    print("\n" + "="*50)
    print("Running Counterdiabatic HMC (Weighted) - Same seed as unweighted")
    print("="*50)
    try:
        key = jax.random.PRNGKey(0)  # Same seed as unweighted
        _, snapshots_cd_weighted, loss_histories_cd_weighted, param_history_cd_weighted = run_simulation(
            M=M, N_steps=N_steps, delta_t=delta_t,
            momentum_refresh_interval=momentum_refresh_interval,
            fit_every=fit_every, num_initial_iterations=num_initial_iterations,
            num_iterations=num_iterations, make_T=make_T, make_V=make_V,
            A_ansatz=ansatz, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
            key=key, dim=dim, learning_rate=learning_rate,
            use_weights=True, ess_threshold=ess_threshold, snapshot_interval=snapshot_interval,
            adaptive_step_size=True, K=0.2
        )
        save_simulation_data_to_file(snapshots_cd_weighted, 'cd_weighted', system_name, save_final_samples, simulation_params)
        print("✓ Counterdiabatic HMC (Weighted) completed successfully")
    except Exception as e:
        print(f"✗ Counterdiabatic HMC (Weighted) failed: {e}")
    
    print(f"\nAll simulations completed for {system_name} with {ansatz_type} ansatz!")
    print("Data saved to data/ folder. Use load_and_plot_precomputed_data() to generate plots.")

if __name__ == "__main__":
    # Step 1: Run simulations and save data

    ansatz_type = "polynomial"
    system_name = "double_well"

    print("="*60)
    print("STEP 1: Running simulations and saving data")
    print("="*60)
    run_simulations_and_save_data(system_name, ansatz_type)
    
    # Step 2: Load precomputed data and generate plots
    print("\n" + "="*60)
    print("STEP 2: Loading data and generating plots")
    print("="*60)
    load_and_plot_precomputed_data(system_name, ansatz_type)
    
    print("\n" + "="*60)
    print("All done! Check the figures/ folder for generated plots.")
    print("="*60)