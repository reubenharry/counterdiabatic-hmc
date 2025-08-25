import jax
import jax.numpy as jnp
import pickle
import os

from src.simulation import run_simulation_and_save_data
from src.plotting import plot_results, create_ridge_plot, create_all_plots
from src.ansatze import PolynomialAnsatz, NeuralNetworkAnsatz, AnalyticAnsatz
from src.systems import get_system
import matplotlib.pyplot as plt
import numpy as np





def main():
    """
    Main function for running counterdiabatic simulations.
    Easily configure the system and ansatz type here.
    """
    # ===== CONFIGURATION =====
    # Choose your system
    system_name = "gaussian_annealing"  # Options: see SYSTEMS in src/systems.py
    
    # Choose your ansatz type
    ansatz_type = "polynomial"  # Options: "polynomial", "neural_network", "analytic"
    
    # Simulation parameters
    M = 2000  # Number of particles
    N_steps = 10  # Number of simulation steps
    delta_t = 0.2  # Time step
    eps = 0.2  # HMC step size
    momentum_refresh_interval = 2  # Momentum refresh interval
    fit_every = 1  # Fit ansatz every N steps
    num_initial_iterations = 100000  # Initial optimization iterations
    num_iterations = 100000  # Optimization iterations per step
    learning_rate = 1e-4  # Learning rate for optimization
    re_equil_steps = 0  # Re-equilibration steps
    ess_threshold = 0.5  # Effective sample size threshold for resampling
    
    # Simulation settings
    run_simulations = True  # Set to False to load from saved data
    snapshot_every = 1  # Record snapshots every N steps
    
    # ===== SYSTEM SETUP =====
    # Get system from systems.py
    make_T, make_V, system_description, dim = get_system(system_name)
    print(f"Using system: {system_name}")
    print(f"Description: {system_description}")
    print(f"Dimension: {dim}")
    
    # Define lambda functions
    v = 0.5
    max_lam = 1.0
    lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
    dot_lam_fn = jax.grad(lam_fn)
    
    # ===== ANSATZ SETUP =====
    if ansatz_type == "polynomial":
        ansatz = PolynomialAnsatz(max_degree=2, dim=dim)
    elif ansatz_type == "neural_network":
        key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
        if dim == 1:
            dims = [2*dim, 32, 16, 1]  # [2, 32, 16, 1]
        else:
            dims = [2*dim, 64, 32, 16, 1]  # [2*dim, 64, 32, 16, 1]
        ansatz = NeuralNetworkAnsatz(dims=dims, key=key, dim=dim)
    elif ansatz_type == "analytic":
        ansatz = AnalyticAnsatz()
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")
    
    # ===== RUN SIMULATIONS =====
    successful_simulations = run_simulation_and_save_data(
        system_name=system_name,
        ansatz=ansatz,
        lam_fn=lam_fn,
        dot_lam_fn=dot_lam_fn,
        run_simulations=run_simulations,
        snapshot_every=snapshot_every,
        M=M,
        N_steps=N_steps,
        delta_t=delta_t,
        eps=eps,
        momentum_refresh_interval=momentum_refresh_interval,
        fit_every=fit_every,
        num_initial_iterations=num_initial_iterations,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        re_equil_steps=re_equil_steps,
        ess_threshold=ess_threshold
    )
    
    # ===== CREATE PLOTS =====
    if len(successful_simulations) > 0:
        create_all_plots(
            successful_simulations=successful_simulations,
            system_name=system_name,
            ansatz=ansatz,
            delta_t=delta_t,
            make_V=make_V,
            make_T=make_T
        )
    else:
        print("No successful simulations to plot.")

if __name__ == "__main__":
    main() 