#!/usr/bin/env python3
"""
Main script for running counterdiabatic simulations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
from src.simulation import run_simulation_and_save_data
from src.ansatze import PolynomialAnsatz, NeuralNetworkAnsatz, AnalyticAnsatz, HermiteAnsatz, PolynomialFAnsatz
from src.systems import get_system
from src.plotting import create_all_plots, create_2d_plots


def main():
    """
    Main function for running counterdiabatic simulations.
    Easily configure the system and ansatz type here.
    """
    # ===== CONFIGURATION =====
    # Choose your system
    system_name = "gaussian_moving_mean"  # Options: see SYSTEMS in src/systems.py
    
    # Choose your ansatz type
    ansatz_type = "hermite"  # Options: "polynomial", "neural_network", "analytic", "hermite"
    
    # Choose your integrator for counterdiabatic simulations
    integrator_type = "leapfrog"  # Options: "leapfrog", "implicit_midpoint"
    
    # Simulation parameters
    M = 4000  # Number of particles (reduced for testing)
    N_steps = 2  # Number of simulation steps (reduced for testing)
    delta_t = 2.0  # Time step (eps = delta_t for this algorithm)
    momentum_refresh_interval = 1  # Momentum refresh interval
    fit_every = 1  # Fit ansatz every N steps
    num_initial_iterations = 10000  # Initial optimization iterations (reduced for testing)
    num_iterations = 10000  # Optimization iterations per step (reduced for testing)
    learning_rate = 1e-4  # Learning rate for optimization
    equilibration_steps = 0  # Equilibration steps after each CD step (reduced for testing)
    ess_threshold = 0.5  # Effective sample size threshold for resampling
    
    # Adaptive step size settings (for CD simulations only)
    adaptive_step_size = False  # Set to True to enable adaptive delta_t = K/sqrt(Var[A])
    K = delta_t  # Constant for adaptive step size calculation
    
    # Simulation settings
    run_simulations = True  # Set to False to load from saved data instead of running simulations
    snapshot_every = 1  # Record snapshots every N steps
    
    # ===== SYSTEM SETUP =====
    # Get system from systems.py
    make_T, make_V, system_description, dim = get_system(system_name)
    print(f"Using system: {system_name}")
    print(f"Description: {system_description}")
    print(f"Dimension: {dim}")
    
    # Define lambda functions
    v = 1.0
    max_lam = 3.0
    lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
    dot_lam_fn = jax.grad(lam_fn)
    
    # ===== ANSATZ SETUP =====
    if ansatz_type == "polynomial":
        ansatz = PolynomialAnsatz(max_degree=5, dim=dim)
    elif ansatz_type == "neural_network":
        key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
        dims = [2*dim, 32, 32, 1] 
        ansatz = NeuralNetworkAnsatz(dims=dims, key=key, dim=dim)
    elif ansatz_type == "analytic":
        ansatz = AnalyticAnsatz()
    elif ansatz_type == "hermite":
        # Hermite ansatz: A(q,p) = f(q) * g(p) where g(p) = Σ_{i odd} α̃ᵢ φᵢ(p)
        # Uses orthonormal Hermite polynomials φᵢ = Hᵢ / √(i!) with only odd indices
        # Use a polynomial ansatz for f(q) (position-only) with degree 0 (constant)
        f_ansatz = PolynomialFAnsatz(max_degree=0, dim=dim)
        # Set the constant term to 1 (f(q) = 1)
        f_ansatz = eqx.tree_at(lambda m: m.params, f_ansatz, f_ansatz.params.at[0].set(1.0))
        # print(f_ansatz(jnp.array([1.0])), f_ansatz(jnp.array([2.0])))
        # raise Exception("Stop here")
        ansatz = HermiteAnsatz(
            f_ansatz=f_ansatz,  # Parameterized ansatz for f(q)
            max_order=5,  # Use Hermite polynomials up to order 5 (odd indices: 1, 3, 5)
            dim=dim
        )
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")
    
    # ===== RUN SIMULATIONS OR LOAD DATA =====
    # Set run_simulations=False above to load existing data instead of running new simulations
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
        momentum_refresh_interval=momentum_refresh_interval,
        fit_every=fit_every,
        num_initial_iterations=num_initial_iterations,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        equilibration_steps=equilibration_steps,
        ess_threshold=ess_threshold,
        adaptive_step_size=adaptive_step_size,
        K=K,
        integrator_type=integrator_type
    )
    
    # ===== CREATE PLOTS =====
    if len(successful_simulations) > 0:
        if dim == 1:
            # Use the existing 1D plotting functions
            create_all_plots(
                successful_simulations=successful_simulations,
                system_name=system_name,
                ansatz=ansatz,
                delta_t=delta_t,
                make_V=make_V,
                make_T=make_T,
                dim=dim,
                integrator_type=integrator_type
            )
        else:
            # For 2D systems, use the 2D plotting function from plotting.py
            print(f"\nCreating 2D plots for {dim}D system...")
            create_2d_plots(successful_simulations, system_name, ansatz, delta_t, make_V, make_T, integrator_type)
    else:
        print("No successful simulations to plot.")

if __name__ == "__main__":
    main()
