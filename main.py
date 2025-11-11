#!/usr/bin/env python3
"""
Main script for running counterdiabatic simulations.
"""

import itertools
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
from src.blackjax_smc import smc_adjusted_hmc
from src.simulation import counterdiabatic_smc
from src.ansatze import PolynomialAnsatz, NeuralNetworkAnsatz, AnalyticAnsatz, HermiteAnsatz, PolynomialFAnsatz
from src.systems import SYSTEMS, get_system
from src.plotting import create_all_plots
from src.utils import save_simulation_data


def main():
    
    # ===== CONFIGURATION =====
    # Choose your system
    # system_name = "geometric_two_target"  # Options: see SYSTEMS in src/
    system_name = "mixture"  # Options: see SYSTEMS in src/
    # ansatz_type = 'polynomial' # Options: "polynomial", "neural_network", "analytic", "hermite"
    integrator_type = "leapfrog"  # Options: "leapfrog", "implicit_midpoint"
    ansatze = ['neural_network']
    weightings = [False]
    
    # Simulation parameters
    M = 4000  # Number of particles (reduced for testing)
    N_steps = 10  # Number of simulation steps (reduced for testing)
    # delta_t = 0.2  # Time step (eps = delta_t for this algorithm)
    final_time = 1.0
    momentum_refresh_interval = 100 # Momentum refresh interval
    fit_every = 1  # Fit ansatz every N steps
    num_iters = 100000  # Optimization iterations per step (reduced for testing)
    learning_rate = 1e-3  # Learning rate for optimization
    ess_threshold = None  # Effective sample size threshold for resampling
        
    # Simulation settings
    snapshot_every = 1  # Record snapshots every N steps
    
    # ===== SYSTEM SETUP =====
    # Get system from systems.py
    make_T, make_V, system_description, dim, initial_sigma = get_system(system_name)
    print(f"Using system: {system_name}")
    print(f"Description: {system_description}")
    print(f"Dimension: {dim}")
    
    # Define lambda functions
    v = 1.0
    max_lam = 1.0

    # 1D lam_fn
    lam_fn = lambda t: jnp.array([t]) # jnp.array([jnp.clip(t, a_min=0.0, a_max=max_lam)])

    # def lam_fn(t):
    #     """Schedule moving along the diagonal from [0, 0] to [1, 1] in the upper half-plane."""
    #     s = jnp.clip(v * t, a_min=0.0, a_max=max_lam)
    #     return jnp.stack([s, 2*s])

    lam_dim = lam_fn(0.0).shape[0]
    dot_lam_fn = jax.jacobian(lam_fn)

    # print(dot_lam_fn(jnp.array(1.0)), "dot_lam_fn(0.0)")
    # raise Exception()
    
    # ===== ANSATZ SETUP =====
    # Hermite ansatz: A(q,p) = f(q) * g(p) where g(p) = Σ_{i odd} α̃ᵢ φᵢ(p)
    f_ansatz = PolynomialFAnsatz(max_degree=5, dim=dim)
    # print(f_ansatz(jnp.array([1.0])), f_ansatz(jnp.array([2.0])))
    # Set the constant term to 1 (f(q) = 1)
    f_ansatz = eqx.tree_at(lambda m: m.params, f_ansatz, f_ansatz.params.at[0].set(1.0))
    hermite_ansatz = HermiteAnsatz(
        f_ansatz=f_ansatz,  # Parameterized ansatz for f(q)
        max_order=5,  # Use Hermite polynomials up to order 5 (odd indices: 1, 3, 5)
        dim=dim,
        output_dim=lam_dim
    )

    ansatz_dict = {
        "polynomial": PolynomialAnsatz(max_degree=5, dim=dim, output_dim=lam_dim),
        "neural_network": NeuralNetworkAnsatz(dims=[2*dim, 32, 32, lam_dim], key=jax.random.PRNGKey(42), dim=dim),
        "analytic": AnalyticAnsatz(),
        "hermite": hermite_ansatz
    }

    # _, plain_smc_snapshots = smc_adjusted_hmc(4000, SYSTEMS[system_name]['make_V'], jax.random.PRNGKey(0), threshold=0.5)
    # save_simulation_data(plain_smc_snapshots, system_name, 'smc_adjusted_hmc')
    # schedule = [t for t in plain_smc_snapshots['times']]
    # def next_time(t, k):
    #     if k < len(schedule) - 1:
    #         return schedule[k+1]
    #     else:
    #         return 1000.0

    for ansatz_type, use_weights in itertools.product(ansatze, weightings):
        A_ansatz = ansatz_dict.get(ansatz_type)
        
        A_ansatz, snapshots, param_history, state = counterdiabatic_smc(
            final_time=final_time,
            M=M, 
            N_steps=N_steps, 
            momentum_refresh_interval=momentum_refresh_interval, 
            make_T=make_T, 
            make_V=make_V, 
            lam_fn=lam_fn, 
            dot_lam_fn=dot_lam_fn, 
            key=jax.random.PRNGKey(0), 
            dim=dim, 
            A_ansatz=A_ansatz, 
            fit_every=fit_every, 
            num_iters=num_iters, 
            learning_rate=learning_rate, 
            ess_threshold=ess_threshold, 
            snapshot_every=snapshot_every, 
            use_weights=use_weights,
            integrator_type=integrator_type,
            initial_sigma=initial_sigma,
            # next_time = next_time
            next_time=lambda t, k: t + 0.01,
            )

        # save simulation data
        ansatz_params = A_ansatz.get_params_for_saving() if A_ansatz is not None else None
        
        save_simulation_data(
            snapshots, 
            system_name, 
            f'{"cd" if A_ansatz is not None else "naive"}_{"weighted" if use_weights else "unweighted"}', 
            ansatz_params, 
            param_history,
            ansatz_type="naive" if A_ansatz is None else ansatz_type,
            integrator_type=integrator_type
        )
       

if __name__ == "__main__":
    main()
