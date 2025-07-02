import jax
import jax.numpy as jnp

from src.physics import m
from src.simulation import run_simulation
from src.plotting import plot_results
from src.ansatze import PolynomialAnsatz, NeuralNetworkAnsatz, AnalyticAnsatz
from src.systems import get_system

def main():
    # Define all routines and parameters here
    M = 512
    N_steps = 500
    eps = 0.01
    delta_t = eps # should this even be a parameter?
    momentum_refresh_interval = 5
    fit_every = 25  # Fit the gauge potential every N steps
    num_initial_iterations = 10000  # Number of iterations for first fitting
    num_iterations = 10000  # Number of iterations for subsequent fittings
    v = 0.5
    max_lam = 1.0
    lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
    dot_lam_fn = jax.grad(lam_fn)
    learning_rate = 1e-4
    
    # Choose the system to simulate
    # system_name = "2d_gaussian_moving_mean"  # Try the 2D system
    # system_name = "2d_gaussian_annealing"  # Try the 2D system
    # system_name = "gaussian_moving_mean"
    # system_name = "gaussian_annealing"
    # system_name = "double_well"
    system_name = "2d_normal_to_rosenbrock"
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
    ansatz = PolynomialAnsatz(max_degree=5, dim=dim)  # Reduced degree for 2D to avoid too many terms
    # For analytic solution:
    # ansatz = AnalyticAnsatz()
    
    # Print polynomial terms if using polynomial ansatz
    if isinstance(ansatz, PolynomialAnsatz):
        print("Polynomial terms:")
        for desc in ansatz.get_term_description():
            print(f"  {desc}")
        print(f"Total number of parameters: {len(ansatz.params)}")
    
    try:
        A_ansatz, snapshots, loss_histories, param_history = run_simulation(
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
            key=key,
            dim=dim,
            learning_rate=learning_rate,
            
        )
        
        # Check if we have enough data to plot
        if len(snapshots['cd']) > 0 and len(snapshots['naive']) > 0:
            plot_results(snapshots, loss_histories, delta_t, make_V, lam_fn, param_history, A_ansatz, system_name, dim)
        else:
            print("⚠️  Not enough data to plot - simulation may have failed early")
            
    except Exception as e:
        print(f"❌ Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 