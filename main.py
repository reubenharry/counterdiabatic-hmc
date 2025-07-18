import jax
import jax.numpy as jnp

from src.simulation import run_simulation
from src.plotting import plot_results
from src.ansatze import PolynomialAnsatz, NeuralNetworkAnsatz, AnalyticAnsatz
from src.systems import get_system

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
    re_equil_steps = 5  # Number of naive HMC steps after each CD step
    
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
    ansatz = PolynomialAnsatz(max_degree=4, dim=dim)  # Reduced from 5 to 3 for better performance
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
            re_equil_steps=re_equil_steps,
        )
        
        simulation_time = time.time() - start_time
        print(f"Simulation completed in {simulation_time:.2f} seconds")
        
        # Check if we have enough data to plot
        if len(snapshots['cd_pre_equil']) > 0 and len(snapshots['naive']) > 0:
            plot_start = time.time()
            plot_results(snapshots, loss_histories, delta_t, make_V, lam_fn, param_history, A_ansatz, system_name, dim, plot_ansatz=False)
            plot_time = time.time() - plot_start
            print(f"Plotting completed in {plot_time:.2f} seconds")
        else:
            print("⚠️  Not enough data to plot - simulation may have failed early")
            
    except Exception as e:
        print(f"❌ Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 