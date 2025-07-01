import jax
import jax.numpy as jnp

from src.physics import m
from src.simulation import run_simulation
from src.plotting import plot_results
from src.ansatze import PolynomialAnsatz, NeuralNetworkAnsatz, AnalyticAnsatz
from src.systems import get_system

def main():
    # Define all routines and parameters here
    M = 3000
    N_steps = 200
    eps = 0.01
    delta_t = eps # should this even be a parameter?
    momentum_refresh_interval = 5
    fit_every = 10  # Fit the gauge potential every N steps
    v = 0.5
    max_lam = 1.0
    lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
    dot_lam_fn = jax.grad(lam_fn)
    
    # Choose the system to simulate
    system_name = "gaussian_moving_mean"
    # system_name = "gaussian_annealing"
    # system_name = "double_well"
    make_T, make_V, system_description = get_system(system_name)
    print(f"Using system: {system_name}")
    print(f"Description: {system_description}")
    
    # Initialize ansatz (either neural network or polynomial)
    key = jax.random.PRNGKey(0)
    # For neural network:
    # d = 1
    # ansatz = NeuralNetworkAnsatz([2*d, 128, 256, 128, d], key)
    # ansatz = NeuralNetworkAnsatz([2*d, 16, d], key)
    # For polynomial:
    ansatz = PolynomialAnsatz(max_degree=5)
    # For analytic solution:
    # ansatz = AnalyticAnsatz()
    
    # Print polynomial terms if using polynomial ansatz
    if isinstance(ansatz, PolynomialAnsatz):
        print("Polynomial terms:")
        for desc in ansatz.get_term_description():
            print(f"  {desc}")
        print(f"Total number of parameters: {len(ansatz.params)}")
    
    A_ansatz, snapshots, loss_histories, param_history = run_simulation(
        M=M, 
        N_steps=N_steps, 
        delta_t=delta_t, 
        eps=eps, 
        momentum_refresh_interval=momentum_refresh_interval,
        fit_every=fit_every,
        make_T=make_T, 
        make_V=make_V, 
        lam_fn=lam_fn, 
        dot_lam_fn=dot_lam_fn, 
        A_ansatz=ansatz, 
        key=key
    )
    plot_results(snapshots, loss_histories, delta_t, make_V, lam_fn, param_history, A_ansatz, system_name)

if __name__ == '__main__':
    main() 