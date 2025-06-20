import jax
import jax.numpy as jnp

from src.physics import m
from src.simulation import run_simulation
from src.plotting import plot_results
from src.ansatze import PolynomialAnsatz, NeuralNetworkAnsatz, AnalyticAnsatz

def main():
    # Define all routines and parameters here
    M = 3000
    N_steps = 20
    eps = 0.05
    delta_t = eps # should this even be a parameter?
    momentum_refresh_interval = 20
    fit_every = 10  # Fit the gauge potential every N steps
    v = 0.5
    max_lam = 1.0
    lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
    dot_lam_fn = jax.grad(lam_fn)
    def make_T(lam):
        return lambda p: 0.5 * (p ** 2) / m
    def make_V(lam):
        # return lambda q: (1-lam)*0.5*(q**2) + lam*(q**2 - 3)**2
        # return lambda q: 0.5 * (q - lam) ** 2
        return lambda q: 0.5 * (lam + 0.1) * (q ** 2)
    
    # Manually specify the potential name
    potential_name = "harmonic_annealing"
    
    # Initialize ansatz (either neural network or polynomial)
    key = jax.random.PRNGKey(0)
    # For neural network:
    d = 1
    ansatz = NeuralNetworkAnsatz([2*d, 128, 256, 128, d], key)
    # ansatz = NeuralNetworkAnsatz([2*d, 16, d], key)
    # For polynomial:
    # ansatz = PolynomialAnsatz(max_degree=4)
    # For analytic solution:
    # ansatz = AnalyticAnsatz(sigma=1.0)
    
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
    plot_results(snapshots, loss_histories, delta_t, make_V, lam_fn, param_history, A_ansatz, potential_name)

if __name__ == '__main__':
    main() 