import jax
import jax.numpy as jnp
import numpy as np
from src.simulation import compute_counterdiabatic_work
from src.ansatze import PolynomialAnsatz
from src.systems import get_system

def test_counterdiabatic_work():
    """Test that counterdiabatic work computation produces non-zero values."""
    
    # Set up a simple system
    system_name = "gaussian_moving_mean"
    dim = 1
    make_T, make_V, lam_fn, dot_lam_fn = get_system(system_name, dim)
    
    # Create a simple polynomial ansatz
    ansatz = PolynomialAnsatz(dim=dim, degree=2)
    
    # Set some non-zero parameters
    ansatz.params = jnp.array([0.1, 0.2, 0.3])
    
    # Create some test samples
    M = 10
    q = jnp.random.normal(jax.random.PRNGKey(0), (M, dim))
    p = jnp.random.normal(jax.random.PRNGKey(1), (M, dim))
    
    # Test lambda values
    lam_k = 0.5
    lam_k1 = 0.6
    
    print("Testing counterdiabatic work computation...")
    print(f"q shape: {q.shape}, p shape: {p.shape}")
    print(f"ansatz params: {ansatz.params}")
    print(f"lam_k: {lam_k}, lam_k1: {lam_k1}")
    
    # Compute work
    work_vals = compute_counterdiabatic_work(q, p, lam_k, lam_k1, ansatz, make_T, make_V)
    
    print(f"Work values: {work_vals}")
    print(f"Work mean: {jnp.mean(work_vals)}")
    print(f"Work std: {jnp.std(work_vals)}")
    print(f"Work range: [{jnp.min(work_vals)}, {jnp.max(work_vals)}]")
    
    # Check if work values are non-zero
    if jnp.any(jnp.abs(work_vals) > 1e-10):
        print("✓ Counterdiabatic work computation is producing non-zero values!")
    else:
        print("✗ Counterdiabatic work computation is producing zero values - this indicates a problem.")
    
    # Test with different lambda values
    lam_k2 = 0.7
    work_vals2 = compute_counterdiabatic_work(q, p, lam_k, lam_k2, ansatz, make_T, make_V)
    print(f"\nWork values with larger lambda change: {work_vals2}")
    print(f"Work mean: {jnp.mean(work_vals2)}")
    
    # Test with different ansatz parameters
    ansatz2 = PolynomialAnsatz(dim=dim, degree=2)
    ansatz2.params = jnp.array([1.0, 0.5, 0.1])
    work_vals3 = compute_counterdiabatic_work(q, p, lam_k, lam_k1, ansatz2, make_T, make_V)
    print(f"\nWork values with different ansatz: {work_vals3}")
    print(f"Work mean: {jnp.mean(work_vals3)}")

if __name__ == "__main__":
    test_counterdiabatic_work()
