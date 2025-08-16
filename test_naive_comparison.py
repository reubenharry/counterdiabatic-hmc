import jax
import jax.numpy as jnp
import numpy as np
from src.simulation import run_simulation, run_naive_hmc_simulation
from src.ansatze import PolynomialAnsatz
from src.systems import get_system

# Set up the system
system_name = "gaussian_moving_mean"
make_T, make_V, system_description, dim = get_system(system_name)

# Define lambda functions
v = 0.5
max_lam = 1.0
lam_fn = lambda t: jnp.where(v*t < max_lam, v * t, max_lam)
dot_lam_fn = jax.grad(lam_fn)

# Parameters
M = 100  # Smaller number for faster testing
N_steps = 10  # Fewer steps for faster testing
delta_t = 0.05
eps = 0.05
momentum_refresh_interval = 1/eps
fit_every = 10
num_initial_iterations = 100
num_iterations = 100
learning_rate = 1e-4
re_equil_steps = 0

# Create ansatz
ansatz = PolynomialAnsatz(max_degree=2, dim=dim)

# Use the same random key for both simulations
key = jax.random.PRNGKey(42)

print("Testing naive HMC implementations...")

# Test 1: Original run_simulation (naive HMC without weights)
print("\n1. Running original run_simulation (naive HMC without weights)")
key1 = jax.random.PRNGKey(42)
try:
    _, snapshots_original, _, _ = run_simulation(
        M=M, N_steps=N_steps, delta_t=delta_t, eps=eps,
        momentum_refresh_interval=momentum_refresh_interval,
        fit_every=fit_every, num_initial_iterations=num_initial_iterations,
        num_iterations=num_iterations, make_T=make_T, make_V=make_V,
        A_ansatz=ansatz, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
        key=key1, dim=dim, learning_rate=learning_rate,
        re_equil_steps=0, use_weights=False
    )
    print("✓ Original simulation completed")
except Exception as e:
    print(f"❌ Original simulation failed: {e}")
    snapshots_original = None

# Test 2: New run_naive_hmc_simulation
print("\n2. Running new run_naive_hmc_simulation")
key2 = jax.random.PRNGKey(42)
try:
    snapshots_new = run_naive_hmc_simulation(
        M=M, N_steps=N_steps, delta_t=delta_t, eps=eps,
        momentum_refresh_interval=momentum_refresh_interval,
        make_T=make_T, make_V=make_V, lam_fn=lam_fn, dot_lam_fn=dot_lam_fn,
        key=key2, dim=dim, use_weights=False
    )
    print("✓ New simulation completed")
except Exception as e:
    print(f"❌ New simulation failed: {e}")
    snapshots_new = None

# Compare results
if snapshots_original is not None and snapshots_new is not None:
    print("\n3. Comparing results...")
    
    # Compare naive snapshots
    if len(snapshots_original['naive']) > 0 and len(snapshots_new['naive']) > 0:
        original_naive = np.array(snapshots_original['naive'])
        new_naive = np.array(snapshots_new['naive'])
        
        print(f"Original naive shape: {original_naive.shape}")
        print(f"New naive shape: {new_naive.shape}")
        
        if original_naive.shape == new_naive.shape:
            diff = np.abs(original_naive - new_naive)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"Maximum difference: {max_diff}")
            print(f"Mean difference: {mean_diff}")
            
            if max_diff < 1e-10:
                print("✓ Results are identical (within numerical precision)")
            else:
                print("⚠️ Results are different!")
                
                # Show some sample differences
                print("\nSample differences (first few particles, first few timesteps):")
                for t in range(min(3, len(original_naive))):
                    for i in range(min(3, M)):
                        orig_val = original_naive[t, i, 0]
                        new_val = new_naive[t, i, 0]
                        print(f"  t={t}, particle={i}: original={orig_val:.6f}, new={new_val:.6f}, diff={abs(orig_val-new_val):.6f}")
        else:
            print("⚠️ Shapes don't match!")
    else:
        print("⚠️ No naive snapshots to compare")
else:
    print("⚠️ Cannot compare - one or both simulations failed")
