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
M = 50  # Small number for debugging
N_steps = 10  # More steps to see evolution
delta_t = 0.05
eps = 0.05
momentum_refresh_interval = 1/eps
fit_every = 1  # Fit every step to match the original behavior
num_initial_iterations = 100
num_iterations = 100
learning_rate = 1e-4
re_equil_steps = 0

# Create ansatz
ansatz = PolynomialAnsatz(max_degree=2, dim=dim)

print("Debugging naive HMC implementations...")

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
    
    # Print some sample trajectories
    if len(snapshots_original['naive']) > 0:
        print("\nOriginal naive HMC trajectories (first 3 particles):")
        for t, snapshot in enumerate(snapshots_original['naive']):
            lam_val = float(lam_fn(t * delta_t))
            print(f"  t={t}, λ={lam_val:.3f}: particles[0:3] = {snapshot[0:3, 0]}")
            
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
    
    # Print some sample trajectories
    if len(snapshots_new['naive']) > 0:
        print("\nNew naive HMC trajectories (first 3 particles):")
        for t, snapshot in enumerate(snapshots_new['naive']):
            lam_val = float(lam_fn(t * delta_t))
            print(f"  t={t}, λ={lam_val:.3f}: particles[0:3] = {snapshot[0:3, 0]}")
            
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
            
            # Show the evolution of the mean position
            print("\nEvolution of mean position:")
            for t in range(len(original_naive)):
                lam_val = float(lam_fn(t * delta_t))
                orig_mean = np.mean(original_naive[t, :, 0])
                new_mean = np.mean(new_naive[t, :, 0])
                print(f"  t={t}, λ={lam_val:.3f}: original_mean={orig_mean:.3f}, new_mean={new_mean:.3f}, diff={abs(orig_mean-new_mean):.3f}")
        else:
            print("⚠️ Shapes don't match!")
    else:
        print("⚠️ No naive snapshots to compare")
else:
    print("⚠️ Cannot compare - one or both simulations failed")
