import jax
import jax.numpy as jnp
import numpy as np
from src.simulation import generate_initial_samples
from src.physics import with_maruyama, make_leapfrog_step
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
M = 50
N_steps = 10
delta_t = 0.05
eps = 0.05
momentum_refresh_interval = 1/eps

print("Testing exact naive HMC logic...")

# Use the same random key
key = jax.random.PRNGKey(42)

# Generate initial samples
initial_lam = float(lam_fn(0.0))
print(f"Generating initial samples with λ = {initial_lam}")
q_naive, p_naive = generate_initial_samples(M, make_T, make_V, initial_lam, key, dim)

print(f"Initial particles[0:3] = {q_naive[0:3, 0]}")

# Run naive HMC steps exactly as in the original implementation
for k in range(N_steps):
    t_k = k * delta_t
    lam_k = float(lam_fn(t_k))
    lam_k1 = float(lam_fn(t_k + delta_t))
    
    print(f"\nStep {k}: t={t_k:.3f}, λ_k={lam_k:.3f}, λ_k1={lam_k1:.3f}")
    
    # Split random key exactly as in original
    key, sub = jax.random.split(key)
    subs = jax.random.split(sub, M)
    
    # Create naive step exactly as in original
    naive_step = jax.vmap(lambda q, p, lam, lam_next, eps, L, rng_key: with_maruyama(make_leapfrog_step(make_T(lam), make_V(lam)))(q,p,eps,L=L, rng_key=rng_key), in_axes=(0, 0, None, None, None, None, 0))
    
    # Execute naive step exactly as in original
    q_naive, p_naive = jax.jit(naive_step)(q_naive, p_naive, lam_k, lam_k1, eps, eps*momentum_refresh_interval, subs)
    
    print(f"  After step: particles[0:3] = {q_naive[0:3, 0]}")
    print(f"  Mean position = {np.mean(q_naive[:, 0]):.3f}")

print(f"\nFinal mean position = {np.mean(q_naive[:, 0]):.3f}")
print(f"Expected λ at final time = {lam_fn(N_steps * delta_t):.3f}")
