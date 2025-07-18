import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from src.systems import get_system
from src.physics import make_leapfrog_step

def sample_rosenbrock_hmc():
    """Use HMC to sample from the Rosenbrock potential at λ=1."""
    
    # Get the Rosenbrock system
    make_T, make_V, system_description, dim = get_system("2d_normal_to_rosenbrock")
    print(f"System: {system_description}")
    
    # Parameters - use smaller step size for stability
    M = 1000  # Number of samples
    num_steps = 1000  # Number of HMC steps for equilibration
    eps = 0.001  # Smaller step size for stability
    lam = 1.0  # Full Rosenbrock potential
    
    # Create the potential and kinetic energy functions
    T = make_T(lam)
    V = make_V(lam)
    
    # Create HMC step function
    step = make_leapfrog_step(T, V)
    
    # Initialize random key
    key = jax.random.PRNGKey(42)
    
    # Start from random positions
    key, sub = jax.random.split(key)
    q = jax.random.normal(sub, (M, dim))
    key, sub = jax.random.split(key)
    p = jax.random.normal(sub, (M, dim)) * jnp.sqrt(1.0)  # m = 1.0
    
    print(f"Starting HMC sampling with {M} particles...")
    print(f"Equilibrating for {num_steps} steps with step size {eps}...")
    
    # Run HMC for equilibration
    for step_idx in range(num_steps):
        q, p = jax.vmap(lambda q, p: step(q, p, eps))(q, p)
        
        # Check for NaNs
        if jnp.isnan(q).any() or jnp.isnan(p).any():
            print(f"⚠️  NaNs detected during equilibration at step {step_idx}")
            break
        
        # Randomize momenta periodically
        if step_idx % 100 == 0:
            key, sub = jax.random.split(key)
            p = jax.random.normal(sub, (M, dim)) * jnp.sqrt(1.0)
    
    print("Equilibration complete. Collecting samples...")
    
    # Collect samples
    samples = []
    num_samples = 1000  # Reduced number of samples
    
    for step_idx in range(num_samples):
        q, p = jax.vmap(lambda q, p: step(q, p, eps))(q, p)
        
        # Check for NaNs
        if jnp.isnan(q).any() or jnp.isnan(p).any():
            print(f"⚠️  NaNs detected during sampling at step {step_idx}")
            break
            
        samples.append(np.array(q))
        
        # Randomize momenta periodically
        if step_idx % 100 == 0:
            key, sub = jax.random.split(key)
            p = jax.random.normal(sub, (M, dim)) * jnp.sqrt(1.0)
    
    # Convert to numpy array
    samples = np.array(samples)  # Shape: (num_samples, M, dim)
    print(f"Collected {samples.shape[0]} samples with {samples.shape[1]} particles each")
    
    # Flatten samples for plotting
    all_samples = samples.reshape(-1, dim)  # Shape: (num_samples * M, dim)
    print(f"Total sample points: {all_samples.shape[0]}")
    
    # Remove any NaN samples
    valid_samples = all_samples[~np.isnan(all_samples).any(axis=1)]
    print(f"Valid sample points (no NaNs): {valid_samples.shape[0]}")
    
    if valid_samples.shape[0] == 0:
        print("❌ No valid samples collected! HMC is too unstable.")
        return
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot samples
    ax.scatter(valid_samples[:, 0], valid_samples[:, 1], alpha=0.6, s=1, color='blue', label='HMC samples')
    
    # Add the true minimum of Rosenbrock function at (1, 1)
    ax.scatter([1.0], [1.0], color='red', s=100, marker='*', label='True minimum (1, 1)', zorder=5)
    
    # Add contour plot of the potential for reference
    q0_range = np.linspace(-2, 4, 100)
    q1_range = np.linspace(-2, 4, 100)
    Q0, Q1 = np.meshgrid(q0_range, q1_range)
    
    V_values = np.zeros_like(Q0)
    for i in range(Q0.shape[0]):
        for j in range(Q0.shape[1]):
            q = jnp.array([Q0[i, j], Q1[i, j]])
            V_values[i, j] = float(V(q))
    
    # Plot contours
    contours = ax.contour(Q0, Q1, V_values, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8)
    
    ax.set_xlabel('q₀')
    ax.set_ylabel('q₁')
    ax.set_title(f'HMC Samples from Rosenbrock Potential (λ = {lam})')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Create figures directory
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('figures/rosenbrock_hmc_samples.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'figures/rosenbrock_hmc_samples.png'")
    
    # Print some statistics
    print(f"\nSample statistics:")
    print(f"Mean q₀: {np.mean(valid_samples[:, 0]):.4f}")
    print(f"Mean q₁: {np.mean(valid_samples[:, 1]):.4f}")
    print(f"Std q₀: {np.std(valid_samples[:, 0]):.4f}")
    print(f"Std q₁: {np.std(valid_samples[:, 1]):.4f}")
    print(f"Min q₀: {np.min(valid_samples[:, 0]):.4f}")
    print(f"Max q₀: {np.max(valid_samples[:, 0]):.4f}")
    print(f"Min q₁: {np.min(valid_samples[:, 1]):.4f}")
    print(f"Max q₁: {np.max(valid_samples[:, 1]):.4f}")

if __name__ == '__main__':
    sample_rosenbrock_hmc() 