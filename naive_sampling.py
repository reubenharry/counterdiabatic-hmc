import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import generate_initial_samples
from src.systems import get_system
from src.physics import make_leapfrog_step, with_maruyama

def sample_1d_gaussian_hmc():
    """Use naive HMC to sample from a 1D Gaussian distribution."""
    
    # Get the 1D Gaussian system (we'll use the annealing version with λ=1 for a standard Gaussian)
    make_T, make_V, system_description, dim = get_system("double_well")
    print(f"System: {system_description}")
    print(f"Dimension: {dim}")
    
    # Parameters
    M = 1000  # Number of samples
    num_equilibration_steps = 0  # Number of HMC steps for equilibration
    num_sampling_steps = 1000  # Number of HMC steps for sampling
    eps = 0.1  # Step size for HMC
    L = 10.0
    lam = 0.5  # Full Gaussian potential (λ=1 gives V(q) = 0.5 * ||q||²)
    
    # Create the potential and kinetic energy functions
    T = make_T(lam)
    V = make_V(lam)
    
    # Create HMC step function
    step = with_maruyama(make_leapfrog_step(T, V))
    step = jax.jit(step)
    
    # Initialize random key
    key = jax.random.PRNGKey(42)
    
    # Start from random positions
    key, sub = jax.random.split(key)
    q = jax.random.normal(sub, (M, dim))
    key, sub = jax.random.split(key)
    p = jax.random.normal(sub, (M, dim))
    
    print(f"Starting HMC sampling with {M} particles...")
    print(f"Equilibrating for {num_equilibration_steps} steps with step size {eps}...")
    
    # Equilibration phase
    for step_idx in range(num_equilibration_steps):
        key, sub = jax.random.split(key)
        q, p = jax.vmap(lambda q, p: step(q, p, eps, L, sub))(q, p)
        
        # Check for NaNs during equilibration
        if jnp.isnan(q).any() or jnp.isnan(p).any():
            print(f"⚠️  NaNs detected during equilibration at step {step_idx}")
            break
            
        # Randomize momenta periodically during equilibration
        # if step_idx % 20 == 0:
        #     key, sub = jax.random.split(key)
        #     p = jax.random.normal(sub, (M, dim))
    
    print(f"Equilibration completed. Starting sampling phase...")
    
    # Collect samples
    samples = []
    q, p = generate_initial_samples(M, make_T, make_V, lam, key, dim, num_steps=1000, eps=0.1, L=10.0)
    # for step_idx in range(num_sampling_steps):
    #     key, sub = jax.random.split(key)
    #     q, p = jax.vmap(lambda q, p: step(q, p, eps, L, sub))(q, p)
        
    #     # Check for NaNs
    #     if jnp.isnan(q).any() or jnp.isnan(p).any():
    #         print(f"⚠️  NaNs detected during sampling at step {step_idx}")
    #         break
            
    #     samples.append(np.array(q))
        
        # Randomize momenta periodically
        # if step_idx % 100 == 0:
        #     key, sub = jax.random.split(key)
        #     p = jax.random.normal(sub, (M, dim))
    
    # Convert to numpy array
    samples = q  # Shape: (num_sampling_steps, M, dim)
    print(samples.shape)
    print(f"Collected {samples.shape[0]} samples with {samples.shape[1]} particles each")
    
    # Flatten samples for plotting
    all_samples = samples.reshape(-1, dim)  # Shape: (num_sampling_steps * M, dim)
    print(f"Total sample points: {all_samples.shape[0]}")
    
    # Remove any NaN samples
    valid_samples = all_samples[~np.isnan(all_samples).any(axis=1)]
    print(f"Valid sample points (no NaNs): {valid_samples.shape[0]}")
    
    if valid_samples.shape[0] == 0:
        print("❌ No valid samples collected! HMC is too unstable.")
        return
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Determine plot range based on data
    x_min = np.min(valid_samples) - 0.5
    x_max = np.max(valid_samples) + 0.5
    
    # Create histogram of samples
    hist, bin_edges = np.histogram(valid_samples, bins=50, density=True, range=(x_min, x_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot histogram
    ax.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], 
           color='blue', alpha=0.6, label='HMC samples')
    
    # Compute exact result via quadrature
    xs = np.linspace(x_min, x_max, 400)
    potential_fn = make_V(lam)
    rho = np.array(jax.vmap(lambda x: jnp.exp(-potential_fn(x)))(xs))
    rho /= np.trapezoid(rho, xs)  # Normalize to integrate to 1
    
    # Plot exact result
    ax.plot(xs, rho, 'r-', linewidth=2, label='Exact distribution')
    
    ax.set_xlabel('q')
    ax.set_ylabel('Density')
    ax.set_title(f'Naive HMC Samples from 1D Gaussian Distribution\nλ = {lam}, V(q) = 0.5 * ||q||²')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Create figures directory
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('figures/naive_hmc_1d_gaussian.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'figures/naive_hmc_1d_gaussian.png'")
    
    # Print some statistics
    print(f"\nSample statistics:")
    print(f"Mean: {np.mean(valid_samples):.4f}")
    print(f"Std: {np.std(valid_samples):.4f}")
    print(f"Min: {np.min(valid_samples):.4f}")
    print(f"Max: {np.max(valid_samples):.4f}")
    
    # Theoretical values for comparison
    theoretical_mean = 0.0
    theoretical_std = 1.0  # σ = 1 for V(q) = 0.5 * ||q||²
    print(f"\nTheoretical values:")
    print(f"Mean: {theoretical_mean:.4f}")
    print(f"Std: {theoretical_std:.4f}")
    print(f"Empirical mean: {np.mean(valid_samples):.4f} (error: {abs(np.mean(valid_samples) - theoretical_mean):.4f})")
    print(f"Empirical std: {np.std(valid_samples):.4f} (error: {abs(np.std(valid_samples) - theoretical_std):.4f})")
    
    # Calculate KL divergence between empirical and theoretical distributions
    # We'll use a simple histogram-based KL divergence
    theoretical_hist, _ = np.histogram(np.random.normal(0, 1, 100000), bins=bin_edges, density=True)
    empirical_hist = hist
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    theoretical_hist += epsilon
    empirical_hist += epsilon
    
    # Normalize
    theoretical_hist /= np.sum(theoretical_hist)
    empirical_hist /= np.sum(empirical_hist)
    
    # Calculate KL divergence
    kl_divergence = np.sum(empirical_hist * np.log(empirical_hist / theoretical_hist))
    print(f"KL divergence (empirical || theoretical): {kl_divergence:.6f}")
    
    # plt.show()

if __name__ == "__main__":
    sample_1d_gaussian_hmc() 