import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt



def sample_2d_gaussian_hmc_self_contained():
    """Self-contained HMC sampler for 2D isotropic Gaussian distribution."""
    
    # Parameters
    M = 1  # Number of particles
    num_equilibration_steps = 0  # Number of HMC steps for equilibration
    num_sampling_steps = 10000  # Number of HMC steps for sampling
    eps = 1.3  # Step size for HMC
    L = 0.4
    mass = 1.0  # Mass parameter

    def partially_refresh_momentum(momentum, rng_key, step_size, L):

        # return momentum

        c1 = jnp.exp(-step_size/L)
        c2 = jnp.sqrt((1-c1**2))
        z = jax.random.normal(rng_key, shape=momentum.shape, dtype=momentum.dtype)
        new_momentum = c1*momentum + c2*z

        return jax.lax.cond(
            jnp.isinf(L),
            lambda _: momentum,
            lambda _: new_momentum,
            operand=None,
        )
    
    # 2D Isotropic Gaussian potential: V(q) = 0.5 * ||q||²
    def potential_energy(q):
        """2D isotropic Gaussian potential energy."""
        return 0.5 * jnp.sum(q**2)
    
    # Kinetic energy: T(p) = ||p||² / (2m)
    def kinetic_energy(p):
        """Kinetic energy."""
        return 0.5 * jnp.sum(p**2) / mass
    
    # Gradients
    def grad_potential(q):
        """Gradient of potential energy with respect to q."""
        return q
    
    def grad_kinetic(p):
        """Gradient of kinetic energy with respect to p."""
        return p / mass
    
    # Leapfrog integrator
    @jax.jit
    def leapfrog_step(q, p, eps, L, rng_key):
        """Single leapfrog step."""
        # Half step in momentum
        
        key1, key2 = jax.random.split(rng_key)

        p = partially_refresh_momentum(p, key1, eps/2, L)

        p_half = p - 0.5 * eps * grad_potential(q)
        
        # Full step in position
        q_new = q + eps * grad_kinetic(p_half)
        
        # Half step in momentum
        p_new = p_half - 0.5 * eps * grad_potential(q_new)

        p_new = partially_refresh_momentum(p_new, key2, eps/2, L)
        
        return q_new, p_new
    
    # Initialize random key
    key = jax.random.PRNGKey(42)
    
    # Start from random positions and momenta
    key, sub = jax.random.split(key)
    q = jax.random.normal(sub, (M, 2))  # 2D positions
    key, sub = jax.random.split(key)
    p = jax.random.normal(sub, (M, 2))  # 2D momenta
    
    print(f"Starting HMC sampling with {M} particles...")
    print(f"Equilibrating for {num_equilibration_steps} steps with step size {eps}...")
    
    # Run HMC for equilibration
    for step_idx in range(num_equilibration_steps):
        key, sub = jax.random.split(key)
        print(step_idx)
        q, p = jax.vmap(lambda q, p: leapfrog_step(q, p, eps, L, sub))(q, p)
        
        # Check for NaNs
        if jnp.isnan(q).any() or jnp.isnan(p).any():
            print(f"⚠️  NaNs detected during equilibration at step {step_idx}")
            break
        
        # Randomize momenta periodically for ergodicity
        if step_idx % 50 == 0:
            key, sub = jax.random.split(key)
            p = jax.random.normal(sub, (M, 2))
    
    print("Equilibration complete. Collecting samples...")
    
    # Collect samples
    samples = []
    for step_idx in range(num_sampling_steps):
        key, sub = jax.random.split(key)
        q, p = jax.vmap(lambda q, p: leapfrog_step(q, p, eps, L, sub))(q, p)
        print(f"q: {q}")
        
        # Check for NaNs
        if jnp.isnan(q).any() or jnp.isnan(p).any():
            print(f"⚠️  NaNs detected during sampling at step {step_idx}")
            break
            
        samples.append(np.array(q))
        
        # Randomize momenta periodically
        # if step_idx % 1 == 0:
        #     key, sub = jax.random.split(key)
        #     p = jax.random.normal(sub, (M, 2))
    
    # Convert to numpy array
    samples = np.array(samples)  # Shape: (num_sampling_steps, M, 2)
    print(f"Collected {samples.shape[0]} samples with {samples.shape[1]} particles each")
    
    # Flatten samples for plotting
    all_samples = samples.reshape(-1, 2)  # Shape: (num_sampling_steps * M, 2)
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
    
    # Add contour plot of the potential
    q0_range = np.linspace(-3, 3, 100)
    q1_range = np.linspace(-3, 3, 100)
    Q0, Q1 = np.meshgrid(q0_range, q1_range)
    
    V_values = np.zeros_like(Q0)
    for i in range(Q0.shape[0]):
        for j in range(Q0.shape[1]):
            q = jnp.array([Q0[i, j], Q1[i, j]])
            V_values[i, j] = float(potential_energy(q))
    
    # Plot contours
    contours = ax.contour(Q0, Q1, V_values, levels=10, colors='black', alpha=0.5, linewidths=1)
    ax.clabel(contours, inline=True, fontsize=8)
    
    # Add the true mean at (0, 0)
    ax.scatter([0.0], [0.0], color='red', s=100, marker='*', label='True mean (0, 0)', zorder=5)
    
    ax.set_xlabel('q₀')
    ax.set_ylabel('q₁')
    ax.set_title('HMC Samples from 2D Isotropic Gaussian Distribution\nV(q) = 0.5 * ||q||²')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Create figures directory
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('figures/naive_hmc_2d_gaussian_self_contained.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'figures/naive_hmc_2d_gaussian_self_contained.png'")
    
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
    
    # Theoretical values for comparison
    theoretical_std = 1.0  # σ = 1 for V(q) = 0.5 * ||q||²
    print(f"\nTheoretical standard deviation: {theoretical_std:.4f}")
    print(f"Empirical std q₀: {np.std(valid_samples[:, 0]):.4f} (error: {abs(np.std(valid_samples[:, 0]) - theoretical_std):.4f})")
    print(f"Empirical std q₁: {np.std(valid_samples[:, 1]):.4f} (error: {abs(np.std(valid_samples[:, 1]) - theoretical_std):.4f})")
    
    # Calculate correlation
    correlation = np.corrcoef(valid_samples[:, 0], valid_samples[:, 1])[0, 1]
    print(f"Correlation between q₀ and q₁: {correlation:.4f} (should be close to 0 for isotropic Gaussian)")
    
    # plt.show()

if __name__ == "__main__":
    sample_2d_gaussian_hmc_self_contained() 