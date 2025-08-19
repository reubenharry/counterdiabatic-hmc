#!/usr/bin/env python3
"""
Generate ground truth samples for benchmarking power-mixture systems.
Runs long HMC chains to estimate E[x^2] for the target distribution π1.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from src.systems import get_power_mixture_system
from src.physics import make_leapfrog_step, with_maruyama

def blackjax_sample(log_density):
    return jnp.zeros((1000, 1))


def generate_ground_truth_samples(
    target_name: str = "double_well",
    dim: int = 1,
    sigma0: float = 1.0,
    num_chains: int = 1000,
    num_steps: int = 10000,
    step_size: float = 0.1,
    L: float = 10.0,
    save_dir: str = "data/ground_truth"
):
    """
    Generate ground truth samples by running long HMC chains on the target distribution.
    
    Args:
        target_name: Name of the target potential (e.g., 'double_well')
        dim: Dimension of the system
        sigma0: Standard deviation of the base Gaussian π0
        num_chains: Number of independent HMC chains
        num_steps: Number of steps per chain
        step_size: HMC step size
        L: Momentum refresh time scale
        save_dir: Directory to save samples
    """
    
    # Get the system
    make_T, make_V, description, system_dim = get_power_mixture_system(
        target_name, dim=dim, sigma0=sigma0
    )
    
    print(f"Generating ground truth for: {description}")
    print(f"Running {num_chains} chains of {num_steps} steps each...")
    
    # We want samples from π1 (λ=1), so we'll run HMC on the target potential
    T = make_T(1.0)  # Kinetic energy
    V = make_V(1.0)  # Target potential (λ=1)
    
    # Create the integrator
    leapfrog = make_leapfrog_step(T, V, T, V)
    integrator = with_maruyama(leapfrog)
    
    # Storage for all samples
    all_samples = []
    all_x2_estimates = []
    
    # Run chains in batches to avoid memory issues
    batch_size = 100
    num_batches = (num_chains + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_chains)
        batch_size_actual = end_idx - start_idx
        
        print(f"Running batch {batch_idx + 1}/{num_batches} (chains {start_idx}-{end_idx-1})")
        
        # Initialize positions and momenta for this batch
        key = jax.random.PRNGKey(batch_idx)
        q_init = jax.random.normal(key, shape=(batch_size_actual, dim))
        p_init = jax.random.normal(key, shape=(batch_size_actual, dim))
        
        # Run HMC simulation for this batch
        q_current = q_init
        p_current = p_init
        
        # Burn-in phase
        burn_in_steps = num_steps // 2
        for step in range(burn_in_steps):
            key, subkey = jax.random.split(key)
            subs = jax.random.split(subkey, batch_size_actual)
            
            # Vectorized step for all chains in batch
            def single_step(q, p, rng_key):
                return integrator(q, p, step_size, L, rng_key)
            
            q_current, p_current, _ = jax.vmap(single_step)(q_current, p_current, subs)
        
        # Sampling phase
        samples_batch = []
        for step in range(burn_in_steps, num_steps):
            key, subkey = jax.random.split(key)
            subs = jax.random.split(subkey, batch_size_actual)
            
            # Vectorized step for all chains in batch
            def single_step(q, p, rng_key):
                return integrator(q, p, step_size, L, rng_key)
            
            q_current, p_current, _ = jax.vmap(single_step)(q_current, p_current, subs)
            
            # Store every 10th sample to reduce memory usage
            if step % 10 == 0:
                samples_batch.append(q_current)
        
        # Compute x^2 for each sample
        samples_batch = jnp.array(samples_batch)  # Shape: (num_samples, batch_size, dim)
        x2_values = jnp.sum(samples_batch ** 2, axis=-1)  # Sum over dimensions
        
        # Store samples and x^2 estimates
        all_samples.append(samples_batch)
        all_x2_estimates.append(x2_values)
    
    # Concatenate all batches
    all_samples = jnp.concatenate(all_samples, axis=1)  # Shape: (num_samples, total_chains, dim)
    all_x2_estimates = jnp.concatenate(all_x2_estimates, axis=1)  # Shape: (num_samples, total_chains)
    
    # Flatten to get all samples
    all_samples_flat = all_samples.reshape(-1, dim)  # Shape: (total_samples, dim)
    all_x2_estimates_flat = all_x2_estimates.reshape(-1)  # Shape: (total_samples,)
    
    # Compute ground truth statistics
    E_x2_ground_truth = jnp.mean(all_x2_estimates_flat)
    Var_x2_ground_truth = jnp.var(all_x2_estimates_flat)
    
    print(f"Ground truth statistics:")
    print(f"  E[x^2] = {E_x2_ground_truth:.6f}")
    print(f"  Var[x^2] = {Var_x2_ground_truth:.6f}")
    print(f"  Total samples: {len(all_samples_flat)}")
    
    # Create save directory
    save_path = Path(save_dir) / f"{target_name}_dim{dim}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save samples and statistics
    np.savez(
        save_path / "samples.npz",
        samples=all_samples_flat,
        x2_values=all_x2_estimates_flat,
        E_x2=E_x2_ground_truth,
        Var_x2=Var_x2_ground_truth,
        metadata={
            'target_name': target_name,
            'dim': dim,
            'sigma0': sigma0,
            'num_chains': num_chains,
            'num_steps': num_steps,
            'step_size': step_size,
            'L': L,
            'description': description
        }
    )
    
    print(f"Saved ground truth samples to: {save_path / 'samples.npz'}")
    return E_x2_ground_truth, Var_x2_ground_truth


if __name__ == "__main__":
    # Generate ground truth for 1D double well
    E_x2, Var_x2 = generate_ground_truth_samples(
        target_name="double_well",
        dim=1,
        sigma0=1.0,
        num_chains=1000,
        num_steps=10000,
        step_size=0.1,
        L=10.0
    )
