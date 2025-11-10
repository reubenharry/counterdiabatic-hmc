"""
Utility functions for the counterdiabatic project.
"""

import jax.numpy as jnp
import jax
import blackjax


# def normalize(weights):
#     """Normalize weights to sum to 1."""
#     return weights / jnp.sum(weights)


def check_nans(name, value, step=None):
    """Helper function to check for NaNs and print warnings."""
    # Convert to numpy for checking to avoid JAX tracing issues
    if hasattr(value, 'numpy'):
        value_np = value.numpy()
    else:
        value_np = value
    
    if jnp.isnan(value_np).any():
        count = jnp.isnan(value_np).sum()
        step_info = f" at step {step}" if step is not None else ""
        print(f"⚠️  NaN detected in {name}{step_info} (count: {count})")
        return True
    return False

def compute_ess(log_weights):
    """Compute effective sample size from log weights using the formula:
    ESS = (Σ_{n=1}^{N} w_t(x_{t-1}^n))^2 / Σ_{n=1}^{N} w_t(x_{t-1}^n)^2
    """
    # Convert log weights to regular weights
    weights = jnp.exp(log_weights - jnp.max(log_weights))  # Numerical stability
    
    # Compute the formula: (sum of weights)^2 / sum of squared weights
    sum_weights = jnp.sum(weights)
    sum_squared_weights = jnp.sum(weights ** 2)
    
    # ESS = (sum_weights)^2 / sum_squared_weights
    ess = (sum_weights ** 2) / sum_squared_weights
    
    return ess

def normalize_log_weights(log_weights):
    return log_weights - jax.scipy.special.logsumexp(log_weights)

def systematic_resample(q, p, log_weights, rng_key, M):
    weights = jnp.exp(normalize_log_weights(log_weights))
    N = q.shape[0]
    indices = blackjax.smc.resampling.systematic(rng_key, weights, N)
    q_resampled = q[indices]
    p_resampled = p[indices]
    log_weights_reset = jnp.zeros(M)
    return q_resampled, p_resampled, log_weights_reset

def multinomial_resample(q, p, log_weights, rng_key, M):
    """Perform multinomial resampling and reset weights to uniform."""
    weights = jnp.exp(normalize_log_weights(log_weights))
    
    # Generate multinomial samples
    indices = jax.random.choice(key=rng_key, a=M, shape=(M,), p=weights, replace=True)
    
    # Resample particles
    q_resampled = q[indices]
    p_resampled = p[indices]
    
    # Reset weights to uniform (log weights = 0)
    log_weights_reset = jnp.zeros(M)
    
    return q_resampled, p_resampled, log_weights_reset

def compute_expectation_over_particles(values):
    """Compute expectation E_p over current particle distribution (unweighted average)."""
    return jnp.mean(values)

def compute_expectation_over_equilibrium(values, log_weights):
    """Compute expectation E_λ over equilibrium distribution (weighted average)."""
    if log_weights is None or len(log_weights) == 0:
        # If no weights, fall back to unweighted average
        raise ValueError("No weights provided")
    weights = jnp.exp(normalize_log_weights(log_weights))
    return jnp.sum(values * weights)

def generate_initial_samples(M, make_T, make_V, lam, key, dim, initial_sigma=1.0):
    """Generate M samples from a Gaussian distribution with variance matching the potential.
    
    Args:
        M: Number of samples
        make_T: Function to create kinetic energy (unused, kept for compatibility)
        make_V: Function to create potential energy (used to determine variance)
        lam: Current lambda value (used to determine variance)
        key: JAX random key
        dim: Dimension of the system
        variance: Variance of the Gaussian distribution (if None, computed from potential)
    """
    # If variance is not provided, compute it from the potential
    # if variance is None:
    #     # For a potential V(q) = 0.5 * k * q², the variance is 1/k
    #     # We can compute this by evaluating the potential at a test point
    #     test_q = jnp.ones(dim)
    #     V = make_V(lam)
    #     potential_value = V(test_q)
    #     # V(q) = 0.5 * k * ||q||², so k = 2 * V(q) / ||q||²
    #     k = 2.0 * potential_value / jnp.sum(test_q ** 2)
    #     variance = 1.0 / k
    #     print(f"Computed variance from potential: {variance:.3f} (k = {k:.3f})")

    # variance = 2.0 ** 2
    
    # Draw independent samples from Gaussian with given variance
    key, sub = jax.random.split(key)
    q = jax.random.normal(sub, (M, dim)) * initial_sigma
    key, sub = jax.random.split(key)
    p = jax.random.normal(sub, (M, dim))
    
    # Check initial samples for NaNs
    check_nans("initial_q", jnp.concatenate([q, p], axis=1))
    
    return q, p

def compute_energy_stats(q, p, lam, make_T, make_V, A_ansatz=None, log_weights=None):
    """Compute average H, H², ∂H/∂λ, (∂H/∂λ)², and {A,H} over particles."""
    T = make_T(lam)
    V = make_V(lam)
    
    # Compute H for each particle
    H_vals = jax.vmap(lambda qr, pr: T(pr) + V(qr))(q, p)
    
    # Compute ∂H/∂λ for each particle
    dH_dlam = lambda q, p: (jax.grad(lambda q, p, lam: make_V(lam)(q), argnums=2)(q, p, lam))
    
    dH_dlam_vals = jax.vmap(lambda qr, pr: dH_dlam(qr, pr))(q, p)
    
    # Compute expectations over current particle distribution (E_p)
    E_p_H = compute_expectation_over_particles(H_vals)
    E_p_H_sq = compute_expectation_over_particles(H_vals ** 2)
    E_p_dH_dlam = compute_expectation_over_particles(dH_dlam_vals)
    E_p_dH_dlam_sq = compute_expectation_over_particles(dH_dlam_vals ** 2)
    
    # Compute expectations over equilibrium distribution (E_λ)
    E_lambda_H = compute_expectation_over_equilibrium(H_vals, log_weights)
    E_lambda_H_sq = compute_expectation_over_equilibrium(H_vals ** 2, log_weights)
    E_lambda_dH_dlam = compute_expectation_over_equilibrium(dH_dlam_vals, log_weights)
    E_lambda_dH_dlam_sq = compute_expectation_over_equilibrium(dH_dlam_vals ** 2, log_weights)
    
    # Compute variance under current distribution
    var_dH_dlam = E_p_dH_dlam_sq - E_p_dH_dlam ** 2
    
    # Compute variance under equilibrium distribution
    var_lambda_dH_dlam = E_lambda_dH_dlam_sq - E_lambda_dH_dlam ** 2
    
    # Compute the expectation difference squared
    expectation_diff_sq = (E_p_dH_dlam - E_lambda_dH_dlam) ** 2
    
    # Compute gauge potential statistics if A_ansatz is provided
    E_p_A_H = 0.0
    E_p_A_H_sq = 0.0
    E_lambda_A_H = 0.0
    E_lambda_A_H_sq = 0.0
    var_A = 0.0
    var_lambda_A = 0.0
    if A_ansatz is not None:
        from .physics import poisson_bracket_fn
        H_fixed = lambda q, p: T(p) + V(q)
        
        # Compute {A,H} for each particle
        A_H_vals = jax.vmap(lambda qr, pr: poisson_bracket_fn(A_ansatz, H_fixed)(qr, pr))(q, p)
        E_p_A_H = compute_expectation_over_particles(A_H_vals)
        E_p_A_H_sq = compute_expectation_over_particles(A_H_vals ** 2)
        E_lambda_A_H = compute_expectation_over_equilibrium(A_H_vals, log_weights)
        E_lambda_A_H_sq = compute_expectation_over_equilibrium(A_H_vals ** 2, log_weights)
        
        # Compute A for each particle
        A_vals = jax.vmap(lambda qr, pr: A_ansatz(qr, pr))(q, p)
        E_p_A = compute_expectation_over_particles(A_vals)
        E_p_A_sq = compute_expectation_over_particles(A_vals ** 2)
        E_lambda_A = compute_expectation_over_equilibrium(A_vals, log_weights)
        E_lambda_A_sq = compute_expectation_over_equilibrium(A_vals ** 2, log_weights)
        
        # Compute variances: Var[A] = <A²> - <A>²
        var_A = E_p_A_sq - E_p_A ** 2
        var_lambda_A = E_lambda_A_sq - E_lambda_A ** 2
    
    return {
        # Current particle distribution expectations (E_p)
        'E_p_H': float(E_p_H),
        'E_p_H_sq': float(E_p_H_sq),
        'E_p_dH_dlam': float(E_p_dH_dlam),
        'E_p_dH_dlam_sq': float(E_p_dH_dlam_sq),
        'var_dH_dlam': float(var_dH_dlam),  # Var_p[∂_λ H] under current distribution
        
        # Equilibrium distribution expectations (E_λ)
        'E_lambda_H': float(E_lambda_H),
        'E_lambda_H_sq': float(E_lambda_H_sq),
        'E_lambda_dH_dlam': float(E_lambda_dH_dlam),
        'E_lambda_dH_dlam_sq': float(E_lambda_dH_dlam_sq),
        'var_lambda_dH_dlam': float(var_lambda_dH_dlam),  # Var_λ[∂_λ H] under equilibrium distribution
        
        # Expectation difference squared
        'expectation_diff_sq': float(expectation_diff_sq),  # (E_p[∂_λ H] - E_λ[∂_λ H])²
        
        # Gauge potential statistics (for CD methods)
        'E_p_A_H': float(E_p_A_H),  # E_p[{A,H}] - Poisson bracket under particle distribution
        'E_p_A_H_sq': float(E_p_A_H_sq),  # E_p[{A,H}²]
        'E_lambda_A_H': float(E_lambda_A_H),  # E_λ[{A,H}] - Poisson bracket under equilibrium
        'E_lambda_A_H_sq': float(E_lambda_A_H_sq),  # E_λ[{A,H}²]
        'var_A': float(var_A),  # Var_p[A] - variance of gauge potential under particle distribution
        'var_lambda_A': float(var_lambda_A),  # Var_λ[A] - variance under equilibrium distribution
        'H_vals': H_vals  # Store individual H values
    }

#### Hermite helper functions
################################################################################

def print_tridiagonal_matrix_info(diagonal, upper_diagonal, L_q, b_vector, num_coeffs):
    """
    Print information about the tridiagonal matrix M and related quantities.
    
    Args:
        diagonal: Main diagonal of the tridiagonal matrix
        upper_diagonal: Upper diagonal of the tridiagonal matrix
        L_q: Linear term coefficient
        b_vector: Right-hand side vector
        num_coeffs: Number of coefficients (matrix size)
    """
    print(f"    Tridiagonal matrix M (size {num_coeffs}x{num_coeffs}):")
    print(f"    Linear term L_q: {L_q:.6f}")
    print(f"    Right-hand side b: {b_vector}")
    
    # Construct full matrix for pretty printing
    M_full = jnp.zeros((num_coeffs, num_coeffs))
    for i in range(num_coeffs):
        M_full = M_full.at[i, i].set(diagonal[i])
        if i < num_coeffs - 1:
            M_full = M_full.at[i, i+1].set(upper_diagonal[i])
            M_full = M_full.at[i+1, i].set(upper_diagonal[i])
    
    print(f"    Full matrix M:")
    for i in range(num_coeffs):
        row_str = "    "
        for j in range(num_coeffs):
            if abs(M_full[i, j]) < 1e-10:
                row_str += "  0.000000  "
            else:
                row_str += f"{M_full[i, j]:10.6f}  "
        print(row_str)


def print_optimization_summary(step_name, initial_loss, final_loss, num_steps=None):
    """
    Print a summary of optimization results.
    
    Args:
        step_name: Name of the optimization step (e.g., "f(q) optimization")
        initial_loss: Initial loss value (can be "N/A" for cases where it's not applicable)
        final_loss: Final loss value
        num_steps: Number of optimization steps (optional)
    """
    if initial_loss == "N/A":
        loss_str = f"loss N/A → {final_loss:.6f}"
    else:
        loss_str = f"loss {initial_loss:.6f} → {final_loss:.6f}"
    
    if num_steps is not None:
        print(f"    {step_name}: {loss_str} ({num_steps} steps)")
    else:
        print(f"    {step_name}: {loss_str}")


def print_coefficients_summary(alpha_coeffs, max_order):
    """
    Print a summary of the learned coefficients.
    
    Args:
        alpha_coeffs: Array of learned coefficients
        max_order: Maximum order of Hermite polynomials
    """
    print(f"    Learned coefficients:")
    for k, alpha_k in enumerate(alpha_coeffs):
        i = 2 * k + 1  # Map k=0,1,2,... to i=1,3,5,...
        print(f"      α̃_{i} = {alpha_k:.6f}")



def save_simulation_data(snapshots, system_name, method_name, ansatz_params=None, param_history=None, ansatz_type=None, integrator_type=None):
    """Save simulation data to a pickle file.
    
    Args:
        snapshots: Dictionary containing simulation snapshots
        system_name: Name of the system (e.g., "double_well")
        method_name: Name of the method (e.g., "cd_unweighted")
        ansatz_params: Ansatz parameters (optional)
        param_history: Parameter history (optional)
        ansatz_type: Type of ansatz (e.g., "polynomial", "neural_network") - for directory organization
        integrator_type: Type of integrator (e.g., "leapfrog", "implicit_midpoint") - for directory organization
    """
    import pickle
    import os
    
    # Create organized directory structure if ansatz_type and integrator_type provided
    if ansatz_type and integrator_type:
        data_dir = f"data/{ansatz_type}/{system_name}/{integrator_type}"
        os.makedirs(data_dir, exist_ok=True)
        filename = f"{data_dir}/{method_name}.pkl"
    else:
        # Fallback to flat structure for backward compatibility
        os.makedirs("data", exist_ok=True)
        filename = f"data/{system_name}_{method_name}.pkl"
    
    # Prepare data to save
    data = {
        'snapshots': snapshots,
        'system_name': system_name,
        'method_name': method_name,
        'ansatz_params': ansatz_params,
        'param_history': param_history
    }
    
    # Save to pickle file
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved simulation data to {filename}")