"""
Utility functions for the counterdiabatic project.
"""

import jax.numpy as jnp


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
