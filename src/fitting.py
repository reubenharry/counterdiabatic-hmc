import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from src.ansatze import AnalyticAnsatz
from .physics import poisson_bracket_fn
from .utils import check_nans, print_tridiagonal_matrix_info, print_optimization_summary
import scipy.linalg



# =============================================================================
#  FIT FUNCTION USING GENERAL POISSON BRACKET
# =============================================================================
def calculate_gauge_potential_loss(lam, samples, make_T, make_V, A_ansatz, use_regularization=False, weights=None):
    
    qp_batch = jnp.array(samples)  # shape (N, 2*dim)
    dim = qp_batch.shape[1] // 2  # Extract dimension from sample shape

    H = lambda lam: lambda q, p: make_T(lam)(p) + make_V(lam)(q)
    H_fixed = H(lam)
    dH_fixed = lambda q, p: (jax.grad(lambda q, p, lam: H(lam)(q, p), argnums=2)(q, p, lam))

    def R(A_ansatz, q, p):
        return poisson_bracket_fn(A_ansatz, H_fixed)(q, p) - dH_fixed(q, p)

    qs = qp_batch[:, :dim]  # First dim columns
    ps = qp_batch[:, dim:]  # Last dim columns
    
    R_vals = jax.vmap(lambda qr, pr, A_ansatz: R(A_ansatz, qr, pr), in_axes=(0, 0, None))(qs, ps, A_ansatz)
    
    # Main loss term - use weighted mean if weights are provided
    if weights is not None:
        # Normalize weights to sum to 1
        weights = jnp.array(weights)
        weights = weights / jnp.sum(weights)
        main_loss = jnp.sum(weights * (R_vals ** 2))
    else:
        main_loss = jnp.mean(R_vals ** 2)
    
    # Add weight regularization to prevent large weights (optional)
    reg_loss = 0.0
    if use_regularization and isinstance(A_ansatz, eqx.Module):
        # Handle different ansatz types
        if hasattr(A_ansatz, 'params') and isinstance(A_ansatz.params, jnp.ndarray):
            # PolynomialAnsatz case - params is a single array
            reg_loss += 1e-4 * jnp.mean(A_ansatz.params ** 2)  # Stronger regularization
        elif hasattr(A_ansatz, 'layers'):
            # Neural network case - handle layers directly
            for layer in A_ansatz.layers:
                if isinstance(layer, eqx.nn.Linear):
                    reg_loss += 1e-6 * jnp.mean(layer.weight ** 2)
                    reg_loss += 1e-6 * jnp.mean(layer.bias ** 2)
    
    total_loss = main_loss + reg_loss
    return total_loss

def fit_gauge_potential(lam, samples, make_T, make_V, A_ansatz, num_iters, lr, use_regularization=False, weights=None):

    if A_ansatz is None:
        return [], None

    if isinstance(A_ansatz, AnalyticAnsatz):
        A_ansatz = eqx.tree_at(lambda m: m.params, A_ansatz, jnp.array([lam]))
        loss = calculate_gauge_potential_loss(lam, samples, make_T, make_V, A_ansatz, 
                                                        use_regularization=False, weights=weights)
        return A_ansatz, [loss]

    # Check if this is a HermiteAnsatz and use special fitting
    if A_ansatz.ansatz_type == 'hermite':
        print(A_ansatz.f_ansatz(jnp.array([1.0])), "A_ansatz.f_ansatz 1 orig")
        return fit_hermite_ansatz(
            lam=lam, samples=samples, make_T=make_T, make_V=make_V, hermite_ansatz=A_ansatz, num_iters=5, lr=lr, use_regularization=use_regularization, weights=weights)
    
    # Check input samples for NaNs
    check_nans("input_samples", samples)
    
    qp_batch = jnp.array(samples)  # shape (N, 2*dim)

    def loss_fn(A_ansatz, qp_batch):
        # Use the reusable loss calculation function
        return calculate_gauge_potential_loss(lam, qp_batch, make_T, make_V, A_ansatz, use_regularization, weights)

    # Use gradient clipping to prevent exploding gradients
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients by global norm
        optax.adam(lr)
    )
    
    # Initialize optimizer based on ansatz type
    if hasattr(A_ansatz, 'params') and isinstance(A_ansatz.params, jnp.ndarray):
        # PolynomialAnsatz case - just use the params array
        opt_state = optimizer.init(A_ansatz.params)
    else:
        # Neural network case - use eqx.filter
        opt_state = optimizer.init(eqx.filter(A_ansatz, eqx.is_array))

    @jax.jit
    def update(A_ansatz, opt_state, qp_batch):
        loss, grads = jax.value_and_grad(loss_fn)(A_ansatz, qp_batch)
        
        if hasattr(A_ansatz, 'params') and isinstance(A_ansatz.params, jnp.ndarray):
            # PolynomialAnsatz case - handle params directly
            # Extract the gradient for the params field
            param_grads = grads.params
            clipped_grads = jnp.clip(param_grads, -10.0, 10.0)
            updates, opt_state = optimizer.update(clipped_grads, opt_state)
            A_ansatz = eqx.tree_at(lambda m: m.params, A_ansatz, A_ansatz.params + updates)
        else:
            # Neural network case - use eqx.filter
            grad_arrays = eqx.filter(grads, eqx.is_array)
            # Clip gradients to prevent extreme values
            clipped_grads = jax.tree_map(lambda g: jnp.clip(g, -10.0, 10.0), grad_arrays)
            
            updates, opt_state = optimizer.update(clipped_grads, opt_state)
            A_ansatz = eqx.apply_updates(A_ansatz, updates)
        
        return A_ansatz, opt_state, loss

    loss_history = []
    best_loss = float('inf')
    patience = 50  # Number of iterations to wait for improvement
    patience_counter = 0
    
    for iteration in range(num_iters):
        A_ansatz, opt_state, loss = update(A_ansatz, opt_state, qp_batch)
        
        # Check loss for NaNs (outside of JIT-compiled function)
        if jnp.isnan(loss):
            print(f"⚠️  NaN detected in loss at iteration {iteration}")
            print(f"  Stopping optimization early due to NaN loss")
            break
            
        loss_history.append(float(loss))
        
        # Early stopping: check if loss has improved
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Stop if loss hasn't improved for patience iterations
        if patience_counter >= patience and iteration > 100:  # Wait at least 100 iterations
            print(f"Early stopping at iteration {iteration} (loss: {loss:.6f})")
            break

    print(f"Fitting completed after {len(loss_history)} iterations")
    return A_ansatz, loss_history

def fit_hermite_ansatz_optimize(lam, samples, make_T, make_V, hermite_ansatz, num_iters, lr, use_regularization=False, weights=None):

    print("  Using gradient descent for g(p) coefficients only (f(q) fixed)")
    
    # Check input samples for NaNs
    check_nans("input_samples", samples)
    
    qp_batch = jnp.array(samples)  # shape (N, 2*dim)

    def loss_fn(alpha_coeffs, qp_batch):
        # Create a temporary ansatz with the given alpha_coeffs
        temp_ansatz = hermite_ansatz
        temp_ansatz = eqx.tree_at(lambda m: m.alpha_coeffs, temp_ansatz, alpha_coeffs)
        return calculate_gauge_potential_loss(lam, qp_batch, make_T, make_V, temp_ansatz, use_regularization, weights)

    # Use gradient clipping to prevent exploding gradients
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients by global norm
        optax.adam(lr)
    )
    
    # Initialize optimizer with only alpha_coeffs
    opt_state = optimizer.init(hermite_ansatz.alpha_coeffs)

    @jax.jit
    def update(alpha_coeffs, opt_state, qp_batch):
        loss, grads = jax.value_and_grad(loss_fn)(alpha_coeffs, qp_batch)
        
        # Clip gradients to prevent extreme values
        clipped_grads = jnp.clip(grads, -10.0, 10.0)
        updates, opt_state = optimizer.update(clipped_grads, opt_state)
        alpha_coeffs = alpha_coeffs + updates
        
        return alpha_coeffs, opt_state, loss

    loss_history = []
    best_loss = float('inf')
    patience = 50  # Number of iterations to wait for improvement
    patience_counter = 0
    
    # Initialize alpha_coeffs with small random values instead of zeros
    import jax.random as jr
    key = jr.PRNGKey(42)
    hermite_ansatz = eqx.tree_at(lambda m: m.alpha_coeffs, hermite_ansatz, 0.01 * jr.normal(key, shape=hermite_ansatz.alpha_coeffs.shape))
    
    for iteration in range(num_iters):
        new_alpha_coeffs, opt_state, loss = update(hermite_ansatz.alpha_coeffs, opt_state, qp_batch)
        hermite_ansatz = eqx.tree_at(lambda m: m.alpha_coeffs, hermite_ansatz, new_alpha_coeffs)
        
        # Check loss for NaNs (outside of JIT-compiled function)
        if jnp.isnan(loss):
            print(f"⚠️  NaN detected in loss at iteration {iteration}")
            print(f"  Stopping optimization early due to NaN loss")
            break
            
        loss_history.append(float(loss))
        
        # Early stopping: check if loss has improved
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Stop if loss hasn't improved for patience iterations
        if patience_counter >= patience and iteration > 100:  # Wait at least 100 iterations
            print(f"Early stopping at iteration {iteration} (loss: {loss:.6f})")
            break

    print(f"Fitting completed after {len(loss_history)} iterations")
    
    # Print learned coefficients
    hermite_ansatz.print_coefficients()
    
    return hermite_ansatz, loss_history


def fit_f_at_fixed_g(lam, samples, make_T, make_V, hermite_ansatz, num_steps, lr, use_regularization=False, weights=None):
    """
    Fit f(q) parameters using gradient descent while keeping g(p) coefficients fixed.
    
    Args:
        lam: Current lambda value
        samples: Array of shape (N, 2*dim) where first dim columns are q and last dim columns are p
        make_T: Function to create kinetic energy
        make_V: Function to create potential energy
        hermite_ansatz: HermiteAnsatz instance to fit
        num_steps: Number of gradient descent steps
        lr: Learning rate
        use_regularization: Whether to use L2 regularization
        weights: Optional array of weights for weighted expectation (shape (N,))
        
    Returns:
        Updated hermite_ansatz and loss history
    """
    qp_batch = jnp.array(samples)
    
    # Initialize optimizer for f(q) parameters
    f_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr)
    )
    f_opt_state = f_optimizer.init(hermite_ansatz.f_ansatz.params)
    
    def f_loss_fn(f_params, qp_batch):
        # Create temporary ansatz with updated f_params
        temp_f_ansatz = eqx.tree_at(lambda m: m.params, hermite_ansatz.f_ansatz, f_params)
        temp_ansatz = eqx.tree_at(lambda m: m.f_ansatz, hermite_ansatz, temp_f_ansatz)
        
        # Compute main loss
        main_loss = calculate_gauge_potential_loss(lam, qp_batch, make_T, make_V, temp_ansatz, False, weights)
        
        # Add L2 regularization to prevent parameters from going to zero
        reg_loss = 1e-4 * jnp.sum(f_params ** 2)
        
        return main_loss + reg_loss
    
    @jax.jit
    def update_f(f_params, f_opt_state, qp_batch):
        loss, grads = jax.value_and_grad(f_loss_fn)(f_params, qp_batch)
        clipped_grads = jnp.clip(grads, -10.0, 10.0)
        updates, f_opt_state = f_optimizer.update(clipped_grads, f_opt_state)
        f_params = f_params + updates
        return f_params, f_opt_state, loss
    
    # Run gradient descent steps
    f_params = hermite_ansatz.f_ansatz.params
    f_losses = []
    
    for step in range(num_steps):
        f_params, f_opt_state, f_loss = update_f(f_params, f_opt_state, qp_batch)
        f_losses.append(float(f_loss))
        
        if jnp.isnan(f_loss):
            print(f"    ⚠️  NaN detected in f(q) loss at step {step}")
            break
    
    # Update the ansatz with new f(q) parameters
    hermite_ansatz = eqx.tree_at(lambda m: m.f_ansatz.params, hermite_ansatz, f_params)
    
    return hermite_ansatz, f_losses


def fit_g_at_fixed_f(lam, samples, make_T, make_V, hermite_ansatz, use_regularization=False, weights=None):
    """
    Fit g(p) coefficients using tridiagonal solver while keeping f(q) parameters fixed.
    
    Args:
        lam: Current lambda value
        samples: Array of shape (N, 2*dim) where first dim columns are q and last dim columns are p
        make_T: Function to create kinetic energy
        make_V: Function to create potential energy
        hermite_ansatz: HermiteAnsatz instance to fit
        use_regularization: Whether to use L2 regularization (ignored for tridiagonal approach)
        weights: Optional array of weights for weighted expectation (shape (N,))
        
    Returns:
        Updated hermite_ansatz and final loss
    """
    # Import the existing functions from ansatze.py
    from .ansatze import construct_hermite_tridiagonal_matrix, compute_linear_term_coefficient
    
    # Get f(q) function from the ansatz
    f_function = hermite_ansatz.f_ansatz
    print(f_function(jnp.array([1.0])), "f_function 1")
    
    # Use existing functions to construct tridiagonal matrix
    diagonal, upper_diagonal = construct_hermite_tridiagonal_matrix(
        f_function, samples, make_V, lam, hermite_ansatz.max_order
    )
    
    # Use existing function to compute linear term
    L_q = compute_linear_term_coefficient(f_function, samples, make_V, lam)
    
    # Construct right-hand side vector b^(o)
    num_coeffs = len(hermite_ansatz.alpha_coeffs)
    b_vector = jnp.zeros(num_coeffs)
    b_vector = b_vector.at[0].set(2.0 * L_q)  # Only first component is non-zero
    
    # Print matrix information using utility function
    print_tridiagonal_matrix_info(diagonal, upper_diagonal, L_q, b_vector, num_coeffs)
    
    # Solve the linear system M^(o) α̃^(o) = -b^(o)
    
    
    # Add regularization to ensure positive definiteness
    regularization = 1e-6
    diagonal_reg = diagonal + regularization
    
    try:
        alpha_optimized = scipy.linalg.solveh_banded(
            2* jnp.vstack([
                jnp.concatenate([jnp.array([0]), upper_diagonal]),  # Upper diagonal
                diagonal_reg,  # Main diagonal with regularization
            ]),
            -b_vector
        )
    except scipy.linalg.LinAlgError:
        print("    ⚠️  Matrix not positive definite, using regular solve...")
        # Fall back to regular solve with full matrix
        M = jnp.zeros((num_coeffs, num_coeffs))
        for i in range(num_coeffs):
            M = M.at[i, i].set(diagonal_reg[i])
            if i < num_coeffs - 1:
                M = M.at[i, i+1].set(upper_diagonal[i])
                M = M.at[i+1, i].set(upper_diagonal[i])
        
        alpha_optimized = jax.scipy.linalg.solve(M, -b_vector)
    
    # Update the ansatz with optimized g(p) coefficients
    hermite_ansatz = eqx.tree_at(lambda m: m.alpha_coeffs, hermite_ansatz, alpha_optimized)
    
    # Compute final loss
    qp_batch = jnp.array(samples)
    final_loss = calculate_gauge_potential_loss(lam, qp_batch, make_T, make_V, hermite_ansatz, False, weights)
    
    return hermite_ansatz, final_loss


def fit_hermite_ansatz(lam, samples, make_T, make_V, hermite_ansatz, num_iters, lr, use_regularization=False, weights=None):
    """
    Fit HermiteAnsatz using one iteration: fit f(q) once, then g(p) once.
    
    This function:
    1. Fits f(q) parameters using gradient descent (one step)
    2. Fits g(p) coefficients using tridiagonal matrix optimization
    
    Args:
        lam: Current lambda value
        samples: Array of shape (N, 2*dim) where first dim columns are q and last dim columns are p
        make_T: Function to create kinetic energy
        make_V: Function to create potential energy
        hermite_ansatz: HermiteAnsatz instance to fit
        num_iters: Number of iterations (ignored, kept for compatibility)
        lr: Learning rate for f(q) gradient descent
        use_regularization: Whether to use L2 regularization for f(q)
        weights: Optional array of weights for weighted expectation (shape (N,))
        
    Returns:
        Updated hermite_ansatz and loss history
    """
    print("  Using one iteration: f(q) gradient descent + g(p) tridiagonal solver")
    
    # Check input samples for NaNs
    check_nans("input_samples", samples)
    
    loss_history = []

    for i in range(num_iters):
    
        # hermite_ansatz, f_losses = fit_f_at_fixed_g(
        #     lam, samples, make_T, make_V, hermite_ansatz, 
        #     num_steps=1000, lr=lr, use_regularization=use_regularization, weights=weights
        # )
        
        hermite_ansatz, final_loss = fit_g_at_fixed_f(
            lam, samples, make_T, make_V, hermite_ansatz, 
            use_regularization=use_regularization, weights=weights
        )
        loss_history.append(final_loss)
    
    print_optimization_summary("g(p) optimization", "N/A", final_loss)
    print(f"  Final loss: {final_loss:.6f}")
    
    # Print learned coefficients
    hermite_ansatz.print_coefficients()
    
    return hermite_ansatz, loss_history 