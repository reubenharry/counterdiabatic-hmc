import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from .physics import poisson_bracket_fn

def check_nans(name, value, iteration=None):
    """Helper function to check for NaNs and print warnings."""
    # Convert to numpy for checking to avoid JAX tracing issues
    if hasattr(value, 'numpy'):
        value_np = value.numpy()
    else:
        value_np = value
    
    if jnp.isnan(value_np).any():
        count = jnp.isnan(value_np).sum()
        iter_info = f" at iteration {iteration}" if iteration is not None else ""
        print(f"⚠️  NaN detected in {name}{iter_info} (count: {count})")
        return True
    return False

# =============================================================================
#  FIT FUNCTION USING GENERAL POISSON BRACKET
# =============================================================================
def calculate_gauge_potential_loss(lam, samples, make_T, make_V, A_ansatz, use_regularization=False, weights=None):
    """
    Calculate the loss for a gauge potential ansatz without optimization.
    This is the same loss function used in fitting.
    
    Args:
        lam: Current lambda value
        samples: Array of shape (N, 2*dim) where first dim columns are q and last dim columns are p
        make_T: Function to create kinetic energy
        make_V: Function to create potential energy
        A_ansatz: The ansatz to evaluate
        use_regularization: Whether to include L2 regularization
        weights: Optional array of weights for weighted expectation (shape (N,))
        
    Returns:
        The loss value (float)
    """
    qp_batch = jnp.array(samples)  # shape (N, 2*dim)
    dim = qp_batch.shape[1] // 2  # Extract dimension from sample shape

    H = lambda lam: lambda q, p: make_T(lam)(p) + make_V(lam)(q)
    H_fixed = H(lam)
    dH_fixed = lambda q, p: (jax.grad(lambda q, p, lam: H(lam)(q, p), argnums=2)(q, p, lam))

    def R(A_ansatz, q, p):
        return poisson_bracket_fn(A_ansatz, H_fixed)(q, p) - dH_fixed(q, p)

    # Split samples into q and p components
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
    """
    Fit A(q,p; θ) by minimizing mean_{samples}[ ( {A,H} - ∂H/∂μ )^2 ].
    Returns both the optimized parameters and the loss history.
    
    Args:
        lam: Current lambda value
        samples: Array of shape (N, 2*dim) where first dim columns are q and last dim columns are p
        make_T: Function to create kinetic energy
        make_V: Function to create potential energy
        A_ansatz: The ansatz to fit
        num_iters: Number of optimization iterations
        lr: Learning rate
        use_regularization: Whether to use L2 regularization
        weights: Optional array of weights for weighted expectation (shape (N,))
    """
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

def fit_hermite_ansatz(lam, samples, make_T, make_V, hermite_ansatz, num_iters, lr, use_regularization=False, weights=None):
    """
    Special fitting function for HermiteAnsatz that only optimizes g(p) coefficients (alpha_coeffs)
    while keeping f(q) parameters fixed.
    
    Args:
        lam: Current lambda value
        samples: Array of shape (N, 2*dim) where first dim columns are q and last dim columns are p
        make_T: Function to create kinetic energy
        make_V: Function to create potential energy
        hermite_ansatz: HermiteAnsatz instance to fit
        num_iters: Number of optimization iterations
        lr: Learning rate
        use_regularization: Whether to use L2 regularization
        weights: Optional array of weights for weighted expectation (shape (N,))
        
    Returns:
        Updated hermite_ansatz and loss history
    """
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