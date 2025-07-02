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
def fit_gauge_potential(lam, samples, make_T, make_V, A_ansatz, num_iters, lr):
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
    """
    # Check input samples for NaNs
    check_nans("input_samples", samples)
    
    qp_batch = jnp.array(samples)  # shape (N, 2*dim)
    dim = qp_batch.shape[1] // 2  # Extract dimension from sample shape

    H = lambda lam: lambda q, p: make_T(lam)(p) + make_V(lam)(q)
    H_fixed = H(lam)
    dH_fixed = lambda q, p: (jax.grad(lambda q, p, lam: H(lam)(q, p), argnums=2)(q, p, lam))

    def R(A_ansatz, q, p):
        return poisson_bracket_fn(A_ansatz, H_fixed)(q, p) - dH_fixed(q, p)

    def loss_fn(A_ansatz, qp_batch):
        # Split samples into q and p components
        qs = qp_batch[:, :dim]  # First dim columns
        ps = qp_batch[:, dim:]  # Last dim columns
        
        R_vals = jax.vmap(lambda qr, pr, A_ansatz: R(A_ansatz, qr, pr), in_axes=(0, 0, None))(qs, ps, A_ansatz)
        
        # Main loss term
        main_loss = jnp.mean(R_vals ** 2)
        
        # Add weight regularization to prevent large weights
        reg_loss = 0.0
        if isinstance(A_ansatz, eqx.Module):
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
            clipped_grads = {}
            for name, grad in grad_arrays.items():
                # Clip individual gradients to prevent extreme values
                clipped_grads[name] = jnp.clip(grad, -10.0, 10.0)
            
            updates, opt_state = optimizer.update(clipped_grads, opt_state)
            A_ansatz = eqx.apply_updates(A_ansatz, updates)
        
        return A_ansatz, opt_state, loss

    loss_history = []
    for iteration in range(num_iters):
        A_ansatz, opt_state, loss = update(A_ansatz, opt_state, qp_batch)
        
        # Check loss for NaNs (outside of JIT-compiled function)
        if jnp.isnan(loss):
            print(f"⚠️  NaN detected in loss at iteration {iteration}")
            print(f"  Stopping optimization early due to NaN loss")
            break
            
        loss_history.append(float(loss))

    print(f"Fitting completed after {len(loss_history)} iterations")
    return A_ansatz, loss_history 