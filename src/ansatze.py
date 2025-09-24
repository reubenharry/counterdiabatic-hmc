import jax
import jax.numpy as jnp
import equinox as eqx
import itertools
import scipy 

def check_nans(name, value):
    """Helper function to check for NaNs and print warnings."""
    # Convert to numpy for checking to avoid JAX tracing issues
    if hasattr(value, 'numpy'):
        value_np = value.numpy()
    else:
        value_np = value
    
    if jnp.isnan(value_np).any():
        count = jnp.isnan(value_np).sum()
        print(f"⚠️  NaN detected in {name} (count: {count})")
        return True
    return False

class A_ansatz(eqx.Module):
    """Base class for gauge potential ansatz."""
    def __call__(self, q, p):
        raise NotImplementedError

class F_ansatz(eqx.Module):
    """Base class for f(q) ansatz (position-only)."""
    def __call__(self, q):
        raise NotImplementedError

def generate_polynomial_terms(max_degree, dim):
    """Generate all polynomial terms up to max_degree in p and q for dim-dimensional vectors.
    
    Returns a list of tuples (coeff_name, q_powers, p_powers) representing terms like:
    - (θ1, [0,0], [1,0]) for p_0
    - (θ2, [1,0], [1,0]) for q_0*p_0
    - (θ3, [2,0], [0,0]) for q_0^2
    etc.
    
    Args:
        max_degree: Maximum total degree of the polynomial
        dim: Dimension of q and p vectors
    """
    terms = []
    term_idx = 1
    
    for total_degree in range(max_degree + 1):
        # Generate all possible combinations of q and p powers that sum to total_degree
        for q_degree in range(total_degree + 1):
            p_degree = total_degree - q_degree
            
            # Generate all possible q power distributions across dimensions
            if q_degree == 0:
                q_powers_list = [[0] * dim]
            else:
                q_powers_list = []
                for powers in itertools.combinations_with_replacement(range(dim), q_degree):
                    q_powers = [0] * dim
                    for power_idx in powers:
                        q_powers[power_idx] += 1
                    q_powers_list.append(q_powers)
            
            # Generate all possible p power distributions across dimensions
            if p_degree == 0:
                p_powers_list = [[0] * dim]
            else:
                p_powers_list = []
                for powers in itertools.combinations_with_replacement(range(dim), p_degree):
                    p_powers = [0] * dim
                    for power_idx in powers:
                        p_powers[power_idx] += 1
                    p_powers_list.append(p_powers)
            
            # Combine q and p power distributions
            for q_powers in q_powers_list:
                for p_powers in p_powers_list:
                    terms.append((f"θ{term_idx}", q_powers, p_powers))
                    term_idx += 1
    
    return terms

class PolynomialAnsatz(A_ansatz):
    """Polynomial ansatz for the gauge potential."""
    params: jnp.ndarray
    terms: list = eqx.static_field()
    ansatz_type: str = eqx.static_field()
    dim: int = eqx.static_field()

    def __init__(self, max_degree, dim=1):
        self.dim = dim
        self.terms = generate_polynomial_terms(max_degree, dim)
        self.params = jnp.zeros(len(self.terms))
        self.ansatz_type = 'polynomial'

    def __call__(self, q, p):
        q = jnp.atleast_1d(q)
        p = jnp.atleast_1d(p)
        result = 0.0
        for i, (_, q_powers, p_powers) in enumerate(self.terms):
            term = self.params[i]
            # Compute q^q_powers * p^p_powers
            for d in range(self.dim):
                if q_powers[d] > 0:
                    term = term * (q[d] ** q_powers[d])
                if p_powers[d] > 0:
                    term = term * (p[d] ** p_powers[d])
            result += term
        
        return result

    def get_term_description(self):
        """Return a description of what each parameter represents."""
        descriptions = []
        for coeff_name, q_powers, p_powers in self.terms:
            term_str = ""
            
            # Add q terms
            for d in range(self.dim):
                if q_powers[d] > 0:
                    if term_str:
                        term_str += " "
                    term_str += f"q_{d}^{q_powers[d]}" if q_powers[d] > 1 else f"q_{d}"
            
            # Add p terms
            for d in range(self.dim):
                if p_powers[d] > 0:
                    if term_str:
                        term_str += " "
                    term_str += f"p_{d}^{p_powers[d]}" if p_powers[d] > 1 else f"p_{d}"
            
            if not term_str:
                term_str = "1"
            
            descriptions.append(f"{coeff_name}: {term_str}")
        return descriptions

class PolynomialFAnsatz(F_ansatz):
    """Polynomial ansatz for f(q) (position-only)."""
    params: jnp.ndarray
    terms: list = eqx.static_field()
    ansatz_type: str = eqx.static_field()
    dim: int = eqx.static_field()

    def __init__(self, max_degree, dim=1):
        self.dim = dim
        # Generate terms for f(q) only (no p terms)
        self.terms = []
        term_idx = 1
        
        for total_degree in range(max_degree + 1):
            # Only q terms, no p terms
            if total_degree == 0:
                q_powers_list = [[0] * dim]
            else:
                q_powers_list = []
                for powers in itertools.combinations_with_replacement(range(dim), total_degree):
                    q_powers = [0] * dim
                    for power_idx in powers:
                        q_powers[power_idx] += 1
                    q_powers_list.append(q_powers)
            
            for q_powers in q_powers_list:
                self.terms.append((f"θ{term_idx}", q_powers))
                term_idx += 1
        
        self.params = jnp.zeros(len(self.terms))
        self.ansatz_type = 'polynomial_f'

    def __call__(self, q):
        q = jnp.atleast_1d(q)
        result = 0.0
        for i, (_, q_powers) in enumerate(self.terms):
            term = self.params[i]
            # Compute q^q_powers
            for d in range(self.dim):
                if q_powers[d] > 0:
                    term = term * (q[d] ** q_powers[d])
            result += term
        
        return result

class AnalyticAnsatz(A_ansatz):
    """
    An ansatz with a fixed analytical form from a screenshot.
    NOTE: This formula is specifically for a 'gaussian_annealing' potential
    of the form V(q) = 0.5 * L * q^2, where L is the potential's coefficient.
    The parameter `lam` in the original formula corresponds to L.
    """
    params: jnp.ndarray  # Will store the current schedule lambda value
    ansatz_type: str = eqx.static_field()

    def __init__(self):
        self.params = jnp.array([0.0])  # Initialize with schedule lambda = 0
        self.ansatz_type = 'analytic'

    def __call__(self, q, p):
        lam_schedule = self.params[0]  # Get current schedule lambda value
        
        # Ensure q and p are arrays and handle different shapes
        q = jnp.atleast_1d(q)
        p = jnp.atleast_1d(p)
        
        # For 1D case, return -q*p
        if q.shape[0] == 1 and p.shape[0] == 1:
            return -q[0] * p[0]
        else:
            # For multi-dimensional case, return the first component
            return -(q * p)[0]

class NeuralNetworkAnsatz(A_ansatz):
    """Neural network ansatz for the gauge potential."""
    layers: list
    ansatz_type: str = eqx.static_field()
    dim: int = eqx.static_field()

    def __init__(self, dims, key, dim=1):
        # dims: list of layer sizes, e.g. [2*dim, 64, 32, 1] for dim-dimensional q and p
        # The input dimension should be 2*dim (q and p concatenated)
        if dims[0] != 2 * dim:
            raise ValueError(f"First layer dimension should be 2*dim = {2*dim}, got {dims[0]}")
        
        keys = jax.random.split(key, len(dims) - 1)
        self.ansatz_type = 'neural'
        self.dim = dim
        
        # Use smaller initialization scale to prevent large initial values
        self.layers = []
        for i in range(len(dims) - 1):
            # Use Xavier/Glorot initialization with smaller scale
            scale = jnp.sqrt(2.0 / dims[i]) * 0.1  # Reduced scale factor
            layer = eqx.nn.Linear(
                dims[i], 
                dims[i+1], 
                key=keys[i]
            )
            # Manually scale the weights and biases
            layer = eqx.tree_at(lambda m: m.weight, layer, layer.weight * scale)
            layer = eqx.tree_at(lambda m: m.bias, layer, layer.bias * 0.01)
            self.layers.append(layer)

    def __call__(self, q, p):
        # Concatenate q and p for input to MLP
        x = jnp.concatenate([q, p])
        
        for i, layer in enumerate(self.layers[:-1]):
            x = (layer)(x)
            x = jax.nn.tanh(x)
        
        # Final layer
        final_layer = self.layers[-1]
        x = (final_layer)(x)
        
        result = x.squeeze()
        
        return result

# the matrix S in \alpha^T S \alpha
def construct_hermite_matrix(n, particles):
    return jnp.ones(n), jnp.ones(n-1)

def minimize_g(n, particles):

    diag, upper = construct_hermite_matrix(n, particles)

    _, v = scipy.linalg.eigh_tridiagonal(d=diag, e=upper, select='i', select_range=(0, 0), eigvals_only=False)

    return v




# =============================================================================
# ORTHONORMAL HERMITE POLYNOMIAL BASIS (PROBABILISTS' DEFINITION)
# =============================================================================

def hermite_polynomial(n, x):
    """
    Compute the n-th Hermite polynomial H_n(x) using the recurrence relation:
    H_0(x) = 1
    H_1(x) = x
    H_{n+1}(x) = x*H_n(x) - n*H_{n-1}(x)
    
    Args:
        n: Order of the Hermite polynomial
        x: Input value(s)
    
    Returns:
        H_n(x)
    """
    if n == 0:
        return jnp.ones_like(x)
    elif n == 1:
        return x
    else:
        # Use recurrence relation for higher orders
        h_prev = jnp.ones_like(x)  # H_0
        h_curr = x  # H_1
        
        for i in range(2, n + 1):
            h_next = x * h_curr - (i - 1) * h_prev
            h_prev = h_curr
            h_curr = h_next
        
        return h_curr

def orthonormal_hermite_basis(n, x):
    # Use the base hermite_polynomial function and normalize
    H_n = hermite_polynomial(n, x)
    # Use JAX-compatible factorial calculation
    factorial_n = jax.scipy.special.factorial(n)
    return H_n / jnp.sqrt(factorial_n)

def evaluate_g(p, alpha_coeffs, max_order):
    """
    Evaluate g(p) = Σ_{i odd} α̃ᵢ φᵢ(p).
    
    Args:
        p: Momentum vector
        alpha_coeffs: Coefficients for odd-indexed Hermite polynomials
        max_order: Maximum order of Hermite polynomials
        
    Returns:
        g(p) = Σ_{i odd} α̃ᵢ φᵢ(p)
    """
    p = jnp.atleast_1d(p)
    g_p = 0.0
    for k, alpha_k in enumerate(alpha_coeffs):
        i = 2 * k + 1  # Map k=0,1,2,... to i=1,3,5,...
        phi_i = orthonormal_hermite_basis(i, p[0])  # For 1D case
        g_p += alpha_k * phi_i
    return g_p

class HermiteAnsatz(A_ansatz):
    """
    Ansatz of the form A(q,p) = f(q) * g(p) where:
    - f(q) is a parameterized ansatz for the position-dependent part
    - g(p) = Σ_{i odd} α̃ᵢ φᵢ(p) where φᵢ are orthonormal Hermite polynomials
    
    This implements the technique described in the notes with:
    - Orthonormal basis: φᵢ = Hᵢ / √(i!)
    - Only odd-indexed coefficients (i ≥ 1, i odd)
    - Tridiagonal quadratic form for efficient optimization
    
    The f(q) ansatz can be any parameterized function (neural network, polynomial, etc.)
    """
    f_ansatz: F_ansatz  # Parameterized ansatz for f(q)
    alpha_coeffs: jnp.ndarray  # Coefficients α̃ᵢ for odd i
    max_order: int = eqx.static_field()
    ansatz_type: str = eqx.static_field()
    dim: int = eqx.static_field()
    
    def __init__(self, f_ansatz, max_order=5, dim=1):
        """
        Initialize the Hermite ansatz.
        
        Args:
            f_ansatz: Parameterized ansatz for f(q) (e.g., PolynomialFAnsatz)
            max_order: Maximum order of Hermite polynomials (must be odd)
            dim: Dimension of q and p vectors
        """
        self.dim = dim
        self.max_order = max_order
        self.ansatz_type = 'hermite'
        self.f_ansatz = f_ansatz
        
        # Initialize coefficients for odd indices only (i = 1, 3, 5, ...)
        # For max_order = 5, we have coefficients for i = 1, 3, 5
        num_coeffs = (max_order + 1) // 2  # Number of odd indices ≤ max_order
        self.alpha_coeffs = jnp.zeros(num_coeffs)
    
    def __call__(self, q, p):
        """
        Evaluate the ansatz A(q,p) = f(q) * g(p).
        
        Args:
            q: Position vector
            p: Momentum vector
            
        Returns:
            A(q,p) = f(q) * g(p)
        """
        q = jnp.atleast_1d(q)
        p = jnp.atleast_1d(p)
        
        # Compute f(q) using the parameterized f_ansatz
        f_q = self.f_ansatz(q)  # f_ansatz only needs q
        
        # Compute g(p) using the global evaluate_g function
        g_p = evaluate_g(p, self.alpha_coeffs, self.max_order)
        
        return f_q * g_p
    
    def get_alpha_coeffs(self):
        """Return the current Hermite coefficients."""
        return self.alpha_coeffs
    
    def set_alpha_coeffs(self, coeffs):
        """Set the Hermite polynomial coefficients."""
        self.alpha_coeffs = coeffs
    
    def get_f_ansatz(self):
        """Return the f(q) ansatz."""
        return self.f_ansatz
    
    def set_f_ansatz(self, f_ansatz):
        """Set the f(q) ansatz."""
        self.f_ansatz = f_ansatz
    
    def get_all_params(self):
        """Return all parameters (f_ansatz params + alpha_coeffs)."""
        # Get f_ansatz parameters
        if hasattr(self.f_ansatz, 'params'):
            f_params = self.f_ansatz.params
        else:
            # For neural networks, extract all array parameters
            f_params = eqx.filter(self.f_ansatz, eqx.is_array)
        
        return {
            'f_params': f_params,
            'alpha_coeffs': self.alpha_coeffs
        }
    
    def set_all_params(self, params_dict):
        """Set all parameters (f_ansatz params + alpha_coeffs)."""
        if 'alpha_coeffs' in params_dict:
            self.alpha_coeffs = params_dict['alpha_coeffs']
        
        if 'f_params' in params_dict and hasattr(self.f_ansatz, 'params'):
            self.f_ansatz = eqx.tree_at(lambda m: m.params, self.f_ansatz, params_dict['f_params'])
        elif 'f_params' in params_dict:
            # For neural networks, update all array parameters
            self.f_ansatz = eqx.tree_at(lambda m: eqx.filter(m, eqx.is_array), self.f_ansatz, params_dict['f_params'])
    
    def print_coefficients(self):
        """Print the learned Hermite coefficients in a readable format."""
        print("=== Hermite Ansatz Coefficients ===")
        print(f"Max order: {self.max_order}")
        print(f"Number of coefficients: {len(self.alpha_coeffs)}")
        print()
        
        # Print f(q) ansatz info
        print("f(q) ansatz:")
        if hasattr(self.f_ansatz, 'ansatz_type'):
            print(f"  Type: {self.f_ansatz.ansatz_type}")
        if hasattr(self.f_ansatz, 'params'):
            print(f"  Parameters: {self.f_ansatz.params}")
        print()
        
        # Print g(p) coefficients
        print("g(p) = Σ_{i odd} α̃ᵢ φᵢ(p) coefficients:")
        for k, alpha_k in enumerate(self.alpha_coeffs):
            i = 2 * k + 1  # Map k=0,1,2,... to i=1,3,5,...
            print(f"  α̃_{i} = {alpha_k:.6f}  (coefficient for φ_{i}(p))")
        
        print()
        print("Note: φᵢ(p) are orthonormal Hermite polynomials")
        print("      Only odd indices (i=1,3,5,...) are used")
        print("=" * 40)


def construct_hermite_tridiagonal_matrix(f_function, samples, make_V, lam, max_order):
    """
    Construct the tridiagonal matrix M^(o) for the quadratic form in the odd sector.
    
    The quadratic form is:
    E[{A,H}²] = Σₖ (α̃ₖ^(o))² M^(o)_{k,k} + 2 Σₖ α̃ₖ^(o) α̃ₖ₊₁^(o) M^(o)_{k,k+1}
    
    where:
    M^(o)_{k,k} = c₀(2k+2) + c₁(2k+1)
    M^(o)_{k,k+1} = c₂√((2k+2)(2k+3))
    
    with:
    c₀ = E_q[(∂f/∂q)²]
    c₁ = E_q[(∂f/∂q - f ∂V/∂q)²] 
    c₂ = E_q[(∂f/∂q)(∂f/∂q - f ∂V/∂q)]
    
    Args:
        f_function: Function f(q)
        samples: Array of samples (N, 2*dim) for computing expectations
        make_V: Function to create potential V(lam)
        lam: Current lambda value
        max_order: Maximum order of Hermite polynomials
        
    Returns:
        (diagonal, upper_diagonal): Tridiagonal matrix components
    """
    qp_batch = jnp.array(samples)
    dim = qp_batch.shape[1] // 2
    qs = qp_batch[:, :dim]
    
    V = make_V(lam)
    
    # Compute gradients of f(q)
    def f_grad(q):
        return jax.grad(f_function)(q)
    
    def f_hess(q):
        return jax.hessian(f_function)(q)
    
    # Compute V gradient
    def V_grad(q):
        return jax.grad(V)(q)
    
    # Compute expectations over q samples
    f_grad_vals = jax.vmap(f_grad)(qs)
    V_grad_vals = jax.vmap(V_grad)(qs)
    f_vals = jax.vmap(f_function)(qs)
    
    # Compute coefficients
    c0 = jnp.mean(jnp.sum(f_grad_vals**2, axis=1))  # E_q[(∂f/∂q)²]
    
    # c₁ = E_q[(∂f/∂q - f ∂V/∂q)²]
    diff_vals = f_grad_vals - f_vals[:, None] * V_grad_vals
    c1 = jnp.mean(jnp.sum(diff_vals**2, axis=1))
    
    # c₂ = E_q[(∂f/∂q)(∂f/∂q - f ∂V/∂q)]
    c2 = jnp.mean(jnp.sum(f_grad_vals * diff_vals, axis=1))
    
    # Construct tridiagonal matrix for odd sector
    num_coeffs = (max_order + 1) // 2  # Number of odd indices ≤ max_order
    
    diagonal = jnp.array([c0 * (2*k + 2) + c1 * (2*k + 1) for k in range(num_coeffs)])
    upper_diagonal = jnp.array([c2 * jnp.sqrt((2*k + 2) * (2*k + 3)) for k in range(num_coeffs - 1)])
    
    return diagonal, upper_diagonal


def compute_linear_term_coefficient(f_function, samples, make_V, lam):
    """
    Compute the linear term coefficient L_q = E_q[f(q) ∂²V/∂q∂λ].
    
    Args:
        f_function: Function f(q)
        samples: Array of samples (N, 2*dim)
        make_V: Function to create potential V(lam)
        lam: Current lambda value
        
    Returns:
        L_q: Linear term coefficient
    """
    qp_batch = jnp.array(samples)
    dim = qp_batch.shape[1] // 2
    qs = qp_batch[:, :dim]
    
    # Compute ∂²V/∂q∂λ using automatic differentiation
    def d2V_dqdlam(q):
        # Define a function that takes both q and lam as arguments
        def V_with_lam(q, lam):
            V_func = make_V(lam)
            return V_func(q)
        
        # Compute ∂²V/∂q∂λ = ∂/∂λ (∂V/∂q)
        # First compute ∂V/∂q, then take ∂/∂λ of that
        def grad_V_wrt_q(q, lam):
            return jax.grad(V_with_lam, argnums=0)(q, lam)
        
        # Use jacfwd to compute the Jacobian of grad_V_wrt_q with respect to λ
        # This gives us ∂²V/∂q∂λ (the Jacobian of the gradient)
        return jax.jacfwd(grad_V_wrt_q, argnums=1)(q, lam)
    
    f_vals = jax.vmap(f_function)(qs)
    d2V_dqdlam_vals = jax.vmap(d2V_dqdlam)(qs)
    
    # L_q = E_q[f(q) ∂²V/∂q∂λ]
    L_q = jnp.mean(f_vals * d2V_dqdlam_vals)
    
    return L_q


def optimize_hermite_ansatz(hermite_ansatz, samples, make_T, make_V, lam, max_order):
    """
    Optimize the Hermite ansatz using the tridiagonal structure.
    
    This solves the linear system M^(o) α̃^(o) = -b^(o) where:
    - M^(o) is the tridiagonal matrix from construct_hermite_tridiagonal_matrix
    - b^(o) has only the first component non-zero: b^(o)₀ = 2*L_q
    - L_q is the linear term coefficient
    
    Args:
        hermite_ansatz: HermiteAnsatz instance to optimize
        samples: Array of samples (N, 2*dim)
        make_T: Function to create kinetic energy
        make_V: Function to create potential energy  
        lam: Current lambda value
        max_order: Maximum order of Hermite polynomials
        
    Returns:
        Updated hermite_ansatz with optimized coefficients
    """
    # Construct tridiagonal matrix
    diagonal, upper_diagonal = construct_hermite_tridiagonal_matrix(
        hermite_ansatz.f_function, samples, make_V, lam, max_order
    )
    
    # Compute linear term coefficient
    L_q = compute_linear_term_coefficient(hermite_ansatz.f_function, samples, make_V, lam)
    
    # Construct right-hand side vector b^(o)
    num_coeffs = len(hermite_ansatz.alpha_coeffs)
    b_vector = jnp.zeros(num_coeffs)
    b_vector = b_vector.at[0].set(2.0 * L_q)  # Only first component is non-zero
    
    # Solve tridiagonal system M^(o) α̃^(o) = -b^(o)
    # Using scipy's tridiagonal solver
    import scipy.linalg
    alpha_optimized = scipy.linalg.solve_banded(
        (1, 1),  # Upper and lower bandwidth
        jnp.vstack([jnp.concatenate([[0], upper_diagonal]), diagonal, jnp.concatenate([upper_diagonal, [0]])]),
        -b_vector
    )
    
    # Update the ansatz with optimized coefficients
    hermite_ansatz.set_alpha_coeffs(alpha_optimized)
    
    return hermite_ansatz
        