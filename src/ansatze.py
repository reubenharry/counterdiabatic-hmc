import jax
import jax.numpy as jnp
import equinox as eqx
import itertools

def check_nans(name, value):
    """Helper function to check for NaNs and print warnings."""
    # Use JAX operations to avoid tracing issues
    has_nan = jnp.isnan(value).any()
    count = jnp.isnan(value).sum()
    
    # Use jax.debug.print for JAX-compatible printing
    jax.debug.print("⚠️  NaN detected in {} (count: {})", name, count)
    return has_nan

class A_ansatz(eqx.Module):
    """Base class for gauge potential ansatz."""
    def __call__(self, q, p):
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

        # return p
        
        # For the 'gaussian_annealing' system, V(q) = 0.5 * (lam_schedule + 0.1) * q^2.
        # The parameter in the analytic formula corresponds to the coefficient of 0.5*q^2.
        # potential_param = lam_schedule + 0.001
        
        # Formula from screenshot: A = (p^2 + L*q^2) / (4*L*sqrt(L)) * arctan(p / (q*sqrt(L)))
        # where L is the potential parameter.
        # numerator = p**2 + potential_param * q**2
        # denominator = 4 * potential_param**(1.5)
        
        # Use arctan2 to handle q=0 case
        # arctan_term = jnp.arctan2(p, jnp.sqrt(potential_param) * q)
        
        # return (numerator / denominator) * arctan_term

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