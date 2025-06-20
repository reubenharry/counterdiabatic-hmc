import jax
import jax.numpy as jnp
import equinox as eqx

class A_ansatz(eqx.Module):
    """Base class for gauge potential ansatz."""
    def __call__(self, q, p):
        raise NotImplementedError

def generate_polynomial_terms(max_degree):
    """Generate all polynomial terms up to max_degree in p and q.
    
    Returns a list of tuples (coeff_name, q_power, p_power) representing terms like:
    - (θ1, 0, 1) for p
    - (θ2, 1, 1) for p*q
    - (θ3, 2, 0) for q^2
    etc.
    """
    terms = []
    term_idx = 1
    
    for total_degree in range(max_degree + 1):
        for q_power in range(total_degree + 1):
            p_power = total_degree - q_power
            terms.append((f"θ{term_idx}", q_power, p_power))
            term_idx += 1
    
    return terms

class PolynomialAnsatz(A_ansatz):
    """Polynomial ansatz for the gauge potential."""
    params: jnp.ndarray
    terms: list = eqx.static_field()
    ansatz_type: str = eqx.static_field()

    def __init__(self, max_degree):
        self.terms = generate_polynomial_terms(max_degree)
        self.params = jnp.zeros(len(self.terms))
        self.ansatz_type = 'polynomial'

    def __call__(self, q, p):
        result = 0.0
        for i, (_, q_power, p_power) in enumerate(self.terms):
            result += self.params[i] * (q ** q_power) * (p ** p_power)
        return result

    def get_term_description(self):
        """Return a description of what each parameter represents."""
        descriptions = []
        for coeff_name, q_power, p_power in self.terms:
            term_str = ""
            if q_power > 0:
                term_str += f"q^{q_power}" if q_power > 1 else "q"
            if p_power > 0:
                term_str += f"p^{p_power}" if p_power > 1 else "p"
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

        # return p
        
        # For the 'gaussian_annealing' system, V(q) = 0.5 * (lam_schedule + 0.1) * q^2.
        # The parameter in the analytic formula corresponds to the coefficient of 0.5*q^2.
        potential_param = lam_schedule + 0.001
        
        # Formula from screenshot: A = (p^2 + L*q^2) / (4*L*sqrt(L)) * arctan(p / (q*sqrt(L)))
        # where L is the potential parameter.
        numerator = p**2 + potential_param * q**2
        denominator = 4 * potential_param**(1.5)
        
        # Use arctan2 to handle q=0 case
        arctan_term = jnp.arctan2(p, jnp.sqrt(potential_param) * q)
        
        return (numerator / denominator) * arctan_term

class NeuralNetworkAnsatz(A_ansatz):
    """Neural network ansatz for the gauge potential."""
    layers: list
    ansatz_type: str = eqx.static_field()

    def __init__(self, dims, key):
        # dims: list of layer sizes, e.g. [2, 64, 32, 1]
        keys = jax.random.split(key, len(dims) - 1)
        self.ansatz_type = 'neural'
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i+1], key=keys[i])
            for i in range(len(dims) - 1)
        ]

    def __call__(self, q, p):
        # Stack q and p for input to MLP
        x = jnp.stack([q, p], axis=-1)
        for layer in self.layers[:-1]:
            x = (layer)(x)
            x = jax.nn.relu(x)
        x = (self.layers[-1])(x)
        return x.squeeze() 