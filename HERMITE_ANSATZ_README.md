# Hermite Ansatz Implementation

This document describes the implementation of the orthonormal Hermite polynomial ansatz technique for representing the gauge potential A(q,p).

## Mathematical Background

The technique uses the ansatz form:
```
A(q,p) = f(q) * g(p)
```

where:
- `f(q)` is a function of position (can be neural network, polynomial, etc.)
- `g(p) = Σ_{i odd} α̃ᵢ φᵢ(p)` is expanded in orthonormal Hermite polynomials

### Orthonormal Basis

The orthonormal Hermite polynomials are defined as:
```
φᵢ(p) = Hᵢ(p) / √(i!)
```

where `Hᵢ(p)` are the probabilists' Hermite polynomials with recurrence:
```
H₀(p) = 1
H₁(p) = p
H_{n+1}(p) = p*H_n(p) - n*H_{n-1}(p)
```

### Key Properties

1. **Orthonormality**: `E_p[φᵢ(p) φⱼ(p)] = δᵢⱼ` under standard normal distribution
2. **Odd indices only**: Only coefficients for `i = 1, 3, 5, ...` are used
3. **Tridiagonal optimization**: The quadratic form has a tridiagonal structure

## Implementation

### Core Classes

#### `HermiteAnsatz`
```python
class HermiteAnsatz(A_ansatz):
    def __init__(self, f_function=None, max_order=5, dim=1, neural_dims=None, key=None):
        # f_function: Function f(q) that takes position and returns scalar (optional)
        # max_order: Maximum order of Hermite polynomials (must be odd)
        # dim: Dimension of q and p vectors
        # neural_dims: List of layer sizes for neural network f(q) (optional)
        # key: JAX random key for neural network initialization (required if neural_dims provided)
```

#### Key Methods
- `__call__(q, p)`: Evaluate A(q,p) = f(q) * g(p)
- `get_alpha_coeffs()`: Return current Hermite coefficients
- `set_alpha_coeffs(coeffs)`: Set Hermite coefficients
- `get_f_function()` / `set_f_function(f_func)`: Manage f(q) function
- `get_neural_params()` / `set_neural_params(params)`: Manage neural network (if using neural f(q))

### Optimization Functions

#### `construct_hermite_tridiagonal_matrix(f_function, samples, make_V, lam, max_order)`
Constructs the tridiagonal matrix M^(o) for the quadratic form:
```
M^(o)_{k,k} = c₀(2k+2) + c₁(2k+1)
M^(o)_{k,k+1} = c₂√((2k+2)(2k+3))
```

where:
- `c₀ = E_q[(∂f/∂q)²]`
- `c₁ = E_q[(∂f/∂q - f ∂V/∂q)²]`
- `c₂ = E_q[(∂f/∂q)(∂f/∂q - f ∂V/∂q)]`

#### `compute_linear_term_coefficient(f_function, samples, make_V, lam)`
Computes the linear term coefficient:
```
L_q = E_q[f(q) ∂²V/∂q∂λ]
```

#### `optimize_hermite_ansatz(hermite_ansatz, samples, make_T, make_V, lam, max_order)`
Solves the linear system `M^(o) α̃^(o) = -b^(o)` where:
- `b^(o)₀ = 2*L_q`
- `b^(o)ₖ = 0` for k > 0

## Usage Examples

### Basic Usage
```python
from src.ansatze import HermiteAnsatz, optimize_hermite_ansatz

# Define f(q) function
def f_function(q):
    return q[0]  # Linear in q

# Create ansatz
hermite_ansatz = HermiteAnsatz(
    f_function=f_function,
    max_order=5,  # Use φ₁, φ₃, φ₅
    dim=1
)

# Optimize coefficients
hermite_optimized = optimize_hermite_ansatz(
    hermite_ansatz, samples, make_T, make_V, lam, max_order
)

# Evaluate
A_value = hermite_optimized(q, p)
```

### Advanced Usage with Neural Networks
```python
# Create ansatz with neural network f(q)
key = jax.random.PRNGKey(42)
hermite_ansatz = HermiteAnsatz(
    neural_dims=[1, 64, 32, 1],  # Neural network architecture
    max_order=7,
    dim=1,
    key=key
)

# Access neural network parameters
neural_params = hermite_ansatz.get_neural_params()
hermite_ansatz.set_neural_params(updated_params)
```

### Mixed Usage (Analytical + Neural)
```python
# Start with analytical f(q)
def analytical_f(q):
    return q[0] + 0.1 * q[0]**2

hermite_ansatz = HermiteAnsatz(
    f_function=analytical_f,
    max_order=5,
    dim=1
)

# Later switch to neural network f(q)
key = jax.random.PRNGKey(123)
neural_hermite_ansatz = HermiteAnsatz(
    neural_dims=[1, 32, 16, 1],
    max_order=5,
    dim=1,
    key=key
)
```

## Mathematical Advantages

1. **Efficient Optimization**: Tridiagonal structure allows O(n) solution instead of O(n³)
2. **Orthonormal Basis**: Exploits orthogonality properties for numerical stability
3. **Symmetry Exploitation**: Only odd indices needed due to symmetry considerations
4. **Separable Form**: Natural for systems where A(q,p) = f(q) * g(p)

## Files

- `src/ansatze.py`: Core implementation
- `example_hermite_ansatz.py`: Basic usage example
- `example_hermite_fitting.py`: Fitting example with comparison
- `HERMITE_ANSATZ_README.md`: This documentation

## Dependencies

- JAX for automatic differentiation
- SciPy for tridiagonal linear system solving
- Equinox for neural network components

## Notes

- The implementation follows the mathematical formulation exactly as described in the notes
- The tridiagonal structure is preserved for computational efficiency
- Only odd-indexed Hermite polynomials are used, exploiting symmetry
- The orthonormal basis ensures numerical stability
