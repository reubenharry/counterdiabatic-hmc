import jax
import jax.numpy as jnp
from .physics import m

def make_T_standard(lam):
    """Standard kinetic energy: T(p) = p^2 / (2m)"""
    return lambda p: 0.5 * (p ** 2) / m

def make_V_gaussian_moving_mean(lam):
    """Gaussian potential with moving mean: V(q) = 0.5 * (q - λ)^2"""
    return lambda q: 0.5 * (q - lam) ** 2

def make_V_gaussian_annealing(lam):
    """Gaussian potential with annealing temperature: V(q) = 0.5 * (λ + 0.1) * q^2"""
    return lambda q: 0.5 * (lam + 0.1) * (q ** 2)

def make_V_double_well(lam):
    """Double well potential: V(q) = (1-λ)*0.5*q^2 + λ*(q^2 - 3)^2"""
    return lambda q: (1-lam)*0.5*(q**2) + lam*(q**2 - 3)**2

# Dictionary of available systems
SYSTEMS = {
    'gaussian_moving_mean': {
        'make_T': make_T_standard,
        'make_V': make_V_gaussian_moving_mean,
        'description': 'Gaussian potential with moving mean V(q) = 0.5 * (q - λ)²'
    },
    'gaussian_annealing': {
        'make_T': make_T_standard,
        'make_V': make_V_gaussian_annealing,
        'description': 'Gaussian potential with annealing temperature V(q) = 0.5 * (λ + 0.1) * q²'
    },
    'double_well': {
        'make_T': make_T_standard,
        'make_V': make_V_double_well,
        'description': 'Double well potential V(q) = (1-λ)*0.5*q² + λ*(q² - 3)²'
    }
}

def get_system(system_name):
    """Get a system by name from the SYSTEMS dictionary."""
    if system_name not in SYSTEMS:
        available = list(SYSTEMS.keys())
        raise ValueError(f"Unknown system '{system_name}'. Available systems: {available}")
    
    system = SYSTEMS[system_name]
    return system['make_T'], system['make_V'], system['description'] 