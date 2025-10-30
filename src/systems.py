import math
import jax
import jax.numpy as jnp


m = 1.0

def make_T_standard(lam):
    """Standard kinetic energy: T(p) = p^2 / (2m)"""
    return lambda p: 0.5 * jnp.sum(p ** 2) / m

def make_V_gaussian_moving_mean(lam):
    """Gaussian potential with moving mean: V(q) = 0.5 * (q - λ)^2"""
    return lambda q: 0.5 * jnp.sum((q - lam) ** 2)

def make_V_gaussian_annealing(lam):
    """Gaussian potential with annealing temperature: V(q) = 0.5 * k(λ) * q^2 where k interpolates from 1 (var=1) to 10 (var=0.1)"""
    # k = 1 at λ=0 (var=1), k = 10 at λ=1 (var=0.1)
    k = 0.5 + 9.5 * lam
    return lambda q: 0.5 * jnp.sum(k * (q ** 2))

# construct a geometric interpolation between a gaussian and a final potential
def make_V_geometric_potential(final_potential, initial_sigma):
    # log normal pdf
    log_normal_pdf = lambda x, mu, sigma: jnp.sum(0.5*((x/sigma)**2))
    def make_V(lam):
        return lambda q: log_normal_pdf(q, 0, (initial_sigma)) * (1-lam) + final_potential(q) * lam
        # return lambda q: final_potential(q) * lam
        # return lambda q: jnp.sum((q**2 - 2)**2)
    return make_V

# make_V_double_well = lambda lam: lambda q: jnp.sum((q**2 - 2)**2)

double_well_potential = lambda q: jnp.sum((q**2 - 2)**2)
# make_V_double_well = make_V_geometric_potential(final_potential=double_well_potential, initial_sigma=1.0)

# def make_V_double_well(lam):
    
#     """Double well potential: V(q) = (1-λ)*0.5*q^2 + λ*(q^2 - 3)^2"""
#     # return lambda q: jnp.sum((1-lam)*0.5*(q**2) + lam*(((q-2)**2 - 3)**2))
#     return lambda q: (1-lam)*jnp.sum(0.5*(q**2)) + lam*jnp.sum((((q*0.5)**2-2))**2)
#     # return lambda q: jnp.sum((1-lam)*0.5*(q**2) + lam*(((q)**2 - 3)**2))

def make_V_2d_gaussian_moving_mean(lam):
    """2D Gaussian potential with moving mean: V(q) = 0.5 * ||q - λ||^2"""
    return lambda q: 0.5 * jnp.sum((q - lam) ** 2)

def make_V_2d_double_well(lam):
    """2D Double well potential: V(q) = (1-λ)*0.5*||q||^2 + λ*((q_0^2 - 3)^2 + (q_1^2 - 3)^2)"""
    return lambda q: jnp.sum((1-lam)*0.5*(q**2) + lam*((q[0]**2 - 3)**2 + (q[1]**2 - 3)**2))

def make_V_2d_gaussian_annealing(lam):
    """2D Gaussian potential with annealing temperature: V(q) = 0.5 * (λ + 0.1) * ||q||^2"""
    return lambda q: 0.5 * jnp.sum((lam + 0.1) * (q ** 2))

def make_V_2d_rosenbrock(lam):
    """2D Rosenbrock potential: V(q) = (1-λ)*0.5*||q||^2 + λ*((1-q_0)^2 + 100*(q_1-q_0^2)^2)"""
    return lambda q: jnp.sum((1-lam)*0.5*(q**2) + 0.5*(lam*((q[0]-1)**2 + ((q[0]**2) - q[1])**2)))

# Dictionary of available systems
SYSTEMS = {
    'gaussian_moving_mean': {
        'make_T': make_T_standard,
        'make_V': make_V_gaussian_moving_mean,
        'description': 'Gaussian potential with moving mean V(q) = 0.5 * (q - λ)²',
        'dim': 1,
        'initial_sigma': 1.0
    },
    'gaussian_annealing': {
        'make_T': make_T_standard,
        'make_V': make_V_gaussian_annealing,
        'description': 'Gaussian potential with annealing temperature V(q) = 0.5 * k(λ) * q² where k interpolates from 1 (var=1) to 10 (var=0.1)',
        'dim': 1,
        'initial_sigma': math.sqrt(2.0)
    },
    'double_well': {
        'make_T': make_T_standard,
        'make_V': make_V_geometric_potential(final_potential=double_well_potential, initial_sigma=2.0),
        'description': 'Double well potential V(q) = (1-λ)*0.5*q² + λ*(q² - 3)²',
        'dim': 1,
        'initial_sigma': 2.0
    },
    '2d_gaussian_moving_mean': {
        'make_T': make_T_standard,
        'make_V': make_V_2d_gaussian_moving_mean,
        'description': '2D Gaussian potential with moving mean V(q) = 0.5 * ||q - λ||²',
        'dim': 2,
        'initial_sigma': 1.0
    },
    '2d_double_well': {
        'make_T': make_T_standard,
        'make_V': make_V_2d_double_well,
        'description': '2D Double well potential V(q) = (1-λ)*0.5*||q||² + λ*((q₀² - 3)² + (q₁² - 3)²)',
        'dim': 2, 
        'initial_sigma': 1.0
    },
    '2d_gaussian_annealing': {
        'make_T': make_T_standard,
        'make_V': make_V_2d_gaussian_annealing,
        'description': '2D Gaussian potential with annealing temperature V(q) = 0.5 * (λ + 0.1) * ||q||²',
        'dim': 2,
        'initial_sigma': 1.0
    },
    '2d_normal_to_rosenbrock': {
        'make_T': make_T_standard,
        'make_V': make_V_2d_rosenbrock,
        'description': '2D interpolating potential: normal → Rosenbrock V(q) = (1-λ)*0.5*||q||² + λ*((1-q₀)² + 100*(q₁-q₀²)²)',
        'dim': 2,
        'initial_sigma': 1.0
    }
}

def get_system(system_name):
    """Get a system by name from the SYSTEMS dictionary."""
    if system_name not in SYSTEMS:
        available = list(SYSTEMS.keys())
        raise ValueError(f"Unknown system '{system_name}'. Available systems: {available}")
    
    system = SYSTEMS[system_name]
    return system['make_T'], system['make_V'], system['description'], system['dim'], system['initial_sigma'] 