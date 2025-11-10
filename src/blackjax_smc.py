import jax

from datetime import date

import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../data')
sys.path.append('../blackjax')
from src.systems import SYSTEMS
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
from src.utils import save_simulation_data

import matplotlib.pyplot as plt

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params

from functools import partial


def smc_inference_loop(rng_key, smc_kernel, initial_state):
    """Run the temepered SMC algorithm.

    We run the adaptive algorithm until the tempering parameter lambda reaches the value
    lambda=1.

    Returns:
        n_iter: Number of iterations
        final_state: Final SMC state
        snapshots: Dictionary with 'particles', 'lam', 'weights' keys containing
                   snapshots at each intermediate distribution
    """
    snapshots = {'particles': [], 'weights': [], 'times': [], 'lam': []}
    
    # Save initial state
    snapshots['particles'].append(np.array(initial_state.particles))
    snapshots['lam'].append(float(initial_state.lmbda))
    snapshots['times'].append(initial_state.lmbda) # here, lmbda is 1D and same as time
    snapshots['weights'].append(np.array(initial_state.weights))

    state = initial_state
    k = rng_key
    n_iter = 0
    max_iter = 10000  # Safety limit

    while state.lmbda < 1 and n_iter < max_iter:
        k, subk = jax.random.split(k, 2)
        state, _ = smc_kernel(subk, state)
        n_iter += 1

        # jax.debug.print(f"weights: {state.weights}")
        # Save snapshot at each step
        snapshots['particles'].append(np.array(state.particles))
        snapshots['lam'].append(float(state.lmbda))
        snapshots['times'].append(float(state.lmbda))
        snapshots['weights'].append(np.log(np.array(state.weights)))

    return n_iter, state, snapshots

def smc_adjusted_hmc(M, make_V, rng_key, threshold):

    inv_mass_matrix = jnp.eye(1)

    loglikelihood = lambda x: -SYSTEMS['mixture']['make_V'](jnp.array([1.0]))(x)
    prior_log_prob = lambda x: -SYSTEMS['mixture']['make_V'](jnp.array([0.0]))(x)


    hmc_parameters = dict(
        step_size=1e-4, inverse_mass_matrix=inv_mass_matrix, num_integration_steps=1
    )

    tempered = blackjax.adaptive_geometric_smc(
        prior_log_prob,
        loglikelihood,
        blackjax.hmc.build_kernel(),
        blackjax.hmc.init,
        extend_params(hmc_parameters),
        resampling.systematic,
        threshold,
        num_mcmc_steps=1,
    )

    rng_key, init_key, sample_key = jax.random.split(rng_key, 3)
    initial_smc_state = jax.random.multivariate_normal(
        init_key, jnp.zeros([1]), jnp.eye(1), (M,)
    )
    initial_smc_state = tempered.init(initial_smc_state)

    n_iter, smc_samples, snapshots = smc_inference_loop(sample_key, tempered.step, initial_smc_state)
    print("Number of steps in the adaptive algorithm: ", n_iter)
    return smc_samples, snapshots

if __name__ == "__main__":
    
    smc_samples, snapshots = smc_adjusted_hmc(4000, SYSTEMS['mixture']['make_V'], rng_key, threshold=0.5)

    # save snapshots
    save_simulation_data(snapshots, 'mixture', 'smc_adjusted_hmc')

    # print(smc_samples)
    # print(f"Snapshots collected: {len(snapshots['particles'])} intermediate distributions")