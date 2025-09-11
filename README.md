# Counterdiabatic Driving with Learned Gauge Potentials

This project explores the use of learned gauge potentials for counterdiabatic driving in Hamiltonian Monte Carlo. It provides a framework for comparing different ansatzes for the gauge potential, including:

- A flexible polynomial ansatz
- A neural network ansatz
- A fixed analytical solution

## Structure

- `main.py`: The main entry point to run simulations: results are generated from here, by choosing a system ('gaussian_moving_mean', 'gaussian_annealing', 'double_well', etc - see systems.py for a list) and a method ('polynomial', 'neural_network')
- `simple_benchmarks`: benchmarking code
- `src/`: Contains the core source code.
  - `ansatze.py`: Defines the different variational ansatzes.
  - `fitting.py`: Contains the code for fitting the ansatz parameters.
  - `physics.py`: Contains physics-related helper functions like the Poisson bracket and leapfrog integrators.
  - `plotting.py`: Contains functions for plotting the results.
  - `simulation.py`: Contains the main simulation loop.
- `requirements.txt`: Project dependencies.

## Usage

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation:
   ```bash
   python main.py
   ```

You can choose the ansatz to use by modifying the `main()` function in `main.py`. 


## Hermite polynomials

We treat the 1D case first. The extension to higher dimensions is straightforward.

Suppose we express $A$ as $A(q,p) = g(p) + f(q)$, where $f(q)$ is a neural net and $g_\alpha(p) = \sum_{i=0}^{\infty} \alpha_i\mathcal{H}_i(p)$.

Here, $\mathcal{H}_i$ are the Hermite polynomials (or some other orthogonal polynomials), with the property that for the metric on function space $\langle a, b \rangle = \int p(x)a(x)b(x)dx$, where $p$ is a unit Gaussian pdf, $\langle \mathcal{H}_i, \mathcal{H}_j\rangle = \delta_{ij}$.

Also note that $\partial_p g = \sum_{i=0}^\infty\partial_p\alpha_i\mathcal{H}_i(p) = \sum_{i=0}^\infty\alpha_i\partial_p(p\mathcal{H}_i(p) + \mathcal{H}_{i+1}(p))$.

Now suppose that we are at step $i$ of the algorithm and are computing $A$ at this step. We treat $f$ as fixed, and write our full loss for minimizing $g_\alpha$ as the functional:

$$
S[\alpha] = \mathbb{E}[||\partial_\lambda H - \{A, H\}||^2] 
$$



$$
= \mathbb{E}[2\partial_\lambda H \sum_i\alpha_i \{f(q)\mathcal{H}_i(p),H\}]
$$

$$
+ \mathbb{E}[\{A,H\}\{A,H\}] + \mathrm{const}(\alpha)
$$

$$
= \mathbb{E}[2 \sum_i\alpha_i f(q)\mathcal{H}_i(p) \{\partial_\lambda H,H\}]
$$

$$
+ \mathbb{E}[(\partial_qfp - \partial_pg\partial_qV)^2] 
$$

$$
= \mathbb{E}[2 \sum_i\alpha_i f(q)\mathcal{H}_i(p) \partial_q\partial_\lambda H p]
$$

$$
+ \mathbb{E}[(\partial_qfp - \sum_{i=0}^\infty\sum_{j=1}^\infty\alpha_i\alpha_j(p^2\mathcal{H}_i\mathcal{H}_j + \mathcal{H}_{i+1}\mathcal{H}_{i+1} + p\mathcal{H}_i\mathcal{H}_{j+1} + p\mathcal{H}_{i+1}\mathcal{H}_j)\partial_qV)]
$$

$$
+ \int \frac{1}{Z}e^{-p^2/2m}2 \sum_i\alpha_i f(q)\mathcal{H}_i(p) \mathcal{H}_0(p)\partial_q\partial_\lambda H  e^{-V(q)}
$$

$$
= \alpha_0 \partial_q\partial_\lambda H  e^{-V(q)}
$$

$$
+ \sum_i todo
$$

We then get a tridiagonal matrix, which we'll also call S, so that we can optimize $g$ by finding $\min_\alpha S\alpha$ todo: formulate as eigenvalue problem

Because this is isomorphic to a hopping problem, as in condensed matter physics, and because it is disordered in precisely the sort of way which one might expect to induce localization, one can argue that only $\alpha_i$ for small $i$ are relevant, so that a low order truncation of $g$ will suffice.
