# Counterdiabatic Driving with Learned Gauge Potentials

This project explores the use of learned gauge potentials for counterdiabatic driving in Hamiltonian Monte Carlo. It provides a framework for comparing different ansatzes for the gauge potential, including:

- A flexible polynomial ansatz
- A neural network ansatz
- A fixed analytical solution

## Structure

- `main.py`: The main entry point to run simulations.
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

## Notes

## Deriving the condition on $A$

Suppose our (classical) Hamiltonian has the form $H_{C}(q,p,t) = H_0(q,p,t) + \dot \lambda A(q,p,t)$.

We have (Liouville's theorem):

$$
\partial_t\rho + \{\rho, H_C\} = 0
$$

which holds for the evolution of any measure under Hamiltonian dynamics (as a consequence of the continuity theorem, which follows from the measure preservation of Hamiltonian dynamics, and then some further properties of the Poisson bracket).

where $\rho(q,p,t) \propto e^{-\beta H_0(q,p,t)}$ is the canonical distribution with respect to $H_0$. Expanding this, we find:

$$
-\beta \dot\lambda(t)\partial_\lambda H_0\rho(q,p,t) + g(\lambda(t)) + \dot\lambda\rho
(q,p,t)(-\beta \{H_0, A\}) = 0
$$

so that when $\dot\lambda$ is finite, $\{\partial_\lambda H_0 - \{A, H_0\}, H_0\} = 0$.

The approach of Sels et al is to minimize $A$ variationally

What we want is that $\rho$ is 

Dividing by λ˙λ˙ (assuming it’s nonzero) yields the first‐order linear PDE for the : but ldot can be 0!

 hat PDE does not by itself fix AA uniquely—you can always add any function F(H0)F(H0​) of the original Hamiltonian and still satisfy it, because {F(H0),H0}=0{F(H0​),H0​}=0

