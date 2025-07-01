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

## Notation

Let $\rho_H$ be the canonical distribution, i.e. $\rho_H(q,p) \propto e^{-H}(q,p)$. We write $L_H\rho := \{H, \rho\}$.

## The problem

Suppose we want to sample from some $P(q)$. Hamiltonian Monte Carlo works by defining a time-independent Hamiltonian $H$, which we choose so that the marginal $\int \rho_H(q,p)dp = P(q)$, and then running (discretized) Hamiltonian dynamics to take samples from $\rho_H$.

This in turn works well because the exact Hamiltonian flow preserves the stationary distribution, in the following sense: $L_H\rho_H=0$, so that $e^{-tL_H}\rho_H = \rho_H, \forall t$.

If samples were initially drawn from $\rho_H$, then exact Hamiltonian dynamics would preserve $\rho_H$ nicely (and discretized Hamiltonian dynamics would only yield small errors, which could be corrected by MH).

However, samples are typically not drawn from $\rho_H$, but from a normal distribution, which could be very far from $\rho_H$. Adding noise in the correct way ensures eventual convergence to $\rho_H$, but this can be arbitrarily slow.

One solution is to "anneal" through a series of distributions $\rho_{H_\lambda}$, equilibrating to each. One can think of this as adiabatic Hamiltonian Monte Carlo, in the sense that one has to vary $\lambda$ only very slowly, in order that after each change of $\lambda$, enough dynamics happen for the samples to re-equilibrate to the new distribution. In fact, [a beautiful paper by Betancourt](https://arxiv.org/abs/1405.3489) shows how to vary $\lambda$ adiabatically.

But this too can be very slow. Instead, we might want to change $\lambda$ faster. In this case, the problem is that the samples will lag behind $\rho_{H_\lambda}$. To see this, observe that if $\Phi(\delta t)$ is the time evolution operator by an infinitesimal amount $\delta t$, then $\Phi_{H_{\lambda(t)}}(\delta t)^*\rho_{H_{\lambda(t)}} = \rho_{H_{\lambda(t)}} + O((\delta t)^2)$, whereas the true distribution at time $t+\delta t$ is $\rho_{H_{\lambda(t)}} + \delta t\partial_t\rho_{H_{\lambda(t)}} + O((\delta t)^2)$.

This discrepancy of $\partial_t\rho_{H_{\lambda(t)}}$ can be expressed differently, since we assume $\rho$ is canonical. That is, $\partial_t\rho_{H_{\lambda(t)}} = \frac{\dot Z}{Z}\rho_{H_{\lambda(t)}} - \partial_\lambda H\dot\lambda\rho$.

Now suppose that we have a function on phase space $A$ with $\{A, H\} = \partial_\lambda H$. We rewrite the true evolution as $\rho_{H_{\lambda(t+\delta t)}} = \rho_{H_{\lambda(t)}} + \delta t\partial_t\rho_{H_{\lambda(t)}} + O((\delta t)^2) = \rho_{H_{\lambda(t)}} + \delta t\dot\lambda \{A,\rho_{H_{\lambda(t)}}\}+ O((\delta t)^2) \approx (I + \delta t\dot\lambda L_A)\rho_{H_{\lambda(t)}}$.

(where we have used that $\{f,g\} = \frac{df}{dr}\{r, g\}$, as is evident from direct calculation).

The intuitive solution is then to construct a new Hamiltonian which accounts for the discrepancy. This will take the form $H'_{\lambda(t)} := H_{\lambda(t)} + \dot\lambda A$


$$
\Phi_{H_{\lambda(t)}+ \dot\lambda A}(\delta t)^*\rho_{H_{\lambda(t)}} = \rho_{H_{\lambda(t)}} + \dot\lambda\{A, \rho_{H_{\lambda(t)}}\} + O((\delta t)^2)
$$

which gives us the desired evolution.


<!-- $L_{H_{\lambda(t)}}\rho_{\lambda(t)} = 0$, so that $e^{-tL_H}\rho_H = \rho_H$ -->


## Solution

Finding $A$ is hard in practice. In fact, it isn't entirely obvious to me that it always exists in the classical setting. But one can tackle the problem variationally, but taking e.g. a polynomial ansatz and minimizing it at regular intervals.

In particular, we assume we are in a multichain setting, so at every time point, we have $M$ particles.

Defining $R = |\partial_\lambda H - \{A, H\}|^2$, our variational problem is then to vary $A$ in order to minimize $\langle R \rangle$, as estimated by our current set of particles.

More concretely, we parametrize $A$ as a sum of polynomials, so $A(q,p) = \theta_1 p + \theta_2 pq + \ldots$

$$

$$

$$
todo
$$

Pseudocode:

```python
for i in 
```


<!-- Standard Hamiltonian Monte Carlo is based on the fact that a time-independent Hamiltonian $H$ satisfies $\partial_t\rho_H = 0$, i.e.  -->

## Questions

Does the condition have to be \partial_\lambda H = {A, H}, or just that this is constant of motion?

How should we integrate the new Hamiltonian? In any case of interest, it is non-separable, but this means that our symplectic integrators do not have the desired behavior.

Shoud we recompute A at the midpoint $\lambda(t + 0.5\delta t)$?

How should we tune the step size of HMC?

How should we tune the hyperparams

Should we be using weighting schemes, or indeed MH, to correct for discretization?

There is presumably gauge freedom in the choice of $A$. How should we gauge fix?

Should we leapfrog update $\lambda$?

Should we consider multiple dimensions in the base manifold? More interesting geometry...

Hannay angles?

notes:

   no even terms like p^2


## Geometric perspective


## Appendix

Suppose $\rho(q,p,t) \propto e^{-\beta H(q,p,t)}$. Then:

$$\frac{d}{dt}\rho(q(t), p(t), t) = \dot q\partial_q\rho + \dot p \partial_p\rho + \partial_t\rho$$

$$
= \{q, H\}\partial_q\rho + `\{p, H\}` \partial_p\rho + \partial_t\rho
$$

$$
= \{\rho, H\} + \partial_t\rho
$$

Defining $L_H\rho = \{H, \rho\}$, we see that $L_H\rho = \partial_t\rho$ is sufficient for 

$$
\rho(q(t), p(t), t) = \rho(q(0), p(0), 0)
$$


<!-- that PDE does not by itself fix AA uniquely—you can always add any function F(H0)F(H0​) of the original Hamiltonian and still satisfy it, because {F(H0),H0}=0{F(H0​),H0​}=0 -->

