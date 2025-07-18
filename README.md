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

## The general problem

Suppose we have an SDE targeting a distribution $\rho(x)\propto e^{-U(x)}$, in the sense that the drift term $\mu_\rho(x,t)$ is chosen such that $\partial_t\rho = 0$. 

Then suppose our distribution depends directly on time, so that $\rho_{\lambda(T)}$ is our desired distribution and $\rho_{\lambda(0)}$ is something easy to sample from. If we use $\mu_{\rho(\lambda(t))}(x,t)$ as our drift, then $\partial_t\rho_{\lambda(t)}$ now has an extra term, $\dot\lambda \partial_\lambda \rho_{\lambda(t)}$, so is no longer $0$.

If we could find a drift term $\nu$ such that $\nabla \cdot (\nu \rho_{\lambda(t)}) = -\dot\lambda \partial_\lambda \rho_{\lambda(t)}$, we could use $\mu + \nu$ as our drift and we would have a way to transport efficiency from a simple distribution to the (potentially complicated) target distribution. Various papers try this.

## Notation

Let $\rho_H$ be the canonical distribution, i.e. $\rho_H(q,p) \propto e^{-H}(q,p)$. 

<!-- We write $L_H\rho := \{H, \rho\}$. -->

## The Hamiltonian case

Suppose we want to sample from some $P(q)$. Hamiltonian Monte Carlo works by defining a time-independent Hamiltonian $H$, which we choose so that the marginal $\int \rho_H(q,p)dp = P(q)$, and then running (discretized) Hamiltonian dynamics to take samples from $\rho_H$. (Typically, one resamples momentum for ergodicity, and does MH to remove discretization bias; we pass over these details here).

This in turn works well because the exact Hamiltonian flow preserves the canonical distribution: $\partial_t\rho_H = 0$ and $\phi_H^*(t')\rho_H = \rho_H$, where $\phi$ is the flow of the differential equation.


If samples were initially drawn from $\rho_H$, then exact Hamiltonian dynamics would preserve $\rho_H$ nicely (and discretized Hamiltonian dynamics would only yield small errors, which could be corrected by MH). However, initial samples are typically not drawn from $\rho_H$, but from a normal distribution, which could be very far from $\rho_H$. Adding noise in the correct way ensures eventual convergence to $\rho_H$, but this can be arbitrarily slow.

One solution is to have a time varying $H$, where $\rho_{H(\lambda(0))}$ is a distribution that is easy to sample from, and $\rho_{H(\lambda(T))}$ is the distribution of interest. What we want is an ODE which gives $\partial_t\rho_{H(\lambda(t))}=0$, so that $\phi_{H(\lambda(t))}^*(t')\rho_{H(\lambda(t))} = \rho_{H(t+t')}$


But the naive approach, which is to follow Hamiltonian dynamics, now time dependent, so $\dot z(t) = \Omega \frac{\partial H(z,t)}{\partial z}$, doesn't work. As in the general case, we have 

$$
\partial_t\rho_{H(\lambda(t))} = \{H, \rho_{H(\lambda(t))}\} + \dot\lambda\partial_\lambda \rho_{H(\lambda(t))}
$$

$$
= 0 - \dot\lambda\partial_\lambda H\rho_{H(\lambda(t))} - \dot\lambda\frac{\partial \log Z}{\partial \lambda}\rho_{H(\lambda(t))} = (-\partial_\lambda H + \langle \partial_\lambda H\rangle_t) \dot\lambda\rho_{H(\lambda(t))}
$$

One option is to make $\partial_\lambda H$ small, by changing time very slowly. In the adiabatic limit, this works ([a beautiful paper by Betancourt](https://arxiv.org/abs/1405.3489) shows how to vary $\lambda$ adiabatically), but the problem is that it is very slow. Another option is to take many steps at each value of $t$, to re-equilibrate. This suffers from essentially the original problem: potentially slow thermalization.

Suppose we had a function $A$ such that $\{A, H\} = \partial_\lambda H - \langle \partial_\lambda H\rangle_t$. Then we'd have:

$$
\partial_\lambda\rho_{H(\lambda(t))} = \{H, A\}\rho_{H(\lambda(t))} = \{\rho_{H(\lambda(t))}, -A\}
$$

(We see that if we were to stay (via canonical transformation) in the frame of the moving Hamiltonian, i.e. by taking the total derivative, we would experience an effective Hamiltonian $-A$.)

But then we need only use a Hamiltonian $H - A$ to arrive at $\partial_t\rho_{H(\lambda(t))} = 0$.

Question: where did get the wrong negation sign?

<!-- , in the following sense: $\frac{d}{dt}\rho_H(p(t),q(t))=0$, which implies that $\phi_H^*(t)\rho_H = \rho_H$,  -->
<!-- $\frac{d}{dt}\rho_{H(\lambda(t))}(p(t),q(t))=0$, so that $\phi_{H(\lambda(t))}^*(t')\rho_{H(\lambda(t))} = \rho_{H(t+t')}$. -->

<!-- $$
\frac{d}{dt}\rho_{H(\lambda(t))} = \{\rho_{H(\lambda(t))}, H\} + \frac{\partial \rho_{H(\lambda(t))}}{\partial t}
$$

$$
= 0 - \frac{\partial H}{\partial t}\rho_{H(t)} - \frac{\partial \log Z}{\partial t}\rho_{H(t)} = (-\partial_tH + \langle \partial_tH\rangle_t) \rho_{H(t)}
$$ -->

## Minimization

This idea of counterdiabatic driving was known for some time (via the usual suspects: Rice, Berry, Jarzinsky), but Sels et al came up with a clever way to approximate it variationally.

In classical terms, the key idea is that we minimize 

$$
\langle |G|^2 \rangle_t
$$

at each time point $t$, where 

$$
G = \{A, H\} - \partial_tH + \langle \partial_tH\rangle_t
$$

Multiplying out, we find $\langle |G|^2\rangle = \langle \{A, H\}^2 + (\partial_tH)^2 + \langle \partial_tH\rangle_t^2 + 2\{A, H\}(\partial_tH) - 2\{A, H\}\langle \partial_tH\rangle - 2(\partial_tH)\langle \partial_tH\rangle_t \rangle_t = \langle \{A, H\}^2 + (\partial_tH)^2 - 2\{A, H\}(\partial_tH) + 2\{A, H\}\langle \partial_tH\rangle_t - \langle \partial_tH\rangle_t^2 \rangle_t $ 

But we can see that a minimum of this is a minimum of

$\langle |\{A, H\} - \partial_tH|^2 \rangle = \langle \{A, H\}^2 -2\{A, H\}\partial_tH + (\partial_tH)^2 \rangle$.

This means that we can minimize $\langle |\{A, H\} - \partial_tH|^2 \rangle$ instead, which is pleasantly simple to calculate.


<!-- $L_H\rho_H=0$, so that $e^{-tL_H}\rho_H = \rho_H, \forall t$. -->



<!-- But this too can be very slow. Instead, we might want to change $\lambda$ faster. In this case, the problem is that the samples will lag behind $\rho_{H_\lambda}$. To see this, observe that if $\Phi(\delta t)$ is the time evolution operator by an infinitesimal amount $\delta t$, then $\Phi_{H_{\lambda(t)}}(\delta t)^*\rho_{H_{\lambda(t)}} = \rho_{H_{\lambda(t)}} + O((\delta t)^2)$, whereas the true distribution at time $t+\delta t$ is $\rho_{H_{\lambda(t)}} + \delta t\partial_t\rho_{H_{\lambda(t)}} + O((\delta t)^2)$.

This discrepancy of $\partial_t\rho_{H_{\lambda(t)}}$ can be expressed differently, since we assume $\rho$ is canonical. That is, $\partial_t\rho_{H_{\lambda(t)}} = \frac{\dot Z}{Z}\rho_{H_{\lambda(t)}} - \partial_\lambda H\dot\lambda\rho$.

Now suppose that we have a function on phase space $A$ with $\{A, H\} = \partial_\lambda H$. We rewrite the true evolution as $\rho_{H_{\lambda(t+\delta t)}} = \rho_{H_{\lambda(t)}} + \delta t\partial_t\rho_{H_{\lambda(t)}} + O((\delta t)^2) = \rho_{H_{\lambda(t)}} + \delta t\dot\lambda \{A,\rho_{H_{\lambda(t)}}\}+ O((\delta t)^2) \approx (I + \delta t\dot\lambda L_A)\rho_{H_{\lambda(t)}}$.

(where we have used that $\{f,g\} = \frac{df}{dr}\{r, g\}$, as is evident from direct calculation).

The intuitive solution is then to construct a new Hamiltonian which accounts for the discrepancy. This will take the form $H'_{\lambda(t)} := H_{\lambda(t)} + \dot\lambda A$


$$
\Phi_{H_{\lambda(t)}+ \dot\lambda A}(\delta t)^*\rho_{H_{\lambda(t)}} = \rho_{H_{\lambda(t)}} + \dot\lambda\{A, \rho_{H_{\lambda(t)}}\} + O((\delta t)^2)
$$

which gives us the desired evolution. -->


<!-- $L_{H_{\lambda(t)}}\rho_{\lambda(t)} = 0$, so that $e^{-tL_H}\rho_H = \rho_H$ -->


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

