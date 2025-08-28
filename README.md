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
