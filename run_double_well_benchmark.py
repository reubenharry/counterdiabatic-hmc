#!/usr/bin/env python3
"""
Script to run double well benchmark using the exact values provided.
"""

from benchmarks import run_benchmark_comparison

def main():
    """Run double well benchmark."""
    print("Running Double Well Benchmark")
    print("=" * 50)
    print("True values at λ=1:")
    print("  E[x²] = 7.3413954")
    print("  Var[x²] = 8.847134")
    print("=" * 50)
    
    # Run benchmark with double well system
    results, summary = run_benchmark_comparison(
        system_name="double_well",
        M=1000,           # Number of particles
        N_steps=40,       # Number of simulation steps
        delta_t=0.05,     # Time step
        eps=0.05,         # HMC step size
        num_trials=5,     # Number of trials (reduced for faster testing)
        ess_threshold=0.5 # Effective sample size threshold
    )
    
    print("\nDouble Well Benchmark completed!")
    print("Check benchmark_results/ for detailed results and plots.")

if __name__ == "__main__":
    main()
