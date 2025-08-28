#!/usr/bin/env python3
"""
Standalone script to plot 2D Rosenbrock simulation results.
This loads existing data and creates visualization plots.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.systems import get_system

def plot_2d_distributions():
    """Load and plot 2D Rosenbrock simulation results."""
    
    # Get the system functions
    make_T, make_V, system_description, dim = get_system("2d_normal_to_rosenbrock")
    print(f"System: {system_description}")
    print(f"Dimension: {dim}")
    
    # Load all simulation data
    methods = ['naive_unweighted', 'naive_weighted', 'cd_unweighted', 'cd_weighted']
    data = {}
    
    for method in methods:
        filename = f"data/2d_normal_to_rosenbrock_{method}.pkl"
        try:
            with open(filename, 'rb') as f:
                data[method] = pickle.load(f)
            print(f"✓ Loaded {method}")
        except FileNotFoundError:
            print(f"✗ Missing {method}")
    
    # Create a comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('2D Rosenbrock: Normal → Rosenbrock Transition', fontsize=16)
    
    colors = {
        'naive_unweighted': 'blue',
        'naive_weighted': 'green', 
        'cd_unweighted': 'red',
        'cd_weighted': 'orange'
    }
    
    # Plot final distributions for each method
    for i, method in enumerate(methods):
        if method not in data:
            continue
            
        ax = axes[i//2, i%2]
        snapshots = data[method]['snapshots']  # Access the correct structure
        
        # Get the final snapshot
        if 'particles' in snapshots and len(snapshots['particles']) > 0:
            final_particles = snapshots['particles'][-1]
            final_time = snapshots['times'][-1] if 'times' in snapshots else 0.0
            final_lam = snapshots['lam'][-1] if 'lam' in snapshots else 0.0
            
            # Plot particles
            ax.scatter(final_particles[:, 0], final_particles[:, 1], 
                      alpha=0.6, s=10, color=colors[method], label=f'{method.replace("_", " ")}')
            
            # Add weights if available
            if 'weights' in snapshots and len(snapshots['weights']) > 0:
                weights = snapshots['weights'][-1]
                if weights is not None and not np.allclose(weights, 0.0):
                    # Normalize weights for visualization
                    weights_norm = np.exp(weights - np.max(weights))
                    weights_norm = weights_norm / np.sum(weights_norm)
                    # Use alpha based on weights
                    alphas = 0.3 + 0.7 * weights_norm
                    ax.scatter(final_particles[:, 0], final_particles[:, 1], 
                              alpha=alphas, s=10, color=colors[method], label=f'{method.replace("_", " ")} (weighted)')
            
            # Plot the Rosenbrock function contour
            x = np.linspace(-2, 2, 100)
            y = np.linspace(-1, 3, 100)
            X, Y = np.meshgrid(x, y)
            
            # Create the Rosenbrock function
            rosenbrock = lambda x, y: (1-x)**2 + 100*(y-x**2)**2
            
            # Plot contours
            Z = rosenbrock(X, Y)
            levels = np.logspace(-1, 2, 10)
            ax.contour(X, Y, Z, levels=levels, colors='black', alpha=0.5, linewidths=0.5)
            
            ax.set_xlabel('q₀')
            ax.set_ylabel('q₁')
            ax.set_title(f'{method.replace("_", " ").title()}\nt={final_time:.2f}, λ={final_lam:.3f}')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('figures/polynomial/2d_rosenbrock_final_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create evolution plot showing how particles move over time
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('2D Rosenbrock: Evolution Over Time', fontsize=16)
    
    # Use CD unweighted for evolution (has more detailed snapshots)
    if 'cd_unweighted' in data:
        snapshots = data['cd_unweighted']['snapshots']  # Access the correct structure
        times = snapshots['times'] if 'times' in snapshots else []
        particles_list = snapshots['particles'] if 'particles' in snapshots else []
        
        # Plot every other snapshot to fit in 10 subplots
        step = max(1, len(particles_list) // 10)
        
        for i in range(min(10, len(particles_list))):
            idx = i * step
            if idx >= len(particles_list):
                break
                
            ax = axes[i//5, i%5]
            particles = particles_list[idx]
            time = times[idx] if idx < len(times) else 0.0
            lam = snapshots['lam'][idx] if idx < len(snapshots['lam']) else 0.0
            
            # Plot particles
            ax.scatter(particles[:, 0], particles[:, 1], alpha=0.6, s=5, color='red')
            
            # Plot Rosenbrock contour
            x = np.linspace(-2, 2, 50)
            y = np.linspace(-1, 3, 50)
            X, Y = np.meshgrid(x, y)
            rosenbrock = lambda x, y: (1-x)**2 + 100*(y-x**2)**2
            Z = rosenbrock(X, Y)
            levels = np.logspace(-1, 2, 5)
            ax.contour(X, Y, Z, levels=levels, colors='black', alpha=0.5, linewidths=0.5)
            
            ax.set_xlabel('q₀')
            ax.set_ylabel('q₁')
            ax.set_title(f't={time:.2f}, λ={lam:.3f}')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/polynomial/2d_rosenbrock_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('2D Rosenbrock: Method Comparison', fontsize=16)
    
    # Plot 1: Final particle positions
    ax1 = axes[0, 0]
    for method in methods:
        if method in data:
            snapshots = data[method]['snapshots']  # Access the correct structure
            if 'particles' in snapshots and len(snapshots['particles']) > 0:
                final_particles = snapshots['particles'][-1]
                ax1.scatter(final_particles[:, 0], final_particles[:, 1], 
                           alpha=0.6, s=10, color=colors[method], label=method.replace('_', ' '))
    
    # Add Rosenbrock minimum
    ax1.scatter([1], [1], color='black', s=100, marker='*', label='Rosenbrock min (1,1)')
    ax1.set_xlabel('q₀')
    ax1.set_ylabel('q₁')
    ax1.set_title('Final Particle Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Energy derivative over time
    ax2 = axes[0, 1]
    for method in methods:
        if method in data:
            snapshots = data[method]['snapshots']  # Access the correct structure
            if 'detailed_energy_stats' in snapshots and 'times' in snapshots:
                energy_stats = snapshots['detailed_energy_stats']
                times = snapshots['times']
                dH_dlam_vals = [stats.get('avg_dH_dlam', 0) for stats in energy_stats]
                ax2.plot(times, dH_dlam_vals, color=colors[method], label=method.replace('_', ' '), linewidth=2)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('<∂H/∂λ>')
    ax2.set_title('Energy Derivative')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Energy variance over time
    ax3 = axes[1, 0]
    for method in methods:
        if method in data:
            snapshots = data[method]['snapshots']  # Access the correct structure
            if 'detailed_energy_stats' in snapshots and 'times' in snapshots:
                energy_stats = snapshots['detailed_energy_stats']
                times = snapshots['times']
                dH2_vals = [stats.get('avg_delta_H_sq', 0) for stats in energy_stats]
                ax3.plot(times, dH2_vals, color=colors[method], label=method.replace('_', ' '), linewidth=2)
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('<ΔH²>')
    ax3.set_title('Energy Change Variance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Var[A] for CD methods
    ax4 = axes[1, 1]
    for method in ['cd_unweighted', 'cd_weighted']:
        if method in data:
            snapshots = data[method]['snapshots']  # Access the correct structure
            if 'detailed_energy_stats' in snapshots and 'times' in snapshots:
                energy_stats = snapshots['detailed_energy_stats']
                times = snapshots['times']
                var_A_vals = [stats.get('var_A', 0) for stats in energy_stats]
                ax4.plot(times, var_A_vals, color=colors[method], label=method.replace('_', ' '), linewidth=2)
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Var[A]')
    ax4.set_title('Gauge Potential Variance (CD only)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/polynomial/2d_rosenbrock_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created 2D Rosenbrock distribution plots")
    print("📁 Plots saved to: figures/polynomial/")
    print("   - 2d_rosenbrock_final_distributions.png")
    print("   - 2d_rosenbrock_evolution.png")
    print("   - 2d_rosenbrock_method_comparison.png")

if __name__ == "__main__":
    plot_2d_distributions()
