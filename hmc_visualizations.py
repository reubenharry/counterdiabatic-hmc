import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal
import seaborn as sns
import matplotlib.animation as animation

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def hamiltonian(q, p, m=1.0, k=1.0):
    """1D harmonic oscillator Hamiltonian: H = p^2/(2m) + (k/2)q^2"""
    return p**2/(2*m) + (k/2)*q**2

def hamiltonian_gradients(q, p, m=1.0, k=1.0):
    """Gradients of the Hamiltonian"""
    dH_dq = k * q
    dH_dp = p / m
    return dH_dq, dH_dp

def exact_dynamics(t, state, m=1.0, k=1.0):
    """Exact Hamiltonian dynamics for harmonic oscillator"""
    q, p = state
    omega = np.sqrt(k/m)
    q_exact = q * np.cos(omega * t) + (p/m/omega) * np.sin(omega * t)
    p_exact = p * np.cos(omega * t) - m * omega * q * np.sin(omega * t)
    return np.array([q_exact, p_exact])

def leapfrog_step(q, p, eps, m=1.0, k=1.0):
    """Single leapfrog step"""
    # Half step in momentum
    p_half = p - (eps/2) * k * q
    
    # Full step in position
    q_new = q + eps * p_half / m
    
    # Half step in momentum
    p_new = p_half - (eps/2) * k * q_new
    
    return q_new, p_new

def generate_energy_contours(q_range, p_range, m=1.0, k=1.0, n_points=100):
    """Generate energy level sets"""
    q = np.linspace(q_range[0], q_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    Q, P = np.meshgrid(q, p)
    
    H = hamiltonian(Q, P, m, k)
    return Q, P, H

def plot_comprehensive_hmc():
    """Comprehensive HMC visualization with all components"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hamiltonian Monte Carlo: Comprehensive Visualization', 
                 fontsize=18, fontweight='bold')
    
    # Parameters
    m, k = 1.0, 1.0
    q_range = (-3, 3)
    p_range = (-3, 3)
    Q, P, H = generate_energy_contours(q_range, p_range, m, k)
    levels = np.linspace(0.5, 4.5, 9)
    
    # Plot 1: Phase Space Trajectories (Exact vs Discretized)
    ax1 = axes[0, 0]
    
    q0, p0 = 2.0, 1.5
    eps = 0.8
    n_steps = 8
    
    # Energy contours
    contour = ax1.contour(Q, P, H, levels=levels, colors='gray', alpha=0.6, linewidths=1)
    ax1.clabel(contour, inline=True, fontsize=8)
    
    # Exact trajectory
    t_exact = np.linspace(0, n_steps * eps, 1000)
    exact_traj = np.array([exact_dynamics(t, [q0, p0], m, k) for t in t_exact])
    ax1.plot(exact_traj[:, 0], exact_traj[:, 1], 'b-', linewidth=3, 
             label='Exact dynamics', alpha=0.8)
    
    # Leapfrog trajectory
    q_lf, p_lf = q0, p0
    lf_traj = [(q_lf, p_lf)]
    for i in range(n_steps):
        q_lf, p_lf = leapfrog_step(q_lf, p_lf, eps, m, k)
        lf_traj.append((q_lf, p_lf))
    
    lf_traj = np.array(lf_traj)
    ax1.plot(lf_traj[:, 0], lf_traj[:, 1], 'ro-', linewidth=3, markersize=10,
             label='Leapfrog steps', alpha=0.8)
    
    ax1.plot(q0, p0, 'go', markersize=15, label='Initial point')
    ax1.set_xlabel('Position (q)', fontsize=12)
    ax1.set_ylabel('Momentum (p)', fontsize=12)
    ax1.set_title('Phase Space Trajectories', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Energy Conservation
    ax2 = axes[0, 1]
    
    H_exact = [hamiltonian(q, p, m, k) for q, p in exact_traj]
    ax2.plot(t_exact, H_exact, 'b-', linewidth=3, label='Exact dynamics')
    
    t_lf = np.arange(n_steps + 1) * eps
    H_lf = [hamiltonian(q, p, m, k) for q, p in lf_traj]
    ax2.plot(t_lf, H_lf, 'ro-', linewidth=3, markersize=8, label='Leapfrog')
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Hamiltonian H(q,p)', fontsize=12)
    ax2.set_title('Energy Conservation', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Measure Preservation with Momentum Resampling
    ax3 = axes[0, 2]
    
    eps_mp = 1.2
    n_grid = 15
    q_grid = np.linspace(1.5, 2.5, n_grid)
    p_grid = np.linspace(1.0, 2.0, n_grid)
    Q_grid, P_grid = np.meshgrid(q_grid, p_grid)
    
    # Transform points through multiple steps with momentum resampling
    Q_curr, P_curr = Q_grid.copy(), P_grid.copy()
    
    for step in range(30):
        # Leapfrog step
        for i in range(n_grid):
            for j in range(n_grid):
                Q_curr[i, j], P_curr[i, j] = leapfrog_step(Q_curr[i, j], P_curr[i, j], eps_mp, m, k)
        
        # Momentum resampling every 10 steps
        if (step + 1) % 10 == 0:
            P_curr = np.random.normal(0, np.sqrt(m), P_curr.shape)
    
    contour = ax3.contour(Q, P, H, levels=levels, colors='gray', alpha=0.4, linewidths=1)
    ax3.plot(Q_grid.flatten(), P_grid.flatten(), 'bo', markersize=4, alpha=0.7, label='Initial')
    ax3.plot(Q_curr.flatten(), P_curr.flatten(), 'ro', markersize=4, alpha=0.7, label='After 30 steps')
    
    ax3.set_xlabel('Position (q)', fontsize=12)
    ax3.set_ylabel('Momentum (p)', fontsize=12)
    ax3.set_title('Measure Preservation + Resampling', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Plot 4: Stationary Distribution Symmetry
    ax4 = axes[1, 0]
    
    # Generate 20000 samples from canonical distribution
    np.random.seed(42)
    n_points = 20000
    
    q_samples = np.random.normal(0, 1/np.sqrt(k), n_points)
    p_samples = np.random.normal(0, np.sqrt(m), n_points)
    
    # Plot initial samples
    ax4.scatter(q_samples, p_samples, c='blue', s=1, alpha=0.3, label='Initial samples')
    
    # Transform each point by one leapfrog step
    q_new = np.zeros_like(q_samples)
    p_new = np.zeros_like(p_samples)
    
    for i in range(n_points):
        q_new[i], p_new[i] = leapfrog_step(q_samples[i], p_samples[i], 0.8, m, k)
    
    # Plot transformed samples
    ax4.scatter(q_new, p_new, c='red', s=1, alpha=0.3, label='After one step')
    
    contour = ax4.contour(Q, P, H, levels=levels, colors='gray', alpha=0.6, linewidths=1)
    
    ax4.set_xlabel('Position (q)', fontsize=12)
    ax4.set_ylabel('Momentum (p)', fontsize=12)
    ax4.set_title('Stationarity of Canonical Distribution', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # Plot 5: Canonical Samples
    ax5 = axes[1, 1]
    
    # Generate 2000 samples for histogram
    n_samples = 2000
    q_hist = np.random.normal(0, 1/np.sqrt(k), n_samples)
    p_hist = np.random.normal(0, np.sqrt(m), n_samples)
    
    ax5.hist2d(q_hist, p_hist, bins=30, cmap='Blues', alpha=0.7, density=True)
    contour = ax5.contour(Q, P, H, levels=levels, colors='gray', alpha=0.4, linewidths=1)
    
    ax5.set_xlabel('Position (q)', fontsize=12)
    ax5.set_ylabel('Momentum (p)', fontsize=12)
    ax5.set_title('2000 Samples from Canonical Distribution', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # Plot 6: Momentum Randomization (like phase space trajectories)
    ax6 = axes[1, 2]
    
    eps_mr = 0.3
    n_steps_mr = 40
    randomize_every = 10
    
    # Run HMC with momentum randomization
    np.random.seed(42)
    q, p = 0.0, 0.0
    
    q_traj = [q]
    p_traj = [p]
    
    for step in range(n_steps_mr):
        # Leapfrog step
        q, p = leapfrog_step(q, p, eps_mr, m, k)
        
        # Momentum randomization every 10 steps
        if (step + 1) % randomize_every == 0:
            p = np.random.normal(0, np.sqrt(m))
            ax6.plot(q, p, 'ro', markersize=8, alpha=0.8)
        
        q_traj.append(q)
        p_traj.append(p)
    
    # Energy contours
    contour = ax6.contour(Q, P, H, levels=levels, colors='gray', alpha=0.6, linewidths=1)
    
    # Plot trajectory
    ax6.plot(q_traj, p_traj, 'b-', linewidth=2, alpha=0.7, label='HMC trajectory')
    
    # Mark start and end
    ax6.plot(q_traj[0], p_traj[0], 'go', markersize=12, label='Start')
    ax6.plot(q_traj[-1], p_traj[-1], 'mo', markersize=12, label='End')
    
    ax6.set_xlabel('Position (q)', fontsize=12)
    ax6.set_ylabel('Momentum (p)', fontsize=12)
    ax6.set_title('HMC with Momentum Randomization', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('hmc_comprehensive_visualization.png', dpi=300, bbox_inches='tight')

def create_basic_trajectory_animation():
    """Show a single HMC trajectory evolving step by step with energy change"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Setup
    m, k = 1.0, 1.0
    eps = 0.8  # Bigger steps
    n_frames = 30
    
    # Initial point
    q, p = 2.0, 1.5
    
    # Energy contours (static background)
    Q, P, H = generate_energy_contours((-3, 3), (-3, 3), m, k)
    levels = np.linspace(0.5, 4.5, 9)
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # Calculate trajectory up to current frame
        q_traj, p_traj = [q], [p]
        energy_traj = [hamiltonian(q, p, m, k)]
        
        for i in range(frame):
            q_new, p_new = leapfrog_step(q_traj[-1], p_traj[-1], eps, m, k)
            q_traj.append(q_new)
            p_traj.append(p_new)
            energy_traj.append(hamiltonian(q_new, p_new, m, k))
        
        # Plot 1: Phase space trajectory
        ax1.contour(Q, P, H, levels=levels, colors='gray', alpha=0.6)
        ax1.plot(q_traj, p_traj, 'b-', linewidth=2, alpha=0.7)
        ax1.plot(q_traj[-1], p_traj[-1], 'ro', markersize=12)
        
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        ax1.set_title(f'HMC Trajectory - Step {frame}')
        ax1.set_xlabel('Position (q)')
        ax1.set_ylabel('Momentum (p)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2: Energy over time
        steps = range(len(energy_traj))
        ax2.plot(steps, energy_traj, 'b-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Energy H(q,p)')
        ax2.set_title('Energy Conservation')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(min(energy_traj) - 0.1, max(energy_traj) + 0.1)
    
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=150)
    anim.save('hmc_trajectory.gif', writer='pillow')
    print("Saved: hmc_trajectory.gif")

def create_measure_preservation_animation():
    """Show side-by-side comparison: measure preservation with and without momentum refresh"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Setup
    m, k = 1.0, 1.0
    eps = 0.8
    n_frames = 20
    
    # Setup grid
    n_grid = 15
    q_grid = np.linspace(1.5, 2.5, n_grid)
    p_grid = np.linspace(1.0, 2.0, n_grid)
    Q_grid, P_grid = np.meshgrid(q_grid, p_grid)
    
    # Energy contours
    Q, P, H = generate_energy_contours((-3, 3), (-3, 3), m, k)
    levels = np.linspace(0.5, 4.5, 9)
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # Draw energy contours
        ax1.contour(Q, P, H, levels=levels, colors='gray', alpha=0.4, linewidths=1)
        ax2.contour(Q, P, H, levels=levels, colors='gray', alpha=0.4, linewidths=1)
        
        # Transform grid for current frame (without momentum refresh)
        Q_curr1, P_curr1 = Q_grid.copy(), P_grid.copy()
        for step in range(frame):
            for i in range(n_grid):
                for j in range(n_grid):
                    Q_curr1[i, j], P_curr1[i, j] = leapfrog_step(Q_curr1[i, j], P_curr1[i, j], eps, m, k)
        
        # Transform grid for current frame (with momentum refresh every 5 steps)
        Q_curr2, P_curr2 = Q_grid.copy(), P_grid.copy()
        for step in range(frame):
            for i in range(n_grid):
                for j in range(n_grid):
                    Q_curr2[i, j], P_curr2[i, j] = leapfrog_step(Q_curr2[i, j], P_curr2[i, j], eps, m, k)
            
            # Momentum refresh every 5 steps
            if (step + 1) % 5 == 0:
                P_curr2 = np.random.normal(0, np.sqrt(m), P_curr2.shape)
        
        # Plot initial grid
        ax1.plot(Q_grid.flatten(), P_grid.flatten(), 'bo', markersize=4, alpha=0.7, label='Initial')
        ax2.plot(Q_grid.flatten(), P_grid.flatten(), 'bo', markersize=4, alpha=0.7, label='Initial')
        
        # Plot current grids
        ax1.plot(Q_curr1.flatten(), P_curr1.flatten(), 'ro', markersize=4, alpha=0.7, label=f'Step {frame}')
        ax2.plot(Q_curr2.flatten(), P_curr2.flatten(), 'ro', markersize=4, alpha=0.7, label=f'Step {frame}')
        
        # Set up both plots
        for ax, title in [(ax1, 'Without Momentum Refresh'), (ax2, 'With Momentum Refresh (every 5 steps)')]:
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_title(title)
            ax.set_xlabel('Position (q)')
            ax.set_ylabel('Momentum (p)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
    
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
    anim.save('hmc_measure_preservation.gif', writer='pillow')
    print("Saved: hmc_measure_preservation.gif")

def create_stationary_distribution_animation():
    """Show how canonical distribution samples evolve"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Setup
    m, k = 1.0, 1.0
    eps = 0.5
    n_frames = 15
    
    # Generate samples from canonical distribution
    np.random.seed(42)
    n_samples = 1000
    q_samples = np.random.normal(0, 1/np.sqrt(k), n_samples)
    p_samples = np.random.normal(0, np.sqrt(m), n_samples)
    
    # Energy contours
    Q, P, H = generate_energy_contours((-3, 3), (-3, 3), m, k)
    levels = np.linspace(0.5, 4.5, 9)
    
    def animate(frame):
        ax.clear()
        
        # Draw energy contours
        ax.contour(Q, P, H, levels=levels, colors='gray', alpha=0.6)
        
        # Transform samples for current frame
        q_curr, p_curr = q_samples.copy(), p_samples.copy()
        for step in range(frame):
            for i in range(n_samples):
                q_curr[i], p_curr[i] = leapfrog_step(q_curr[i], p_curr[i], eps, m, k)
        
        # Plot samples
        ax.scatter(q_samples, p_samples, c='blue', s=10, alpha=0.6, label='Initial')
        ax.scatter(q_curr, p_curr, c='red', s=10, alpha=0.6, label=f'Step {frame}')
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(f'Canonical Distribution Evolution - Step {frame}')
        ax.set_xlabel('Position (q)')
        ax.set_ylabel('Momentum (p)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=300)
    anim.save('hmc_stationary_distribution.gif', writer='pillow')
    print("Saved: hmc_stationary_distribution.gif")

def create_adaptive_step_size_animation():
    """Show adaptive step size selection based on energy error variance"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Setup
    m, k = 1.0, 1.0
    initial_eps = 0.8
    target_variance = 5e-4
    n_frames = 60
    
    # Initial point
    q, p = 2.0, 1.5
    
    # Energy contours
    Q, P, H = generate_energy_contours((-3, 3), (-3, 3), m, k)
    levels = np.linspace(0.5, 4.5, 9)
    
    # Store history for variance calculation
    energy_errors = []
    step_sizes = [initial_eps]
    trajectories = {'q': [q], 'p': [p]}
    
    def calculate_exact_energy(q, p):
        """Calculate exact energy at a point"""
        return hamiltonian(q, p, m, k)
    
    def calculate_energy_error(q_traj, p_traj, eps):
        """Calculate energy error along trajectory"""
        if len(q_traj) < 2:
            return 0.0
        
        # Calculate energy at each point
        energies = [calculate_exact_energy(q, p) for q, p in zip(q_traj, p_traj)]
        
        # Calculate energy change (error from conservation)
        initial_energy = energies[0]
        energy_errors = [abs(e - initial_energy) for e in energies]
        
        return energy_errors
    
    def calculate_running_variance(errors, window=10):
        """Calculate running variance of energy errors"""
        if len(errors) < window:
            return np.var(errors) if len(errors) > 1 else 0.0
        
        # Use rolling window for variance
        recent_errors = errors[-window:]
        return np.var(recent_errors)
    
    def adjust_step_size(current_eps, variance, target_variance):
        """Adjust step size based on variance"""
        if variance > target_variance * 1.5:  # Too high
            return current_eps * 0.9
        elif variance < target_variance * 0.5:  # Too low
            return current_eps * 1.1
        else:
            return current_eps
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        
        # Update trajectory for current frame
        if frame > 0:
            # Use current step size
            current_eps = step_sizes[-1]
            
            # Add one more step
            q_new, p_new = leapfrog_step(trajectories['q'][-1], trajectories['p'][-1], current_eps, m, k)
            trajectories['q'].append(q_new)
            trajectories['p'].append(p_new)
            
            # Calculate energy errors
            errors = calculate_energy_error(trajectories['q'], trajectories['p'], current_eps)
            energy_errors.extend(errors)
            
            # Calculate running variance
            variance = calculate_running_variance(energy_errors)
            
            # Adjust step size based on variance
            new_eps = adjust_step_size(current_eps, variance, target_variance)
            step_sizes.append(new_eps)
        
        # Plot 1: Phase space trajectory
        ax1.contour(Q, P, H, levels=levels, colors='gray', alpha=0.6)
        ax1.plot(trajectories['q'], trajectories['p'], 'b-', linewidth=2, alpha=0.7)
        ax1.plot(trajectories['q'][-1], trajectories['p'][-1], 'ro', markersize=12)
        
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        ax1.set_title(f'Trajectory (Step {frame})')
        ax1.set_xlabel('Position (q)')
        ax1.set_ylabel('Momentum (p)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2: Energy error over time
        if len(energy_errors) > 0:
            steps = range(len(energy_errors))
            ax2.plot(steps, energy_errors, 'b-', linewidth=2, alpha=0.7)
            ax2.axhline(y=target_variance, color='r', linestyle='--', alpha=0.7, label='Target')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Energy Error |ΔH|')
            ax2.set_title('Energy Error Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        # Plot 3: Running variance
        if len(energy_errors) > 0:
            # Calculate running variance for each point
            running_variances = []
            for i in range(1, len(energy_errors) + 1):
                window_errors = energy_errors[:i]
                if len(window_errors) >= 5:  # Need at least 5 points for meaningful variance
                    recent_errors = window_errors[-10:]  # Last 10 points
                    running_variances.append(np.var(recent_errors))
                else:
                    running_variances.append(0.0)
            
            steps = range(len(running_variances))
            ax3.plot(steps, running_variances, 'g-', linewidth=2, alpha=0.7)
            ax3.axhline(y=target_variance, color='r', linestyle='--', alpha=0.7, label='Target')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Running Variance of Energy Error')
            ax3.set_title('Energy Error Variance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
        
        # Plot 4: Step size evolution
        if len(step_sizes) > 1:
            steps = range(len(step_sizes))
            ax4.plot(steps, step_sizes, 'purple', linewidth=2, alpha=0.7, marker='o', markersize=4)
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Step Size ε')
            ax4.set_title('Adaptive Step Size')
            ax4.grid(True, alpha=0.3)
            
            # Add current step size as text
            current_eps = step_sizes[-1]
            ax4.text(0.02, 0.98, f'Current ε: {current_eps:.3f}', 
                    transform=ax4.transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
    anim.save('hmc_adaptive_step_size.gif', writer='pillow')
    print("Saved: hmc_adaptive_step_size.gif")

if __name__ == "__main__":
    # Generate the comprehensive visualization
    print("Generating comprehensive HMC visualization...")
    plot_comprehensive_hmc()
    
    print("\nGenerating HMC animations...")
    
    # Generate animations
    print("Creating basic trajectory animation...")
    create_basic_trajectory_animation()
    
    print("Creating measure preservation animation...")
    create_measure_preservation_animation()
    
    print("Creating stationary distribution animation...")
    create_stationary_distribution_animation()
    
    print("Creating adaptive step size animation...")
    create_adaptive_step_size_animation()
    
    print("\nAll files saved:")
    print("- hmc_comprehensive_visualization.png")
    print("- hmc_trajectory.gif")
    print("- hmc_measure_preservation.gif")
    print("- hmc_stationary_distribution.gif")
    print("- hmc_adaptive_step_size.gif") 