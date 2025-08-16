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
    
    # Plot 3: Measure Preservation (without momentum resampling)
    ax3 = axes[0, 2]
    
    eps_mp = 1.5  # Larger step size
    n_grid = 15
    q_grid = np.linspace(1.5, 2.5, n_grid)
    p_grid = np.linspace(1.0, 2.0, n_grid)
    Q_grid, P_grid = np.meshgrid(q_grid, p_grid)
    
    # Transform points through multiple steps without momentum resampling
    Q_curr, P_curr = Q_grid.copy(), P_grid.copy()
    
    for step in range(30):
        # Leapfrog step only
        for i in range(n_grid):
            for j in range(n_grid):
                Q_curr[i, j], P_curr[i, j] = leapfrog_step(Q_curr[i, j], P_curr[i, j], eps_mp, m, k)
    
    contour = ax3.contour(Q, P, H, levels=levels, colors='gray', alpha=0.4, linewidths=1)
    ax3.plot(Q_grid.flatten(), P_grid.flatten(), 'bo', markersize=4, alpha=0.7, label='Initial')
    ax3.plot(Q_curr.flatten(), P_curr.flatten(), 'ro', markersize=4, alpha=0.7, label='After 30 steps')
    
    ax3.set_xlabel('Position (q)', fontsize=12)
    ax3.set_ylabel('Momentum (p)', fontsize=12)
    ax3.set_title('Measure Preservation', fontsize=14, fontweight='bold')
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
    n_steps_mr = 100
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
    
    ax6.set_xlabel('Position (q)', fontsize=12)
    ax6.set_ylabel('Momentum (p)', fontsize=12)
    ax6.set_title('HMC with Momentum Randomization', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')
    
    # Plot 7: Exact Dynamics (Energy and Measure Preservation)
    # Create a new figure for this since we're at 2x3 layout
    fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Setup
    m, k = 1.0, 1.0
    t_final = 4.0
    n_points = 100
    
    # Initial conditions
    q0, p0 = 1.5, 1.0
    
    # Time points for exact solution
    t_points = np.linspace(0, t_final, n_points)
    
    # Solve exact dynamics using scipy
    from scipy.integrate import solve_ivp
    
    def exact_dynamics_system(t, state):
        q, p = state
        dq_dt = p / m
        dp_dt = -k * q
        return [dq_dt, dp_dt]
    
    # Solve for single trajectory
    solution = solve_ivp(exact_dynamics_system, [0, t_final], [q0, p0], 
                        t_eval=t_points, method='RK45')
    q_exact = solution.y[0]
    p_exact = solution.y[1]
    
    # Calculate energy along trajectory
    energy_exact = [hamiltonian(q, p, m, k) for q, p in zip(q_exact, p_exact)]
    energy_initial = hamiltonian(q0, p0, m, k)
    
    # Plot 7: Exact trajectory in phase space
    Q, P, H = generate_energy_contours((-3, 3), (-3, 3), m, k)
    levels = np.linspace(0.5, 4.5, 9)
    
    ax7.contour(Q, P, H, levels=levels, colors='gray', alpha=0.6, linewidths=1)
    ax7.plot(q_exact, p_exact, 'b-', linewidth=2, alpha=0.8)
    ax7.plot(q_exact[0], p_exact[0], 'bo', markersize=10)
    ax7.plot(q_exact[-1], p_exact[-1], 'ro', markersize=10)
    
    # Add energy values next to start and end points
    ax7.annotate(f'H = {energy_initial:.3f}', xy=(q_exact[0], p_exact[0]), 
                xytext=(10, 10), textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax7.annotate(f'H = {energy_initial:.3f}', xy=(q_exact[-1], p_exact[-1]), 
                xytext=(10, 10), textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax7.set_xlabel('Position (q)', fontsize=12)
    ax7.set_ylabel('Momentum (p)', fontsize=12)
    ax7.set_title('Initial Energy = Final Energy', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.set_aspect('equal')
    
    # Plot 8: Energy conservation and measure preservation
    # Create a larger volume of points around initial condition
    n_volume = 30
    q_volume = np.linspace(q0 - 0.3, q0 + 0.3, n_volume)
    p_volume = np.linspace(p0 - 0.3, p0 + 0.3, n_volume)
    Q_vol, P_vol = np.meshgrid(q_volume, p_volume)
    
    # Transform volume through exact dynamics
    Q_vol_final = np.zeros_like(Q_vol)
    P_vol_final = np.zeros_like(P_vol)
    
    for i in range(n_volume):
        for j in range(n_volume):
            solution_vol = solve_ivp(exact_dynamics_system, [0, t_final], 
                                   [Q_vol[i, j], P_vol[i, j]], method='RK45')
            Q_vol_final[i, j] = solution_vol.y[0, -1]
            P_vol_final[i, j] = solution_vol.y[1, -1]
    
    # Calculate initial and final volumes
    initial_area = (q_volume[1] - q_volume[0]) * (p_volume[1] - p_volume[0]) * n_volume**2
    
    # Plot initial and final volumes
    ax8.contour(Q, P, H, levels=levels, colors='gray', alpha=0.6, linewidths=1)
    ax8.plot(Q_vol.flatten(), P_vol.flatten(), 'bo', markersize=4, alpha=0.7)
    ax8.plot(Q_vol_final.flatten(), P_vol_final.flatten(), 'ro', markersize=4, alpha=0.7)
    
    # Calculate volume
    initial_area = (q_volume[1] - q_volume[0]) * (p_volume[1] - p_volume[0]) * n_volume**2
    
    # Add volume values next to initial and final volume centers
    q_vol_center_initial = np.mean(Q_vol)
    p_vol_center_initial = np.mean(P_vol)
    q_vol_center_final = np.mean(Q_vol_final)
    p_vol_center_final = np.mean(P_vol_final)
    
    ax8.annotate(f'Volume = {initial_area:.3f}', xy=(q_vol_center_initial, p_vol_center_initial), 
                xytext=(10, 10), textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax8.annotate(f'Volume = {initial_area:.3f}', xy=(q_vol_center_final, p_vol_center_final), 
                xytext=(10, 10), textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax8.set_xlabel('Position (q)', fontsize=12)
    ax8.set_ylabel('Momentum (p)', fontsize=12)
    ax8.set_title('Initial Volume = Final Volume', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.set_aspect('equal')
    

    
    plt.tight_layout()
    plt.savefig('hmc_exact_dynamics.png', dpi=300, bbox_inches='tight')
    
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
        
        # Transform grid for current frame (with momentum refresh every 5 steps, starting from step 0)
        Q_curr2, P_curr2 = Q_grid.copy(), P_grid.copy()
        for step in range(frame):
            for i in range(n_grid):
                for j in range(n_grid):
                    Q_curr2[i, j], P_curr2[i, j] = leapfrog_step(Q_curr2[i, j], P_curr2[i, j], eps, m, k)
            
            # Momentum refresh every 5 steps (including step 0)
            if step % 5 == 0:
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
    eps = 0.5  # Larger step size to increase chance of rejections
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
    target_variance = 1e-2
    n_frames = 60
    
    # Initial point
    q, p = 2.0, 1.5
    
    # Energy contours
    Q, P, H = generate_energy_contours((-3, 3), (-3, 3), m, k)
    levels = np.linspace(0.5, 4.5, 9)
    
    # Store history for variance calculation using running averages
    energy_errors = []
    running_avg_delta_H = 0.0
    running_avg_delta_H_sq = 0.0
    step_sizes = [initial_eps]
    trajectories = {'q': [q], 'p': [p]}
    
    def calculate_exact_energy(q, p):
        """Calculate exact energy at a point"""
        return hamiltonian(q, p, m, k)
    
    def update_running_averages(delta_H, n, current_avg_delta_H, current_avg_delta_H_sq):
        """Update running averages for variance calculation using window of 5 samples"""
        window_size = 5
        
        if n <= window_size:
            # Use all samples if we have 5 or fewer
            new_avg_delta_H = (current_avg_delta_H * (n-1) + delta_H) / n
            new_avg_delta_H_sq = (current_avg_delta_H_sq * (n-1) + delta_H**2) / n
        else:
            # Use only the last 5 samples
            recent_errors = energy_errors[-window_size:]
            new_avg_delta_H = np.mean(recent_errors)
            new_avg_delta_H_sq = np.mean([e**2 for e in recent_errors])
        
        return new_avg_delta_H, new_avg_delta_H_sq
    
    def calculate_variance_from_averages(avg_delta_H, avg_delta_H_sq):
        """Calculate variance from running averages: Var = E[X^2] - E[X]^2"""
        return avg_delta_H_sq - avg_delta_H**2
    
    def adjust_step_size(current_eps, variance, target_variance):
        """Adjust step size based on variance"""
        if variance > target_variance * 1.5:  # Too high
            return current_eps * 0.9
        elif variance < target_variance * 0.5:  # Too low
            return current_eps * 1.1
        else:
            return current_eps
    
    def animate(frame):
        nonlocal running_avg_delta_H, running_avg_delta_H_sq
        
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
            
            # Calculate energy error for this step
            initial_energy = calculate_exact_energy(trajectories['q'][0], trajectories['p'][0])
            current_energy = calculate_exact_energy(q_new, p_new)
            delta_H = abs(current_energy - initial_energy)
            energy_errors.append(delta_H)
            
            # Update running averages
            n = len(energy_errors)
            running_avg_delta_H, running_avg_delta_H_sq = update_running_averages(
                delta_H, n, running_avg_delta_H, running_avg_delta_H_sq)
            
            # Calculate variance from running averages
            variance = calculate_variance_from_averages(running_avg_delta_H, running_avg_delta_H_sq)
            
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
        
        # Plot 2: Average of x^2 over time
        if len(trajectories['q']) > 0:
            # Calculate running average of x^2
            x_squared_avg = []
            for i in range(1, len(trajectories['q']) + 1):
                recent_q = trajectories['q'][:i]
                avg_x_sq = np.mean([q**2 for q in recent_q])
                x_squared_avg.append(avg_x_sq)
            
            steps = range(len(x_squared_avg))
            ax2.plot(steps, x_squared_avg, 'b-', linewidth=2, alpha=0.7)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Average of x²')
            ax2.set_title('Average Position Squared')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Instantaneous delta_H^2
        if len(energy_errors) > 0:
            # Calculate instantaneous delta_H^2 at each step
            delta_H_squared = [delta_H**2 for delta_H in energy_errors]
            
            steps = range(len(delta_H_squared))
            ax3.plot(steps, delta_H_squared, 'g-', linewidth=2, alpha=0.7)
            ax3.axhline(y=target_variance, color='r', linestyle='--', alpha=0.7, label='Target')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Instantaneous (ΔH)²')
            ax3.set_title('Energy Error Squared')
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

def microcanonical_update1(x, u, eps):
    """First microcanonical update: (x + eps * u, u)"""
    return x + eps * u, u

def microcanonical_update2(x, u, eps):
    """Second microcanonical update as in the screenshot."""
    # For this example, let's assume e is a unit vector (direction), and delta = eps
    # This is a 1D version, so e = sign(u) or just 1
    # In higher dimensions, e would be a direction vector
    delta = eps
    e = 1.0  # For 1D, e = 1
    numerator = u + (np.sinh(delta) + e * u * (np.cosh(delta) - 1)) * e
    denominator = np.cosh(delta) + e * u * np.sinh(delta)
    u_new = numerator / denominator
    return x, u_new

def microcanonical_leapfrog(x0, u0, eps, n_steps):
    """Perform microcanonical leapfrog integration."""
    x, u = x0, u0
    x_traj = [x]
    u_traj = [u]
    for _ in range(n_steps):
        x, u = microcanonical_update1(x, u, eps)
        x, u = microcanonical_update2(x, u, eps)
        x_traj.append(x)
        u_traj.append(u)
    return np.array(x_traj), np.array(u_traj)


def create_microcanonical_animation():
    """Animate microcanonical dynamics in phase space."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots(figsize=(7, 7))
    eps = 0.3
    n_steps = 40
    x0, u0 = 1.5, 1.0
    
    x_traj, u_traj = microcanonical_leapfrog(x0, u0, eps, n_steps)
    
    def animate(frame):
        ax.clear()
        ax.plot(x_traj[:frame+1], u_traj[:frame+1], 'b-', linewidth=2)
        ax.plot(x_traj[0], u_traj[0], 'bo', markersize=10)
        ax.plot(x_traj[frame], u_traj[frame], 'ro', markersize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title('Microcanonical Dynamics Trajectory')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
    
    anim = animation.FuncAnimation(fig, animate, frames=n_steps+1, interval=150)
    anim.save('microcanonical_trajectory.gif', writer='pillow')
    print('Saved: microcanonical_trajectory.gif')

def V_1d(x):
    """1D Gaussian potential energy."""
    return 0.5 * x**2

def grad_V_1d(x):
    return x

def T_1d(u):
    """1D kinetic energy: log(|u|)."""
    return np.log(np.abs(u))

def grad_T_1d(u):
    return 1.0 / u


def leapfrog_1d(x0, u0, eps, n_steps):
    x, u = x0, u0
    x_traj = [x]
    u_traj = [u]
    for _ in range(n_steps):
        # Half step for momentum
        u = u - 0.5 * eps * grad_V_1d(x)
        # Full step for position
        x = x + eps * grad_T_1d(u)
        # Half step for momentum
        u = u - 0.5 * eps * grad_V_1d(x)
        x_traj.append(x)
        u_traj.append(u)
    return np.array(x_traj), np.array(u_traj)


def create_hmc_logkinetic_animation_1d():
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots(figsize=(7, 7))
    eps = 0.003
    n_steps = 2000
    x0 = 0.0
    u0 = 0.5  # Must be > 0 for log(|u|)
    
    x_traj, u_traj = leapfrog_1d(x0, u0, eps, n_steps)
    
    def animate(frame):
        frame = frame*10
        ax.clear()
        ax.plot(x_traj[:frame+1], u_traj[:frame+1], 'b-', linewidth=2)
        ax.plot(x_traj[0], u_traj[0], 'bo', markersize=10)
        ax.plot(x_traj[frame], u_traj[frame], 'ro', markersize=10)
        # Rug plot of x samples at the bottom
        ax.plot(x_traj[:frame+1], np.full(frame+1, -2.8), '|', color='k', markersize=10, alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title('HMC Trajectory with log(|u|) Kinetic (1D)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
    
    anim = animation.FuncAnimation(fig, animate, frames=n_steps//10+1, interval=30)
    anim.save('hmc_logkinetic_trajectory_1d.gif', writer='pillow')
    print('Saved: hmc_logkinetic_trajectory_1d.gif')

def V(x):
    """2D Gaussian potential energy."""
    return 0.5 * np.sum(x**2)

def grad_V(x):
    """Gradient of 2D Gaussian potential energy."""
    return x

def T(u):
    """Kinetic energy: log(|u|) where |u| is the Euclidean norm."""
    return np.log(np.linalg.norm(u))

def grad_T(u):
    """Gradient of kinetic energy log(|u|)."""
    norm = np.linalg.norm(u)
    return u / (norm**2)


def microcanonical_update1_2d(x, u, eps):
    """First microcanonical update: (x + eps * u, u) in 2D."""
    return x + eps * u, u

def microcanonical_update2_2d(x, u, eps):
    """Second microcanonical update in 2D using provided formulas."""
    d = 2
    gradV = grad_V(x)
    gradV_norm = np.linalg.norm(gradV)
    if gradV_norm == 0:
        return x, u  # No update if gradient is zero
    delta = eps * gradV_norm / (d - 1)
    e = -gradV / gradV_norm
    u_dot_e = np.dot(u, e)
    sinh_delta = np.sinh(delta)
    cosh_delta = np.cosh(delta)
    numerator = u + (sinh_delta + u_dot_e * (cosh_delta - 1)) * e
    denominator = cosh_delta + u_dot_e * sinh_delta
    u_new = numerator / denominator
    return x, u_new

def microcanonical_leapfrog_2d(x0, u0, eps, n_steps):
    x, u = np.array(x0), np.array(u0)
    x_traj = [x.copy()]
    u_traj = [u.copy()]
    for _ in range(n_steps):
        x, u = microcanonical_update1_2d(x, u, eps)
        x, u = microcanonical_update2_2d(x, u, eps)
        x_traj.append(x.copy())
        u_traj.append(u.copy())
    return np.array(x_traj), np.array(u_traj)


def create_microcanonical_animation_2d():
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots(figsize=(7, 7))
    eps = 0.3
    n_steps = 40
    x0 = np.array([1.0, 0.5])
    u0 = np.array([1.0, 1.0])
    
    x_traj, u_traj = microcanonical_leapfrog_2d(x0, u0, eps, n_steps)
    
    def animate(frame):
        ax.clear()
        ax.plot(x_traj[:frame+1,0], x_traj[:frame+1,1], 'b-', linewidth=2, label='x trajectory')
        ax.plot(x_traj[0,0], x_traj[0,1], 'bo', markersize=10)
        ax.plot(x_traj[frame,0], x_traj[frame,1], 'ro', markersize=10)
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_title('Microcanonical Dynamics Trajectory (2D)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
    
    anim = animation.FuncAnimation(fig, animate, frames=n_steps+1, interval=150)
    anim.save('microcanonical_trajectory_2d.gif', writer='pillow')
    print('Saved: microcanonical_trajectory_2d.gif')

def plot_shadow_hamiltonian():
    """Plot showing that discretized updates follow a shadow Hamiltonian"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Parameters
    m, k = 1.0, 1.0
    eps = 0.8
    n_steps = 12
    
    # Initial point
    q0, p0 = 2.0, 1.5
    
    # Generate exact and leapfrog trajectories
    t_exact = np.linspace(0, n_steps * eps, 1000)
    exact_traj = np.array([exact_dynamics(t, [q0, p0], m, k) for t in t_exact])
    
    q_lf, p_lf = q0, p0
    lf_traj = [(q_lf, p_lf)]
    for i in range(n_steps):
        q_lf, p_lf = leapfrog_step(q_lf, p_lf, eps, m, k)
        lf_traj.append((q_lf, p_lf))
    lf_traj = np.array(lf_traj)
    
    # Calculate shadow Hamiltonian terms
    def shadow_hamiltonian_terms(q, p, eps, m, k):
        """Calculate first few terms of the shadow Hamiltonian expansion"""
        H0 = hamiltonian(q, p, m, k)  # Original Hamiltonian
        
        # First correction term: ε²/12 * (p²/m² * ∂²V/∂q² - V'(q)²/m)
        # For harmonic oscillator: V(q) = k/2 * q², V'(q) = k*q, V''(q) = k
        H1 = (eps**2/12) * (p**2/m**2 * k - (k*q)**2/m)
        
        # Second correction term: ε⁴/720 * (higher order terms)
        # Simplified for harmonic oscillator
        H2 = (eps**4/720) * (k**2 * p**2 / m**2)
        
        return H0, H1, H2
    
    # Calculate shadow Hamiltonian along leapfrog trajectory
    shadow_H0 = []
    shadow_H1 = []
    shadow_H2 = []
    shadow_H_total = []
    
    for q, p in lf_traj:
        h0, h1, h2 = shadow_hamiltonian_terms(q, p, eps, m, k)
        shadow_H0.append(h0)
        shadow_H1.append(h1)
        shadow_H2.append(h2)
        shadow_H_total.append(h0 + h1 + h2)
    
    # Plot 1: Phase space with shadow Hamiltonian contours
    ax1.plot(exact_traj[:, 0], exact_traj[:, 1], 'b-', linewidth=2, 
             label='Exact dynamics', alpha=0.8)
    ax1.plot(lf_traj[:, 0], lf_traj[:, 1], 'ro-', linewidth=2, markersize=8,
             label='Leapfrog steps', alpha=0.8)
    
    # Generate shadow Hamiltonian contours
    q_range = (-3, 3)
    p_range = (-3, 3)
    Q, P, H = generate_energy_contours(q_range, p_range, m, k)
    
    # Calculate shadow Hamiltonian on grid
    shadow_H_grid = np.zeros_like(H)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            h0, h1, h2 = shadow_hamiltonian_terms(Q[i, j], P[i, j], eps, m, k)
            shadow_H_grid[i, j] = h0 + h1 + h2
    
    # Plot shadow Hamiltonian contours
    levels_shadow = np.linspace(0.5, 4.5, 9)
    contour_shadow = ax1.contour(Q, P, shadow_H_grid, levels=levels_shadow, 
                                colors='red', alpha=0.6, linewidths=1, linestyles='--')
    ax1.clabel(contour_shadow, inline=True, fontsize=8, colors='red')
    
    # Original Hamiltonian contours
    contour_orig = ax1.contour(Q, P, H, levels=levels_shadow, 
                              colors='gray', alpha=0.4, linewidths=1)
    ax1.clabel(contour_orig, inline=True, fontsize=8, colors='gray')
    
    ax1.plot(q0, p0, 'go', markersize=12, label='Initial point')
    ax1.set_xlabel('Position (q)', fontsize=12)
    ax1.set_ylabel('Momentum (p)', fontsize=12)
    ax1.set_title('Leapfrog Follows Shadow Hamiltonian', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Shadow Hamiltonian terms over time
    steps = range(len(shadow_H0))
    ax2.plot(steps, shadow_H0, 'b-', linewidth=2, label='H_0 (Original)', alpha=0.8)
    ax2.plot(steps, shadow_H1, 'r-', linewidth=2, label='H_1 (ε² correction)', alpha=0.8)
    ax2.plot(steps, shadow_H2, 'g-', linewidth=2, label='H_2 (ε⁴ correction)', alpha=0.8)
    ax2.plot(steps, shadow_H_total, 'k-', linewidth=3, label='H_shadow (Total)', alpha=0.9)
    
    ax2.set_xlabel('Leapfrog Step', fontsize=12)
    ax2.set_ylabel('Energy', fontsize=12)
    ax2.set_title('Shadow Hamiltonian Terms', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add LaTeX formulas for shadow Hamiltonian
    general_formula = r'$H_{\text{shadow}} = H + \frac{\epsilon^2}{12}\{H,\{H,T\}\} + \frac{\epsilon^4}{720}\{H,\{H,\{H,\{H,T\}\}\}\} + \mathcal{O}(\epsilon^6)$'
    specific_formula = r'$H_{\text{shadow}} = H_0 + \frac{\epsilon^2}{12}H_1 + \frac{\epsilon^4}{720}H_2 + \mathcal{O}(\epsilon^6)$'
    
    ax2.text(0.02, 0.98, general_formula, transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(0.02, 0.85, specific_formula, transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('hmc_shadow_hamiltonian.png', dpi=300, bbox_inches='tight')
    print("Saved: hmc_shadow_hamiltonian.png")

def create_momentum_refresh_animation():
    """Show momentum refreshes in HMC with clear visualization of the process"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Setup
    m, k = 1.0, 1.0
    eps = 0.5
    steps_per_refresh = 8
    n_refreshes = 3
    n_frames_per_phase = 12  # Frames for each phase (trajectory or refresh) - shorter pause
    
    # Initial point
    q, p = 1.5, 1.0
    
    # Energy contours
    Q, P, H = generate_energy_contours((-3, 3), (-3, 3), m, k)
    levels = np.linspace(0.5, 4.5, 9)
    
    # Store trajectory and refresh points
    q_traj = [q]
    p_traj = [p]
    refresh_points = []
    
    # Generate full trajectory with refreshes
    for refresh in range(n_refreshes):
        # Trajectory phase
        for step in range(steps_per_refresh):
            q_new, p_new = leapfrog_step(q_traj[-1], p_traj[-1], eps, m, k)
            q_traj.append(q_new)
            p_traj.append(p_new)
        
        # Momentum refresh
        p_refresh = np.random.normal(0, np.sqrt(m))
        refresh_points.append((q_traj[-1], p_traj[-1], p_refresh))
        q_traj.append(q_traj[-1])  # Same position
        p_traj.append(p_refresh)   # New momentum
    
    q_traj = np.array(q_traj)
    p_traj = np.array(p_traj)
    
    # Calculate total phases
    total_phases = n_refreshes * 3  # trajectory + pause + refresh for each cycle
    
    def animate(frame):
        ax.clear()
        
        # Draw energy contours
        ax.contour(Q, P, H, levels=levels, colors='gray', alpha=0.6, linewidths=1)
        
        # Determine current phase
        phase = frame // n_frames_per_phase
        phase_progress = (frame % n_frames_per_phase) / n_frames_per_phase
        
        if phase >= total_phases:
            phase = total_phases - 1
            phase_progress = 1.0
        
        # Calculate how much of the trajectory to show
        if phase % 3 == 0:  # Trajectory phase
            refresh_idx = phase // 3
            start_idx = refresh_idx * (steps_per_refresh + 1)
            end_idx = start_idx + steps_per_refresh + 1
            
            # Show trajectory up to current progress
            progress_steps = int(phase_progress * steps_per_refresh)
            current_end = start_idx + progress_steps + 1
            
            # Plot completed trajectory
            if current_end > start_idx:
                ax.plot(q_traj[start_idx:current_end], p_traj[start_idx:current_end], 
                       'b-', linewidth=3, alpha=0.8)
                ax.plot(q_traj[start_idx:current_end], p_traj[start_idx:current_end], 
                       'bo', markersize=6, alpha=0.8)
            
            # Current position
            if current_end <= len(q_traj):
                ax.plot(q_traj[current_end-1], p_traj[current_end-1], 'ro', markersize=12)
            
            title = f'Trajectory Phase {refresh_idx + 1}'
            
        elif phase % 3 == 1:  # Pause before refresh
            refresh_idx = phase // 3
            
            # Show completed trajectory
            current_start = refresh_idx * (steps_per_refresh + 1)
            current_end = current_start + steps_per_refresh + 1
            ax.plot(q_traj[current_start:current_end], p_traj[current_start:current_end], 
                   'b-', linewidth=3, alpha=0.8)
            ax.plot(q_traj[current_start:current_end], p_traj[current_start:current_end], 
                   'bo', markersize=6, alpha=0.8)
            
            # Show current position (pause)
            ax.plot(q_traj[current_end-1], p_traj[current_end-1], 'ro', markersize=12)
            
            title = f'Pause Before Refresh {refresh_idx + 1}'
            
        else:  # Refresh phase
            refresh_idx = phase // 3
            
            # Show current trajectory up to refresh point
            current_start = refresh_idx * (steps_per_refresh + 1)
            current_end = current_start + steps_per_refresh + 1
            ax.plot(q_traj[current_start:current_end], p_traj[current_start:current_end], 
                   'b-', linewidth=3, alpha=0.8)
            ax.plot(q_traj[current_start:current_end], p_traj[current_start:current_end], 
                   'bo', markersize=6, alpha=0.8)
            
            # Show momentum refresh as a single jump
            if refresh_idx < len(refresh_points):
                q_refresh, p_old, p_new = refresh_points[refresh_idx]
                
                # Show old momentum
                ax.plot(q_refresh, p_old, 'ro', markersize=12, label='Before refresh')
                
                # Show new momentum (instantaneous jump)
                ax.plot(q_refresh, p_new, 'go', markersize=12, label='After refresh')
                
                # Draw arrow for momentum change
                ax.annotate('', xy=(q_refresh, p_new), xytext=(q_refresh, p_old),
                           arrowprops=dict(arrowstyle='->', color='red', lw=3))
            
            title = f'Momentum Refresh {refresh_idx + 1}'
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Position (q)', fontsize=12)
        ax.set_ylabel('Momentum (p)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    total_frames = total_phases * n_frames_per_phase
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=200)
    anim.save('hmc_momentum_refresh.gif', writer='pillow')
    print("Saved: hmc_momentum_refresh.gif")

def create_langevin_dynamics_animation():
    """Show underdamped Langevin dynamics with Euler-Maruyama integration"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Setup
    m, k = 1.0, 1.0
    gamma = 0.1  # Damping coefficient
    T = 1.0      # Temperature
    dt = 0.15    # Time step (larger for more visible steps)
    n_steps = 20  # Reduced from 100 to make GIF shorter
    
    # Initial point
    q, p = 1.5, 1.0
    
    # Energy contours
    Q, P, H = generate_energy_contours((-3, 3), (-3, 3), m, k)
    levels = np.linspace(0.5, 4.5, 9)
    
    # Store trajectory
    q_traj = [q]
    p_traj = [p]
    step_types = ['initial']  # Track what type of step each point represents
    
    # Generate Langevin trajectory with symmetric splitting
    np.random.seed(42)
    for step in range(n_steps):
        # Step 1: Noise/2
        noise1 = np.random.normal(0, np.sqrt(2 * gamma * T * dt / 2))
        p_noise1 = p + noise1
        q_traj.append(q)
        p_traj.append(p_noise1)
        step_types.append('noise')
        
        # Step 2: Position/2
        q_pos1 = q + (dt/2) * p_noise1 / m
        q_traj.append(q_pos1)
        p_traj.append(p_noise1)
        step_types.append('position')
        
        # Step 3: Momentum (force + damping)
        p_momentum = p_noise1 - dt * k * q_pos1 - dt * gamma * p_noise1 / m
        q_traj.append(q_pos1)
        p_traj.append(p_momentum)
        step_types.append('force')
        
        # Step 4: Position/2
        q_pos2 = q_pos1 + (dt/2) * p_momentum / m
        q_traj.append(q_pos2)
        p_traj.append(p_momentum)
        step_types.append('position')
        
        # Step 5: Noise/2
        noise2 = np.random.normal(0, np.sqrt(2 * gamma * T * dt / 2))
        p_new = p_momentum + noise2
        q_traj.append(q_pos2)
        p_traj.append(p_new)
        step_types.append('noise')
        
        q, p = q_pos2, p_new
    
    q_traj = np.array(q_traj)
    p_traj = np.array(p_traj)
    
    def animate(frame):
        ax.clear()
        
        # Draw energy contours
        ax.contour(Q, P, H, levels=levels, colors='gray', alpha=0.6, linewidths=1)
        
        # Show trajectory up to current frame
        if frame > 0:
            # Plot trajectory with different colors for different step types
            for i in range(1, min(frame + 1, len(q_traj))):
                if step_types[i] == 'position':
                    color = 'blue'
                    alpha = 0.8
                elif step_types[i] == 'force':
                    color = 'green'
                    alpha = 0.8
                elif step_types[i] == 'noise':
                    color = 'red'
                    alpha = 0.8
                else:
                    color = 'black'
                    alpha = 0.8
                
                # Draw line from previous point
                if i > 0:
                    ax.plot([q_traj[i-1], q_traj[i]], [p_traj[i-1], p_traj[i]], 
                           color=color, linewidth=2, alpha=alpha)
                
                # Draw current point (smaller dots, same shape)
                ax.plot(q_traj[i], p_traj[i], color=color, marker='o', 
                       markersize=3, alpha=alpha)
        
        # Show initial point
        ax.plot(q_traj[0], p_traj[0], 'ko', markersize=12, label='Initial')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', marker='o', linestyle='-', label='Position Update'),
            Line2D([0], [0], color='green', marker='o', linestyle='-', label='Force Update'),
            Line2D([0], [0], color='red', marker='o', linestyle='-', label='Noise Update')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(f'Underdamped Langevin Dynamics - Step {frame}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Position (q)', fontsize=12)
        ax.set_ylabel('Momentum (p)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(q_traj), interval=100)
    anim.save('langevin_dynamics.gif', writer='pillow')
    print("Saved: langevin_dynamics.gif")

def create_hmc_metropolis_animation():
    """Show HMC with Metropolis-Hastings adjustment: trajectory, accept/reject, momentum refresh"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Setup
    m, k = 1.0, 1.0
    eps = 2.  # Larger step size to show energy conservation errors
    steps_per_trajectory = 3
    n_cycles = 3
    
    # Initial point
    q, p = 1.5, 1.0
    
    # Energy contours
    Q, P, H = generate_energy_contours((-3, 3), (-3, 3), m, k)
    levels = np.linspace(0.5, 4.5, 9)
    
    # Store trajectory and cycle information
    q_traj = [q]
    p_traj = [p]
    cycle_info = []  # Store (start_idx, end_idx, accepted, energy_change)
    
    # Generate HMC trajectory with Metropolis-Hastings
    np.random.seed(42)
    for cycle in range(n_cycles):
        # Store initial state for this cycle
        q_start = q_traj[-1]
        p_start = p_traj[-1]
        start_idx = len(q_traj) - 1
        
        # Generate trajectory
        q_traj_cycle = [q_start]
        p_traj_cycle = [p_start]
        
        for step in range(steps_per_trajectory):
            q_new, p_new = leapfrog_step(q_traj_cycle[-1], p_traj_cycle[-1], eps, m, k)
            q_traj_cycle.append(q_new)
            p_traj_cycle.append(p_new)
        
        # Calculate energy change
        energy_initial = hamiltonian(q_start, p_start, m, k)
        energy_final = hamiltonian(q_traj_cycle[-1], p_traj_cycle[-1], m, k)
        energy_change = energy_final - energy_initial
        
        # Metropolis-Hastings accept/reject
        if energy_change <= 0 or np.random.random() < np.exp(-energy_change):
            accepted = True
            q, p = q_traj_cycle[-1], p_traj_cycle[-1]
        else:
            accepted = False
            q, p = q_start, p_start  # Reject: stay at initial point
        
        # Add trajectory to main arrays
        q_traj.extend(q_traj_cycle[1:])  # Skip first point (already in main array)
        p_traj.extend(p_traj_cycle[1:])
        
        # Momentum refresh
        p_refresh = np.random.normal(0, np.sqrt(m))
        q_traj.append(q)
        p_traj.append(p_refresh)
        
        # Store cycle information
        end_idx = len(q_traj) - 1
        cycle_info.append((start_idx, end_idx, accepted, energy_change))
        
        # Update for next cycle
        q, p = q, p_refresh
    
    q_traj = np.array(q_traj)
    p_traj = np.array(p_traj)
    
    # Animation parameters
    n_frames_per_trajectory = 15
    n_frames_per_metropolis_pause = 8  # Pause before showing decision
    n_frames_per_metropolis_decision = 10  # Show accept/reject decision
    n_frames_per_refresh = 8
    
    def animate(frame):
        ax.clear()
        
        # Draw energy contours
        ax.contour(Q, P, H, levels=levels, colors='gray', alpha=0.6, linewidths=1)
        
        # Determine current cycle and phase
        frames_per_cycle = n_frames_per_trajectory + n_frames_per_metropolis_pause + n_frames_per_metropolis_decision + n_frames_per_refresh
        cycle = frame // frames_per_cycle
        phase_frame = frame % frames_per_cycle
        
        if cycle >= n_cycles:
            cycle = n_cycles - 1
            phase_frame = frames_per_cycle - 1
        
        if phase_frame < n_frames_per_trajectory:
            # Trajectory phase
            phase_progress = phase_frame / n_frames_per_trajectory
            start_idx, end_idx, accepted, energy_change = cycle_info[cycle]
            
            # Calculate how much of trajectory to show
            traj_length = end_idx - start_idx - 1  # Exclude refresh point
            progress_steps = int(phase_progress * traj_length)
            current_end = start_idx + progress_steps + 1
            
            # Show trajectory up to current progress
            if current_end > start_idx:
                ax.plot(q_traj[start_idx:current_end], p_traj[start_idx:current_end], 
                       'b-', linewidth=3, alpha=0.8)
                ax.plot(q_traj[start_idx:current_end], p_traj[start_idx:current_end], 
                       'bo', markersize=6, alpha=0.8)
            
            # Current position
            if current_end <= len(q_traj):
                ax.plot(q_traj[current_end-1], p_traj[current_end-1], 'ro', markersize=12)
            
            title = f'Trajectory {cycle + 1}'
            
        elif phase_frame < n_frames_per_trajectory + n_frames_per_metropolis_pause:
            # Metropolis-Hastings pause phase - show trajectory but no decision yet
            start_idx, end_idx, accepted, energy_change = cycle_info[cycle]
            
            # Show full trajectory
            traj_length = end_idx - start_idx - 1
            ax.plot(q_traj[start_idx:start_idx+traj_length+1], p_traj[start_idx:start_idx+traj_length+1], 
                   'b-', linewidth=3, alpha=0.8)
            ax.plot(q_traj[start_idx:start_idx+traj_length+1], p_traj[start_idx:start_idx+traj_length+1], 
                   'bo', markersize=6, alpha=0.8)
            
            # Highlight both potential endpoints
            ax.plot(q_traj[start_idx], p_traj[start_idx], 'yo', markersize=12, label='Initial state')
            ax.plot(q_traj[end_idx-1], p_traj[end_idx-1], 'co', markersize=12, label='Proposed state')
            
            title = f'Metropolis: Evaluating ΔE = {energy_change:.3f}...'
            
        elif phase_frame < n_frames_per_trajectory + n_frames_per_metropolis_pause + n_frames_per_metropolis_decision:
            # Metropolis-Hastings decision phase
            start_idx, end_idx, accepted, energy_change = cycle_info[cycle]
            
            # Show full trajectory
            traj_length = end_idx - start_idx - 1
            ax.plot(q_traj[start_idx:start_idx+traj_length+1], p_traj[start_idx:start_idx+traj_length+1], 
                   'b-', linewidth=3, alpha=0.8)
            ax.plot(q_traj[start_idx:start_idx+traj_length+1], p_traj[start_idx:start_idx+traj_length+1], 
                   'bo', markersize=6, alpha=0.8)
            
            # Show accept/reject decision
            if accepted:
                ax.plot(q_traj[end_idx-1], p_traj[end_idx-1], 'go', markersize=15, label='Accepted')
                title = f'Metropolis: ACCEPTED (ΔE = {energy_change:.3f})'
            else:
                ax.plot(q_traj[start_idx], p_traj[start_idx], 'ro', markersize=15, label='Rejected')
                title = f'Metropolis: REJECTED (ΔE = {energy_change:.3f})'
            
        else:
            # Momentum refresh phase
            start_idx, end_idx, accepted, energy_change = cycle_info[cycle]
            
            # Show final state after Metropolis decision
            if accepted:
                final_q, final_p = q_traj[end_idx-1], p_traj[end_idx-1]
            else:
                final_q, final_p = q_traj[start_idx], p_traj[start_idx]
            
            # Show trajectory
            traj_length = end_idx - start_idx - 1
            ax.plot(q_traj[start_idx:start_idx+traj_length+1], p_traj[start_idx:start_idx+traj_length+1], 
                   'b-', linewidth=3, alpha=0.8)
            ax.plot(q_traj[start_idx:start_idx+traj_length+1], p_traj[start_idx:start_idx+traj_length+1], 
                   'bo', markersize=6, alpha=0.8)
            
            # Show momentum refresh
            refresh_q, refresh_p = q_traj[end_idx], p_traj[end_idx]
            ax.plot(final_q, final_p, 'go', markersize=12, label='Before refresh')
            ax.plot(refresh_q, refresh_p, 'mo', markersize=12, label='After refresh')
            
            # Draw arrow for momentum change
            ax.annotate('', xy=(float(refresh_q), float(refresh_p)), xytext=(float(final_q), float(final_p)),
                       arrowprops=dict(arrowstyle='->', color='purple', lw=2))
            
            title = f'Momentum Refresh {cycle + 1}'
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Position (q)', fontsize=12)
        ax.set_ylabel('Momentum (p)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    total_frames = n_cycles * (n_frames_per_trajectory + n_frames_per_metropolis_pause + n_frames_per_metropolis_decision + n_frames_per_refresh)
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=200)
    anim.save('hmc_metropolis.gif', writer='pillow')
    print("Saved: hmc_metropolis.gif")

def create_optimal_L_animation():
    """Show how to calculate optimal trajectory length L = epsilon * n / ESS"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Setup
    m, k = 1.0, 1.0
    eps = 0.5
    L_test = 1.0  # Test with L = 1
    n_leaps_test = int(L_test / eps)  # Number of leaps for L = 1
    n_steps_chain = 100  # Steps in the test chain
    
    # Run single chain at L = 1
    np.random.seed(42)
    q_samples = []
    p_samples = []
    
    q, p = 1.0, 1.0
    for step in range(n_steps_chain):
        # Generate trajectory
        q_traj = [q]
        p_traj = [p]
        
        for leap in range(n_leaps_test):
            q_new, p_new = leapfrog_step(q_traj[-1], p_traj[-1], eps, m, k)
            q_traj.append(q_new)
            p_traj.append(p_new)
        
        # Metropolis-Hastings accept/reject
        energy_initial = hamiltonian(q, p, m, k)
        energy_final = hamiltonian(q_traj[-1], p_traj[-1], m, k)
        energy_change = energy_final - energy_initial
        
        if energy_change <= 0 or np.random.random() < np.exp(-energy_change):
            q, p = q_traj[-1], p_traj[-1]  # Accept
        # else: reject (stay at current point)
        
        # Momentum refresh
        p = np.random.normal(0, np.sqrt(m))
        
        q_samples.append(q)
        p_samples.append(p)
    
    # Calculate ESS from the chain
    q_samples = np.array(q_samples)
    p_samples = np.array(p_samples)
    
    # Calculate autocorrelation
    def autocorr(x, maxlag=20):
        acf = []
        for lag in range(1, min(maxlag, len(x)//2)):
            corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
            if np.isnan(corr):
                corr = 0
            acf.append(corr)
        return np.array(acf)
    
    acf_q = autocorr(q_samples)
    
    # Calculate ESS using autocorrelation
    def ess_from_acf(acf):
        if len(acf) == 0:
            return 1
        # Sum autocorrelation function
        tau = 1 + 2 * np.sum(acf)
        return max(1, len(q_samples) / tau)
    
    ess_q = ess_from_acf(acf_q)
    
    # Calculate optimal L using the formula
    L_optimal = eps * n_leaps_test / ess_q
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # Plot 1: Chain samples
        if frame > 0:
            n_show = min(frame * 2, len(q_samples))  # Show samples progressively
            ax1.plot(q_samples[:n_show], p_samples[:n_show], 'b-', linewidth=1, alpha=0.6)
            ax1.plot(q_samples[:n_show], p_samples[:n_show], 'bo', markersize=3, alpha=0.8)
        
        ax1.set_xlabel('Position (q)', fontsize=12)
        ax1.set_ylabel('Momentum (p)', fontsize=12)
        ax1.set_title(f'HMC Chain at L = {L_test} (Step {frame})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2: Autocorrelation function
        if frame > 20:
            lags = np.arange(1, len(acf_q) + 1)
            ax2.plot(lags, acf_q, 'b-', linewidth=2, alpha=0.8, label='Position (q)')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Lag', fontsize=12)
        ax2.set_ylabel('Autocorrelation', fontsize=12)
        ax2.set_title('Position Autocorrelation Function', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Formula and calculation
        if frame > 30:
            ax3.text(0.1, 0.9, f'Test Parameters:', fontsize=14, fontweight='bold')
            ax3.text(0.1, 0.8, f'L_test = {L_test}', fontsize=12)
            ax3.text(0.1, 0.7, f'ε = {eps}', fontsize=12)
            ax3.text(0.1, 0.6, f'ESS = {ess_q:.1f}', fontsize=12)
            
            formula_text = r'$L_{\text{optimal}} = \epsilon \times \frac{n}{\text{ESS}}$'
            ax3.text(0.1, 0.5, formula_text, fontsize=14, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            ax3.text(0.1, 0.4, f'L_optimal = {eps} × {n_leaps_test} / {ess_q:.1f}', fontsize=12)
            ax3.text(0.1, 0.3, f'L_optimal = {L_optimal:.2f}', fontsize=16, color='red', fontweight='bold')
            
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.set_title('Optimal L Calculation', fontsize=14, fontweight='bold')
            ax3.axis('off')
    
    anim = animation.FuncAnimation(fig, animate, frames=60, interval=200)
    anim.save('optimal_L_calculation.gif', writer='pillow')
    print("Saved: optimal_L_calculation.gif")

def create_stationary_distribution_step_size_animation():
    """Show how stationary distribution changes with step size epsilon"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Setup
    m, k = 1.0, 1.0
    n_steps = 1000  # Number of steps to run for each epsilon
    eps_values = np.linspace(0.1, 1.8, 20)  # Range of step sizes to test
    
    # Grid for plotting distributions
    q_range = (-4, 4)
    p_range = (-4, 4)
    n_points = 100
    Q, P = np.meshgrid(np.linspace(q_range[0], q_range[1], n_points),
                       np.linspace(p_range[0], p_range[1], n_points))
    
    def exact_gaussian_density(q, p, eps):
        """Exact stationary distribution for discretized dynamics"""
        # Momentum is always standard normal
        p_density = np.exp(-p**2 / 2) / np.sqrt(2 * np.pi)
        
        # Position density depends on step size
        if eps < 2:  # Stable regime
            sigma_q = 1 / np.sqrt(1 - (eps**2 / 4))
            q_density = np.exp(-q**2 / (2 * sigma_q**2)) / (sigma_q * np.sqrt(2 * np.pi))
        else:  # Unstable regime
            q_density = np.zeros_like(q)
        
        return q_density * p_density
    
    def target_gaussian_density(q, p):
        """Target distribution (unit normal in both directions)"""
        return (np.exp(-q**2 / 2) / np.sqrt(2 * np.pi)) * (np.exp(-p**2 / 2) / np.sqrt(2 * np.pi))
    

    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # Current epsilon value
        eps_idx = frame % len(eps_values)
        eps = eps_values[eps_idx]
        
        # Calculate distributions
        exact_density = exact_gaussian_density(Q, P, eps)
        target_density = target_gaussian_density(Q, P)
        
        # Plot 1: Exact stationary distribution
        levels = np.linspace(0, 0.4, 20)
        contour = ax1.contour(Q, P, exact_density, levels=levels, colors='blue', alpha=0.7, linewidths=2)
        ax1.contourf(Q, P, exact_density, levels=levels, alpha=0.3, cmap='Blues')
        
        # Plot target distribution (unit normal) in red
        target_levels = np.linspace(0, 0.4, 10)
        ax1.contour(Q, P, target_density, levels=target_levels, colors='red', alpha=0.8, linewidths=1.5, linestyles='--')
        
        ax1.set_xlabel('Position (q)', fontsize=12)
        ax1.set_ylabel('Momentum (p)', fontsize=12)
        ax1.set_title(f'Stationary Distribution vs Target\nε = {eps:.2f}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.set_xlim(q_range)
        ax1.set_ylim(p_range)
        
        # Add legend
        from matplotlib.patches import Patch
        blue_patch = Patch(color='blue', alpha=0.7, label='Stationary Distribution')
        red_patch = Patch(color='red', alpha=0.8, label='Target (Unit Normal)')
        ax1.legend(handles=[blue_patch, red_patch], loc='upper right')
        
        # Add text showing the position variance
        if eps < 2:
            sigma_q = 1 / np.sqrt(1 - (eps**2 / 4))
            ax1.text(0.02, 0.98, f'σ_q = {sigma_q:.3f}', transform=ax1.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax1.text(0.02, 0.98, 'Unstable', transform=ax1.transAxes, 
                    fontsize=12, verticalalignment='top', color='red',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Bias in E[x²] as function of epsilon
        stable_eps = eps_values[eps_values < 2]
        unstable_eps = eps_values[eps_values >= 2]
        
        # Calculate variances for stable epsilons
        stable_variances = 1 / (1 - (stable_eps**2 / 4))
        target_variance = 1.0  # Target variance
        
        # Plot the bias curve
        ax2.plot(stable_eps, stable_variances, 'b-', linewidth=3, label='Discretized Variance')
        ax2.axhline(y=target_variance, color='red', linestyle='--', linewidth=2, label='Target Variance')
        
        # Highlight current epsilon
        if eps < 2:
            current_variance = 1 / (1 - (eps**2 / 4))
            ax2.plot(eps, current_variance, 'ko', markersize=10, label=f'Current: ε={eps:.2f}')
        else:
            ax2.plot(eps, 0, 'ro', markersize=10, label=f'Unstable: ε={eps:.2f}')
        
        ax2.set_xlabel('Step Size (ε)', fontsize=12)
        ax2.set_ylabel('Variance E[x²]', fontsize=12)
        ax2.set_title('Bias in Position Variance', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add bias calculation
        if eps < 2:
            bias = current_variance - target_variance
            ax2.text(0.02, 0.98, f'Bias = {bias:.3f}', transform=ax2.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax2.text(0.02, 0.98, 'Unstable', transform=ax2.transAxes, 
                    fontsize=12, verticalalignment='top', color='red',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create animation
    total_frames = len(eps_values) * 2  # Show each epsilon for 2 frames
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=300)
    anim.save('stationary_distribution_step_size.gif', writer='pillow')
    print("Saved: stationary_distribution_step_size.gif")

if __name__ == "__main__":

    # print("Creating HMC 1D log kinetic animation...")
    # create_hmc_logkinetic_animation_1d() 

    # # Generate the comprehensive visualization
    # print("Generating comprehensive HMC visualization...")
    # plot_comprehensive_hmc()
    
    # # Generate shadow Hamiltonian visualization
    # print("Generating shadow Hamiltonian visualization...")
    # plot_shadow_hamiltonian()
    
    # print("\nGenerating HMC animations...")
    
    # # Generate animations
    # print("Creating basic trajectory animation...")
    # create_basic_trajectory_animation()
    
    # print("Creating measure preservation animation...")
    # create_measure_preservation_animation()
    
    # print("Creating stationary distribution animation...")
    # create_stationary_distribution_animation()
    
    # print("Creating adaptive step size animation...")
    # create_adaptive_step_size_animation()
    
    # print("Creating momentum refresh animation...")
    # create_momentum_refresh_animation()
    
    # print("Creating Langevin dynamics animation...")
    # create_langevin_dynamics_animation()
    
    # print("Creating HMC Metropolis animation...")
    # create_hmc_metropolis_animation()

    print("Creating stationary distribution step size animation...")
    create_stationary_distribution_step_size_animation()

    # print("Creating optimal L animation...")
    # create_optimal_L_animation()

    # print("Creating microcanonical animation...")
    # create_microcanonical_animation()

    # print("Creating microcanonical 2D animation...")
    # create_microcanonical_animation_2d()
    
    print("\nAll files saved:")
    print("- hmc_comprehensive_visualization.png")
    print("- hmc_shadow_hamiltonian.png")
    print("- hmc_trajectory.gif")
    print("- hmc_measure_preservation.gif")
    print("- hmc_stationary_distribution.gif")
    print("- hmc_adaptive_step_size.gif")
    print("- hmc_momentum_refresh.gif")
    print("- langevin_dynamics.gif")
    print("- hmc_metropolis.gif") 
    print("- optimal_L_calculation.gif") 
    print("- stationary_distribution_step_size.gif")