"""
Advanced Trajectory Visualization Script
Creates:
1. Animated video (.gif or .mp4) of trajectory tracking from multiple random starts
2. Multi-start comparison plots showing convergence from different positions
3. Detailed performance analysis

Compatible with your existing project structure
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
import os
from datetime import datetime

# Your existing imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.position_env import PositionEnv
from agents.td3_agent import TD3Agent
from configs.config import SystemConfig, AttitudeControllerConfig, PositionControllerConfig


def run_trajectory_from_start(env, position_agent, start_pos=None, max_time=None):
    """
    Run single episode from given start position
    
    Args:
        env: Position environment (with attitude agent loaded)
        position_agent: Trained position controller
        start_pos: [x, y, z] or None for default
        max_time: Maximum time in seconds (None = full trajectory)
        
    Returns:
        Dictionary with trajectory data
    """
    # Set max_steps if max_time specified
    if max_time is not None:
        original_max = env.max_steps
        env.max_steps = int(max_time / env.sys_cfg.DT)
    
    # Reset with specified start position
    state = env.reset(initial_pos=start_pos)
    
    # Storage
    data = {
        'time': [],
        'pos_actual': [],
        'pos_desired': [],
        'vel_actual': [],
        'vel_desired': [],
        'attitude': [],
        'att_desired': [],
        'pos_error': [],
        'vel_error': [],
        'thrust': [],
        'torques': [],
        'start_pos': env.state[0:3].copy(),
    }
    
    done = False
    while not done:
        # Get action
        action = position_agent.select_action(state, explore=False)
        
        # Step
        next_state, reward, done, info = env.step(action)
        
        # Record
        pos = env.state[0:3]
        vel = env.state[6:9]
        att = env.state[3:6]
        
        pos_des, vel_des, _ = env.get_trajectory(env.time, env.trajectory_scale)
        
        data['time'].append(env.time)
        data['pos_actual'].append(pos.copy())
        data['pos_desired'].append(pos_des.copy())
        data['vel_actual'].append(vel.copy())
        data['vel_desired'].append(vel_des.copy())
        data['attitude'].append(np.rad2deg(att.copy()))
        data['att_desired'].append(info['att_des'].copy())
        data['pos_error'].append(info['pos_error'] / 1000.0)  # Convert to meters
        data['vel_error'].append(info['vel_error'])
        data['thrust'].append(info['thrust'])
        data['torques'].append(info['torques'].copy())
        
        state = next_state
    
    data['termination'] = info.get('termination', 'unknown')
    
    # Convert to arrays
    for key in ['time', 'pos_actual', 'pos_desired', 'vel_actual', 'vel_desired',
                'attitude', 'att_desired', 'pos_error', 'vel_error', 'thrust', 'torques']:
        data[key] = np.array(data[key])
    
    if max_time is not None:
        env.max_steps = original_max
    
    return data


def create_animated_video(env, position_agent, save_path='trajectory_video.gif',
                         num_starts=3, start_radius=3.0, fps=20, duration=50):
    """
    Create animated video showing multiple trajectories from random starts
    
    Args:
        env: Position environment
        position_agent: Trained position controller
        save_path: Output path (.gif or .mp4)
        num_starts: Number of random starting positions
        start_radius: Radius for random starts (meters)
        fps: Frames per second
        duration: Video duration (seconds)
    """
    print("\n" + "="*70)
    print("CREATING ANIMATED TRAJECTORY VIDEO")
    print("="*70)
    
    # Generate random start positions
    print(f"\nGenerating {num_starts} random start positions (radius: {start_radius}m)...")
    start_positions = []
    for i in range(num_starts):
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(1.5, start_radius)  # Min 1.5m from origin
        z = 0 #np.random.uniform(0, 2.0)
        
        start_pos = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            z
        ])
        start_positions.append(start_pos)
        dist = np.linalg.norm(start_pos[:2])
        print(f"  Start {i+1}: ({start_pos[0]:6.2f}, {start_pos[1]:6.2f}, {start_pos[2]:5.2f}) "
              f"→ {dist:.2f}m from origin")
    
    # Run episodes
    print(f"\nRunning {num_starts} episodes...")
    episodes = []
    for i, start_pos in enumerate(start_positions):
        data = run_trajectory_from_start(env, position_agent, start_pos, max_time=duration)
        episodes.append(data)
        print(f"  Episode {i+1}: {len(data['time'])} steps, "
              f"final error: {data['pos_error'][-1]:.3f}m")
    
    # Create figure
    print(f"\nGenerating animation ({fps} fps)...")
    fig = plt.figure(figsize=(18, 10))
    
    # Create subplots
    ax_3d = fig.add_subplot(2, 3, (1, 4), projection='3d')  # Large 3D plot
    ax_xy = fig.add_subplot(2, 3, 2)                        # XY view
    ax_z = fig.add_subplot(2, 3, 3)                         # Altitude
    ax_error = fig.add_subplot(2, 3, 5)                     # Error
    ax_att = fig.add_subplot(2, 3, 6)                       # Attitude
    
    # Full desired trajectory
    t_ref = np.linspace(0, duration, 500)
    traj_ref = np.array([env.get_trajectory(t, env.trajectory_scale)[0] for t in t_ref])
    
    # Plot desired trajectory (static)
    ax_3d.plot(traj_ref[:, 0], traj_ref[:, 1], traj_ref[:, 2], 
              'k--', linewidth=2, alpha=0.4, label='Desired')
    ax_xy.plot(traj_ref[:, 0], traj_ref[:, 1], 
              'k--', linewidth=2, alpha=0.4, label='Desired')
    ax_z.plot(t_ref, traj_ref[:, 2], 
             'k--', linewidth=2, alpha=0.4, label='Desired')
    
    # Origin marker
    ax_3d.scatter([0], [0], [0], c='gold', s=400, marker='*', 
                 edgecolors='black', linewidths=2, label='Origin', zorder=100)
    ax_xy.scatter([0], [0], c='gold', s=300, marker='*',
                 edgecolors='black', linewidths=2, label='Origin', zorder=100)
    
    # Initialize plot elements for each episode
    colors = plt.cm.rainbow(np.linspace(0, 1, num_starts))
    lines_3d, markers_3d = [], []
    lines_xy, markers_xy = [], []
    lines_z, lines_error, lines_roll, lines_pitch = [], [], [], []
    
    for i, color in enumerate(colors):
        # 3D trajectory
        line_3d, = ax_3d.plot([], [], [], color=color, linewidth=2.5, 
                              alpha=0.8, label=f'Drone {i+1}')
        marker_3d = ax_3d.scatter([], [], [], c=[color], s=200, 
                                 marker='o', edgecolors='black', linewidths=2)
        lines_3d.append(line_3d)
        markers_3d.append(marker_3d)
        
        # XY view
        line_xy, = ax_xy.plot([], [], color=color, linewidth=2.5, alpha=0.8)
        marker_xy = ax_xy.scatter([], [], c=[color], s=150, 
                                 marker='o', edgecolors='black', linewidths=2)
        lines_xy.append(line_xy)
        markers_xy.append(marker_xy)
        
        # Other plots
        line_z, = ax_z.plot([], [], color=color, linewidth=2, alpha=0.8)
        line_error, = ax_error.plot([], [], color=color, linewidth=2, alpha=0.8)
        line_roll, = ax_att.plot([], [], color=color, linewidth=1.5, linestyle='-', alpha=0.8)
        line_pitch, = ax_att.plot([], [], color=color, linewidth=1.5, linestyle='--', alpha=0.8)
        
        lines_z.append(line_z)
        lines_error.append(line_error)
        lines_roll.append(line_roll)
        lines_pitch.append(line_pitch)
    
    # Configure 3D plot
    max_r = max(np.max(np.abs(traj_ref[:, :2])), start_radius) + 2
    ax_3d.set_xlim(-max_r, max_r)
    ax_3d.set_ylim(-max_r, max_r)
    ax_3d.set_zlim(0, traj_ref[:, 2].max() + 5)
    ax_3d.set_xlabel('X (m)', fontsize=10)
    ax_3d.set_ylabel('Y (m)', fontsize=10)
    ax_3d.set_zlabel('Z (m)', fontsize=10)
    ax_3d.set_title('3D Trajectory Tracking', fontweight='bold', fontsize=12)
    ax_3d.legend(loc='upper left', fontsize=8)
    ax_3d.grid(True, alpha=0.3)
    ax_3d.view_init(elev=20, azim=45)
    
    # Configure XY plot
    ax_xy.set_xlim(-max_r, max_r)
    ax_xy.set_ylim(-max_r, max_r)
    ax_xy.set_xlabel('X (m)', fontsize=10)
    ax_xy.set_ylabel('Y (m)', fontsize=10)
    ax_xy.set_title('Top View (XY Plane)', fontweight='bold', fontsize=12)
    ax_xy.axis('equal')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc='upper right', fontsize=8)
    
    # Draw start radius circle
    circle = plt.Circle((0, 0), start_radius, fill=False, color='gray',
                       linestyle=':', linewidth=2, alpha=0.5)
    ax_xy.add_patch(circle)
    
    # Configure altitude plot
    ax_z.set_xlim(0, duration)
    ax_z.set_ylim(0, traj_ref[:, 2].max() + 5)
    ax_z.set_xlabel('Time (s)', fontsize=10)
    ax_z.set_ylabel('Altitude (m)', fontsize=10)
    ax_z.set_title('Altitude vs Time', fontweight='bold', fontsize=12)
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(loc='lower right', fontsize=8)
    
    # Configure error plot
    ax_error.set_xlim(0, duration)
    ax_error.set_ylim(0, 5)
    ax_error.axhline(y=0.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='0.5m')
    ax_error.axhline(y=1.0, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='1.0m')
    ax_error.set_xlabel('Time (s)', fontsize=10)
    ax_error.set_ylabel('Position Error (m)', fontsize=10)
    ax_error.set_title('Tracking Error', fontweight='bold', fontsize=12)
    ax_error.grid(True, alpha=0.3)
    ax_error.legend(loc='upper right', fontsize=8)
    
    # Configure attitude plot
    ax_att.set_xlim(0, duration)
    ax_att.set_ylim(-25, 25)
    ax_att.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_att.axhline(y=20, color='red', linestyle=':', linewidth=1, alpha=0.5, label='±20° limit')
    ax_att.axhline(y=-20, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax_att.set_xlabel('Time (s)', fontsize=10)
    ax_att.set_ylabel('Angle (deg)', fontsize=10)
    ax_att.set_title('Attitude (solid=roll, dash=pitch)', fontweight='bold', fontsize=12)
    ax_att.grid(True, alpha=0.3)
    ax_att.legend(loc='upper right', fontsize=8)
    
    # Time display
    time_text = ax_3d.text2D(0.02, 0.98, '', transform=ax_3d.transAxes,
                            fontsize=14, fontweight='bold', verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Animation parameters
    dt = env.sys_cfg.DT
    frame_skip = max(1, int(1 / (fps * dt)))
    max_frames = min(len(ep['time']) for ep in episodes)
    frame_indices = range(0, max_frames, frame_skip)
    
    def init():
        """Initialize animation"""
        for i in range(num_starts):
            lines_3d[i].set_data([], [])
            lines_3d[i].set_3d_properties([])
            markers_3d[i]._offsets3d = ([], [], [])
            lines_xy[i].set_data([], [])
            markers_xy[i].set_offsets(np.empty((0, 2)))
            lines_z[i].set_data([], [])
            lines_error[i].set_data([], [])
            lines_roll[i].set_data([], [])
            lines_pitch[i].set_data([], [])
        time_text.set_text('')
        return (lines_3d + markers_3d + lines_xy + markers_xy + 
                lines_z + lines_error + lines_roll + lines_pitch + [time_text])
    
    def animate(frame_num):
        """Update animation frame"""
        idx = frame_indices[frame_num]
        
        for i, ep in enumerate(episodes):
            if idx < len(ep['time']):
                # Update 3D trajectory
                lines_3d[i].set_data(ep['pos_actual'][:idx, 0], ep['pos_actual'][:idx, 1])
                lines_3d[i].set_3d_properties(ep['pos_actual'][:idx, 2])
                markers_3d[i]._offsets3d = ([ep['pos_actual'][idx, 0]], 
                                            [ep['pos_actual'][idx, 1]], 
                                            [ep['pos_actual'][idx, 2]])
                
                # Update XY view
                lines_xy[i].set_data(ep['pos_actual'][:idx, 0], ep['pos_actual'][:idx, 1])
                markers_xy[i].set_offsets([[ep['pos_actual'][idx, 0], ep['pos_actual'][idx, 1]]])
                
                # Update other plots
                lines_z[i].set_data(ep['time'][:idx], ep['pos_actual'][:idx, 2])
                lines_error[i].set_data(ep['time'][:idx], ep['pos_error'][:idx])
                lines_roll[i].set_data(ep['time'][:idx], ep['attitude'][:idx, 0])
                lines_pitch[i].set_data(ep['time'][:idx], ep['attitude'][:idx, 1])
        
        # Update time
        current_time = episodes[0]['time'][min(idx, len(episodes[0]['time'])-1)]
        time_text.set_text(f'Time: {current_time:.2f}s')
        
        return (lines_3d + markers_3d + lines_xy + markers_xy + 
                lines_z + lines_error + lines_roll + lines_pitch + [time_text])
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(frame_indices),
                        interval=1000/fps, blit=True, repeat=True)
    
    # Save
    print(f"Saving animation to: {save_path}")
    if save_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
    elif save_path.endswith('.mp4'):
        anim.save(save_path, writer='ffmpeg', fps=fps, dpi=150)
    
    plt.tight_layout()
    print("✓ Animation saved!")
    print("="*70)
    
    return fig, anim, episodes


def plot_multi_start_comparison(env, position_agent, num_starts=8, 
                                start_radius=5.0, save_dir='results/multi_start'):
    """
    Create detailed comparison plots from multiple random starts
    
    Args:
        env: Position environment
        position_agent: Trained position controller
        num_starts: Number of random starting positions
        start_radius: Maximum radius for starts (meters)
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"MULTI-START TRAJECTORY COMPARISON ({num_starts} starts)")
    print("="*70)
    
    # Generate random starts
    print(f"\nGenerating random starts (radius: 0-{start_radius}m)...")
    starts = []
    for i in range(num_starts):
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(1.0, start_radius)
        z = 0 #np.random.uniform(0, 2.0)
        
        start = np.array([radius * np.cos(angle), radius * np.sin(angle), z])
        starts.append(start)
        print(f"  Start {i+1}: ({start[0]:6.2f}, {start[1]:6.2f}, {start[2]:5.2f}) "
              f"→ {np.linalg.norm(start[:2]):.2f}m from origin")
    
    # Run episodes
    print("\nRunning episodes...")
    episodes = []
    for i, start in enumerate(starts):
        ep = run_trajectory_from_start(env, position_agent, start)
        episodes.append(ep)
        print(f"  Episode {i+1}: {len(ep['time'])} steps, "
              f"final error: {ep['pos_error'][-1]:.3f}m, term: {ep['termination']}")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_starts))
    
    # Get reference trajectory
    max_time = max(ep['time'][-1] for ep in episodes)
    t_ref = np.linspace(0, max_time, 500)
    traj_ref = np.array([env.get_trajectory(t, env.trajectory_scale)[0] for t in t_ref])
    
    # ============ 3D Trajectory ============
    ax1 = fig.add_subplot(2, 4, (1, 5), projection='3d')
    ax1.plot(traj_ref[:, 0], traj_ref[:, 1], traj_ref[:, 2],
            'k-', linewidth=3, alpha=0.4, label='Desired')
    
    for i, (ep, color) in enumerate(zip(episodes, colors)):
        ax1.plot(ep['pos_actual'][:, 0], ep['pos_actual'][:, 1], ep['pos_actual'][:, 2],
                color=color, linewidth=2, alpha=0.7)
        # Start marker
        ax1.scatter([ep['start_pos'][0]], [ep['start_pos'][1]], [ep['start_pos'][2]],
                   c=[color], s=150, marker='o', edgecolors='black', linewidths=2)
        # End marker
        ax1.scatter([ep['pos_actual'][-1, 0]], [ep['pos_actual'][-1, 1]], 
                   [ep['pos_actual'][-1, 2]],
                   c=[color], s=150, marker='s', edgecolors='black', linewidths=2)
    
    ax1.scatter([0], [0], [0], c='gold', s=500, marker='*',
               edgecolors='black', linewidths=3, label='Origin', zorder=100)
    
    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_zlabel('Z (m)', fontsize=11)
    ax1.set_title('3D Trajectories from Multiple Starts', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.view_init(elev=25, azim=45)
    
    # ============ XY View ============
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.plot(traj_ref[:, 0], traj_ref[:, 1], 'k-', linewidth=3, alpha=0.4, label='Desired')
    
    for i, (ep, color) in enumerate(zip(episodes, colors)):
        ax2.plot(ep['pos_actual'][:, 0], ep['pos_actual'][:, 1],
                color=color, linewidth=2, alpha=0.7, label=f'Start {i+1}')
        ax2.scatter([ep['start_pos'][0]], [ep['start_pos'][1]],
                   c=[color], s=120, marker='o', edgecolors='black', linewidths=2)
    
    ax2.scatter([0], [0], c='gold', s=400, marker='*',
               edgecolors='black', linewidths=3, label='Origin', zorder=100)
    
    # Draw radius circle
    circle = plt.Circle((0, 0), start_radius, fill=False, color='gray',
                       linestyle='--', linewidth=2, alpha=0.5, label=f'{start_radius}m radius')
    ax2.add_patch(circle)
    
    ax2.set_xlabel('X (m)', fontsize=11)
    ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_title('XY Plane View', fontweight='bold', fontsize=13)
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=7, ncol=2)
    
    # ============ Position Error ============
    ax3 = fig.add_subplot(2, 4, 3)
    for i, (ep, color) in enumerate(zip(episodes, colors)):
        ax3.plot(ep['time'], ep['pos_error'], color=color, linewidth=2, alpha=0.7)
    
    ax3.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.6, label='0.5m')
    ax3.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='1.0m')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Position Error (m)', fontsize=11)
    ax3.set_title('Tracking Error Evolution', fontweight='bold', fontsize=13)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, min(8, max(ep['pos_error'].max() for ep in episodes) + 1))
    
    # ============ Altitude ============
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.plot(t_ref, traj_ref[:, 2], 'k-', linewidth=3, alpha=0.4, label='Desired')
    
    for i, (ep, color) in enumerate(zip(episodes, colors)):
        ax4.plot(ep['time'], ep['pos_actual'][:, 2], color=color, linewidth=2, alpha=0.7)
    
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Altitude (m)', fontsize=11)
    ax4.set_title('Altitude Tracking', fontweight='bold', fontsize=13)
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # ============ Convergence Analysis ============
    ax5 = fig.add_subplot(2, 4, 6)
    
    conv_times = []
    final_errors = []
    mean_errors = []
    initial_dists = []
    
    for i, ep in enumerate(episodes):
        # Convergence time (when error < 1m)
        conv_idx = np.where(ep['pos_error'] < 1.0)[0]
        conv_time = ep['time'][conv_idx[0]] if len(conv_idx) > 0 else ep['time'][-1]
        conv_times.append(conv_time)
        
        # Errors
        final_errors.append(ep['pos_error'][-1])
        mean_errors.append(np.mean(ep['pos_error']))
        
        # Initial distance
        initial_dists.append(np.linalg.norm(ep['start_pos'][:2]))
    
    x = np.arange(1, num_starts + 1)
    width = 0.3
    
    bars1 = ax5.bar(x - width, conv_times, width, label='Conv. Time (s)', color='steelblue')
    ax5_twin = ax5.twinx()
    bars2 = ax5_twin.bar(x, final_errors, width, label='Final Error (m)', color='coral')
    bars3 = ax5_twin.bar(x + width, mean_errors, width, label='Mean Error (m)', color='lightgreen')
    
    ax5.set_xlabel('Episode', fontsize=11)
    ax5.set_ylabel('Convergence Time (s)', fontsize=11, color='steelblue')
    ax5_twin.set_ylabel('Error (m)', fontsize=11, color='coral')
    ax5.set_title('Performance Metrics', fontweight='bold', fontsize=13)
    ax5.set_xticks(x)
    ax5.tick_params(axis='y', labelcolor='steelblue')
    ax5_twin.tick_params(axis='y', labelcolor='coral')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Combined legend
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    # ============ Attitude ============
    ax6 = fig.add_subplot(2, 4, 7)
    for i, (ep, color) in enumerate(zip(episodes, colors)):
        ax6.plot(ep['time'], ep['attitude'][:, 0], color=color, 
                linewidth=1.5, linestyle='-', alpha=0.7, label=f'Roll {i+1}')
        ax6.plot(ep['time'], ep['attitude'][:, 1], color=color,
                linewidth=1.5, linestyle='--', alpha=0.7)
    
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax6.axhline(y=20, color='red', linestyle=':', linewidth=1, alpha=0.5, label='±20° limit')
    ax6.axhline(y=-20, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('Angle (deg)', fontsize=11)
    ax6.set_title('Attitude (solid=roll, dash=pitch)', fontweight='bold', fontsize=13)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-30, 30)
    
    # ============ Statistics ============
    ax7 = fig.add_subplot(2, 4, 8)
    ax7.axis('off')
    
    stats_text = [
        "PERFORMANCE STATISTICS",
        "="*45,
        f"Number of Episodes: {num_starts}",
        f"Start Radius: 0-{start_radius}m",
        "",
        "Convergence (<1m):",
        f"  Mean time: {np.mean(conv_times):.2f}s",
        f"  Min time:  {np.min(conv_times):.2f}s",
        f"  Max time:  {np.max(conv_times):.2f}s",
        "",
        "Final Error:",
        f"  Mean: {np.mean(final_errors):.3f}m",
        f"  Min:  {np.min(final_errors):.3f}m",
        f"  Max:  {np.max(final_errors):.3f}m",
        "",
        "Mean Error (entire trajectory):",
        f"  Mean: {np.mean(mean_errors):.3f}m",
        f"  Min:  {np.min(mean_errors):.3f}m",
        f"  Max:  {np.max(mean_errors):.3f}m",
        "",
        "Success Rate:",
        f"  <0.5m: {sum(1 for e in final_errors if e < 0.5)/num_starts*100:.0f}%",
        f"  <1.0m: {sum(1 for e in final_errors if e < 1.0)/num_starts*100:.0f}%",
        f"  <2.0m: {sum(1 for e in final_errors if e < 2.0)/num_starts*100:.0f}%",
        "",
        "Terminations:",
    ]
    
    term_counts = {}
    for ep in episodes:
        term = ep['termination']
        term_counts[term] = term_counts.get(term, 0) + 1
    
    for term, count in term_counts.items():
        stats_text.append(f"  {term}: {count}")
    
    ax7.text(0.1, 0.5, '\n'.join(stats_text), fontsize=9, family='monospace',
            verticalalignment='center', transform=ax7.transAxes)
    
    plt.suptitle(f'Multi-Start Trajectory Analysis ({num_starts} Random Starts)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save
    save_path = f'{save_dir}/multi_start_comparison_{num_starts}starts.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {save_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Mean convergence time: {np.mean(conv_times):.2f}s")
    print(f"Mean final error: {np.mean(final_errors):.3f}m")
    print(f"Success rate (<1m): {sum(1 for e in final_errors if e < 1.0)/num_starts*100:.0f}%")
    print("="*70)
    
    return episodes


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trajectory Visualization')
    parser.add_argument('--attitude_agent', type=str, default=r'results\run_20260322_101645\attitude_checkpoint_ep7000.pth', #'results/run_20260227_150442/attitude_controller_final.pth',
                       help='Path to attitude agent .pth')
    parser.add_argument('--position_agent', type=str, default=r'weights/position_best.pth', #'results/run_20260301_102438/position_phase_2_ep3000.pth',
                       help='Path to position agent .pth')
    parser.add_argument('--video', action='store_true',
                       help='Create animated video')
    parser.add_argument('--multi_start', action='store_true',
                       help='Create multi-start comparison')
    parser.add_argument('--num_starts', type=int, default=8,
                       help='Number of random starts')
    parser.add_argument('--start_radius', type=float, default=3.0,
                       help='Max start radius (m)')
    parser.add_argument('--video_starts', type=int, default=3,
                       help='Number of starts in video')
    parser.add_argument('--fps', type=int, default=20,
                       help='Video FPS')
    parser.add_argument('--duration', type=float, default=50.0,
                       help='Video duration (s)')
    parser.add_argument('--output_dir', type=str, default='results/visualization2',
                       help='Output directory')
    parser.add_argument('--video_name', type=str, default='trajectory.gif',
                       help='Video filename (.gif or .mp4)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("TRAJECTORY VISUALIZATION TOOL")
    print("="*70)
    
    # Load agents
    print("\nLoading agents...")
    
    att_cfg = AttitudeControllerConfig()
    attitude_agent = TD3Agent(
        state_dim=att_cfg.STATE_DIM,
        action_dim=att_cfg.ACTION_DIM,
        max_action=att_cfg.MAX_TORQUE,
        hidden_dims=att_cfg.HIDDEN_DIMS
    )
    attitude_agent.load(args.attitude_agent)
    print(f"  ✓ Attitude: {args.attitude_agent}")
    
    pos_cfg = PositionControllerConfig()
    position_agent = TD3Agent(
        state_dim=pos_cfg.STATE_DIM,
        action_dim=pos_cfg.ACTION_DIM,
        max_action=pos_cfg.MAX_BODY_ACCELERATION,
        hidden_dims=pos_cfg.HIDDEN_DIMS
    )
    position_agent.load(args.position_agent)
    print(f"  ✓ Position: {args.position_agent}")
    
    # Create environment
    env = PositionEnv(attitude_agent=attitude_agent)
    env.set_trajectory_scale(1.0)
    env.set_wind([0.0, 0.0, 0.0])
    
    # Generate visualizations
    if args.multi_start:
        plot_multi_start_comparison(
            env, position_agent,
            num_starts=args.num_starts,
            start_radius=args.start_radius,
            save_dir=args.output_dir
        )
    
    if args.video:
        video_path = os.path.join(args.output_dir, args.video_name)
        create_animated_video(
            env, position_agent,
            save_path=video_path,
            num_starts=args.video_starts,
            start_radius=args.start_radius,
            fps=args.fps,
            duration=args.duration
        )
    
    print(f"\n✓ Complete! Results in: {args.output_dir}")


if __name__ == "__main__":
    main()