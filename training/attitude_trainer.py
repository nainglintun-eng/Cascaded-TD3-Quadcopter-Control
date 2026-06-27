"""
Attitude Controller Training Module
Trains the inner-loop attitude stabilization controller
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_attitude_controller(env, agent, config, save_dir):
    """
    Train the attitude controller (inner loop)
    
    Args:
        env: AttitudeEnv instance
        agent: TD3Agent instance
        config: AttitudeControllerConfig instance
        save_dir: Directory to save checkpoints
        
    Returns:
        stats: Dictionary with training statistics
    """
    print("\n" + "="*70)
    print("TRAINING ATTITUDE CONTROLLER (INNER LOOP)")
    print("="*70)
    print(f"Episodes: {config.EPISODES_PER_PHASE}")
    print(f"Success criterion: < {np.rad2deg(config.SUCCESS_THRESHOLD):.1f}° error")
    print(f"Required consecutive successes: {config.CONSECUTIVE_SUCCESSES}")
    print("="*70 + "\n")
    
    stats = {
        'rewards': [],
        'att_errors': [],
        'rate_errors': [],
        'successes': [],
        'episode_lengths': []
    }
    
    consecutive_successes = 0
    best_success_rate = 0
    agent.set_noise_scale(config.EXPLORATION_NOISE)
    
    for episode in range(config.EPISODES_PER_PHASE):
        # Reset environment with random desired attitude
        state = env.reset()
        episode_reward = 0
        episode_att_errors = []
        episode_rate_errors = []
        episode_length = 0
        done = False
        
        # Episode loop
        while not done:
            # Select action
            if agent.total_steps < config.WARMUP_STEPS:
                # Random exploration during warmup
                action = np.random.uniform(-config.MAX_TORQUE, config.MAX_TORQUE, 
                                          config.ACTION_DIM)
            else:
                action = agent.select_action(state, explore=True)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.replay_buffer.append((state, action, reward, next_state, done))
            
            # Update agent (after warmup)
            if agent.total_steps >= config.WARMUP_STEPS:
                losses = agent.update(batch_size=config.BATCH_SIZE)
            
            # Track metrics
            episode_reward += reward
            episode_att_errors.append(info['att_error'])
            episode_rate_errors.append(info['rate_error'])
            episode_length += 1
            agent.total_steps += 1
            
            state = next_state
        
        # Episode complete
        agent.episode_count += 1
        
        # Decay exploration noise
        agent.noise_scale = max(config.MIN_NOISE, 
                               agent.noise_scale * config.NOISE_DECAY)
        
        # Check success
        mean_att_error = np.mean(episode_att_errors)
        mean_rate_error = np.mean(episode_rate_errors)
        success = (mean_att_error < np.rad2deg(config.SUCCESS_THRESHOLD) and 
                  info.get('termination') == 'max_steps')
        
        if success:
            consecutive_successes += 1
        else:
            consecutive_successes = 0
        
        # Record stats
        stats['rewards'].append(episode_reward)
        stats['att_errors'].append(mean_att_error)
        stats['rate_errors'].append(mean_rate_error)
        stats['successes'].append(1 if success else 0)
        stats['episode_lengths'].append(episode_length)
        
        # Logging
        if (episode + 1) % 10 == 0 or episode < 5:
            recent_rewards = np.mean(stats['rewards'][-100:]) if len(stats['rewards']) >= 100 else np.mean(stats['rewards'])
            recent_att_error = np.mean(stats['att_errors'][-100:]) if len(stats['att_errors']) >= 100 else np.mean(stats['att_errors'])
            recent_success = np.mean(stats['successes'][-100:]) if len(stats['successes']) >= 100 else np.mean(stats['successes'])
            
            print(f"Episode {episode+1}/{config.EPISODES_PER_PHASE}")
            print(f"  Reward: {episode_reward:.1f} (Avg: {recent_rewards:.1f})")
            print(f"  Att Error: {mean_att_error:.2f}° (Avg: {recent_att_error:.2f}°)")
            print(f"  Rate Error: {mean_rate_error:.3f} rad/s")
            print(f"  Length: {episode_length}/{env.max_steps}")
            print(f"  Success: {'✓' if success else '✗'} (Rate: {recent_success:.1%})")
            print(f"  Consecutive: {consecutive_successes}/{config.CONSECUTIVE_SUCCESSES}")
            print(f"  Noise: {agent.noise_scale:.4f}")
            print()
        
        # Save checkpoint
        if (episode + 1) % 500 == 0:
            checkpoint_path = f'{save_dir}/attitude_checkpoint_ep{episode+1}.pth'
            agent.save(checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}\n")
        
        # Track best model
        if recent_success > best_success_rate:
            best_success_rate = recent_success
            best_path = f'{save_dir}/attitude_best.pth'
            agent.save(best_path)
        
        # Early stopping
        if consecutive_successes >= config.CONSECUTIVE_SUCCESSES and episode >= 500:
            print(f"\n{'='*70}")
            print(f"✓✓✓ CONVERGED at episode {episode+1}!")
            print(f"Achieved {consecutive_successes} consecutive successes")
            print(f"Final attitude error: {mean_att_error:.2f}°")
            print(f"{'='*70}\n")
            
            # Pad stats for remaining episodes
            remaining = config.EPISODES_PER_PHASE - (episode + 1)
            for _ in range(remaining):
                stats['rewards'].append(stats['rewards'][-1])
                stats['att_errors'].append(stats['att_errors'][-1])
                stats['rate_errors'].append(stats['rate_errors'][-1])
                stats['successes'].append(1)
                stats['episode_lengths'].append(stats['episode_lengths'][-1])
            break
    
    # Final summary
    print("\n" + "="*70)
    print("ATTITUDE CONTROLLER TRAINING COMPLETE")
    print("="*70)
    print(f"Total episodes: {episode+1}")
    print(f"Final success rate: {np.mean(stats['successes'][-100:]):.1%}")
    print(f"Final attitude error: {np.mean(stats['att_errors'][-100:]):.2f}°")
    print(f"Best success rate: {best_success_rate:.1%}")
    print("="*70 + "\n")
    
    return stats
