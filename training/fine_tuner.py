"""
Joint Fine-Tuning Module
Fine-tunes both attitude and position controllers together
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fine_tune_cascaded_system(position_agent, attitude_agent, config, save_dir):
    """
    Joint fine-tuning of both controllers
    
    Args:
        position_agent: Trained DDPGAgent for position control
        attitude_agent: Trained DDPGAgent for attitude control
        config: Configuration for fine-tuning
        save_dir: Directory to save checkpoints
        
    Returns:
        stats: Training statistics
    """
    print("\n" + "="*70)
    print("PHASE 3: JOINT FINE-TUNING")
    print("="*70)
    print("Fine-tuning both controllers together on full trajectory")
    print(f"Episodes: {config.FINE_TUNE_EPISODES}")
    print(f"Position LR: {position_agent.actor_lr * 0.5:.2e} (50% of original)")
    print(f"Attitude LR: {attitude_agent.actor_lr * 0.3:.2e} (30% of original)")
    print(f"Exploration noise: 0.5x original")
    print("="*70 + "\n")
    
    # Create environment with both agents
    from environments.position_env import PositionEnv
    env = PositionEnv(attitude_agent=attitude_agent)
    
    # Set full trajectory
    env.set_trajectory_scale(1.0)
    
    # Reduce learning rates
    position_agent.reduce_learning_rate(0.5)
    attitude_agent.reduce_learning_rate(0.3)
    
    # Reduce exploration
    position_agent.set_noise_scale(0.04)  # Half of normal
    attitude_agent.set_noise_scale(0.05)
    
    stats = {
        'rewards': [],
        'pos_errors': [],
        'vel_errors': [],
        'att_errors': [],
        'successes': [],
        'episode_lengths': [],
        'position_losses': [],
        'attitude_losses': []
    }
    
    best_success_rate = 0
    consecutive_successes = 0
    
    for episode in range(config.FINE_TUNE_EPISODES):
        state = env.reset()
        
        episode_reward = 0
        episode_pos_errors = []
        episode_vel_errors = []
        episode_att_errors = []
        episode_length = 0
        done = False
        
        position_losses_ep = []
        attitude_losses_ep = []
        
        while not done:
            # Position agent selects body accelerations
            pos_action = position_agent.select_action(state, explore=True)
            
            # Control allocation (built into environment)
            # Environment handles: acc → [F_t, att_des] → attitude agent → torques
            next_state, reward, done, info = env.step(pos_action)
            
            # Store transition for position agent
            position_agent.replay_buffer.append((state, pos_action, reward, next_state, done))
            
            # Update position agent
            if position_agent.total_steps >= position_agent.warmup_steps:
                pos_loss = position_agent.update(batch_size=256)
                if pos_loss:
                    al = pos_loss.get('actor_loss')
                    if al is not None:
                        position_losses_ep.append(al)
            
            # Get attitude controller's state and action for its own learning
            # The attitude agent was called inside env.step(), but we need to
            # train it on the experience it just had
            att = state[6:9]  # attitude from full state
            rates = state[9:12]  # angular rates
            # Desired attitude is computed inside env.step by control allocation
            # We'll get it from the info dict
            if 'att_des' in info:
                att_des_deg = info['att_des']
                att_des = np.deg2rad(att_des_deg)
                
                # Reconstruct attitude controller's state
                att_state = np.concatenate([att, rates, att_des])
                
                # Get next attitude state
                next_att = next_state[6:9]
                next_rates = next_state[9:12]
                next_att_state = np.concatenate([next_att, next_rates, att_des])  # Same desired attitude
                
                # Attitude controller gets a shaped reward based on tracking
                att_error = np.linalg.norm(att - att_des)
                rate_error = np.linalg.norm(rates)
                att_reward = -10.0 * att_error**2 - 1.0 * rate_error**2
                if att_error < np.deg2rad(2):
                    att_reward += 5.0
                
                # Store transition for attitude agent
                attitude_agent.replay_buffer.append((att_state, info.get('torques', np.zeros(3)), 
                                                    att_reward, next_att_state, done))
                
                # Update attitude agent
                if attitude_agent.total_steps >= attitude_agent.warmup_steps:
                    att_loss = attitude_agent.update(batch_size=256)
                    if att_loss:
                        al = att_loss.get('actor_loss')
                        if al is not None:
                            attitude_losses_ep.append(al)
            
            # Track metrics
            episode_reward += reward
            episode_pos_errors.append(info['pos_error'])
            episode_vel_errors.append(info['vel_error'])
            if 'att_error' in info:
                episode_att_errors.append(info['att_error'])
            episode_length += 1
            
            position_agent.total_steps += 1
            attitude_agent.total_steps += 1
            
            state = next_state
        
        position_agent.episode_count += 1
        attitude_agent.episode_count += 1
        
        # Decay noise
        position_agent.noise_scale = max(0.005, position_agent.noise_scale * 0.9995)
        attitude_agent.noise_scale = max(0.005, attitude_agent.noise_scale * 0.9995)
        
        # Success check
        mean_pos_error = np.mean(episode_pos_errors)
        mean_vel_error = np.mean(episode_vel_errors)
        mean_att_error = np.mean(episode_att_errors) if episode_att_errors else 0
        
        success = (mean_pos_error < 15.0 and  # < 15mm
                  info.get('termination') == 'max_steps')
        
        if success:
            consecutive_successes += 1
        else:
            consecutive_successes = 0
        
        # Record stats
        stats['rewards'].append(episode_reward)
        stats['pos_errors'].append(mean_pos_error)
        stats['vel_errors'].append(mean_vel_error)
        stats['att_errors'].append(mean_att_error)
        stats['successes'].append(1 if success else 0)
        stats['episode_lengths'].append(episode_length)
        stats['position_losses'].append(np.mean(position_losses_ep) if position_losses_ep else 0)
        stats['attitude_losses'].append(np.mean(attitude_losses_ep) if attitude_losses_ep else 0)
        
        # Compute recent stats (always needed for best-model tracking)
        recent_rewards = np.mean(stats['rewards'][-100:]) if len(stats['rewards']) >= 100 else np.mean(stats['rewards'])
        recent_error = np.mean(stats['pos_errors'][-100:]) if len(stats['pos_errors']) >= 100 else np.mean(stats['pos_errors'])
        recent_success = np.mean(stats['successes'][-100:]) if len(stats['successes']) >= 100 else np.mean(stats['successes'])
        recent_att_error = np.mean(stats['att_errors'][-100:]) if len(stats['att_errors']) >= 100 else np.mean(stats['att_errors'])

        # Logging
        if (episode + 1) % 10 == 0 or episode < 5:
            print("env reset radius:", env.start_radius)
            print(f"Fine-tune Episode {episode+1}/{config.FINE_TUNE_EPISODES}")
            print(f"  Reward: {episode_reward:.1f} (Avg: {recent_rewards:.1f})")
            print(f"  Pos Error: {mean_pos_error:.1f}mm (Avg: {recent_error:.1f}mm)")
            print(f"  Att Error: {mean_att_error:.2f}° (Avg: {recent_att_error:.2f}°)")
            print(f"  Success: {'✓' if success else '✗'} (Rate: {recent_success:.1%})")
            print(f"  Consecutive: {consecutive_successes}")
            print(f"  Pos Noise: {position_agent.noise_scale:.4f}, Att Noise: {attitude_agent.noise_scale:.4f}")
            print()
        
        # Save checkpoints
        if (episode + 1) % 500 == 0:
            pos_path = f'{save_dir}/position_finetune_ep{episode+1}.pth'
            att_path = f'{save_dir}/attitude_finetune_ep{episode+1}.pth'
            position_agent.save(pos_path)
            attitude_agent.save(att_path)
            print(f"  → Checkpoints saved: ep{episode+1}\n")
        
        # Track best
        if recent_success > best_success_rate:
            best_success_rate = recent_success
            position_agent.save(f'{save_dir}/position_finetune_best.pth')
            attitude_agent.save(f'{save_dir}/attitude_finetune_best.pth')
        
        # Early stopping
        if consecutive_successes >= 50 and episode >= 1000:
            print(f"\n{'='*70}")
            print(f"✓✓✓ FINE-TUNING CONVERGED at episode {episode+1}!")
            print(f"Achieved {consecutive_successes} consecutive successes")
            print(f"Final position error: {mean_pos_error:.1f}mm")
            print(f"Final attitude error: {mean_att_error:.2f}°")
            print(f"{'='*70}\n")
            break
    
    # Final summary
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE")
    print("="*70)
    print(f"Total episodes: {episode+1}")
    print(f"Final success rate: {np.mean(stats['successes'][-100:]):.1%}")
    print(f"Final position error: {np.mean(stats['pos_errors'][-100:]):.1f}mm")
    print(f"Final attitude error: {np.mean(stats['att_errors'][-100:]):.2f}°")
    print(f"Best success rate: {best_success_rate:.1%}")
    print("="*70 + "\n")
    
    return stats