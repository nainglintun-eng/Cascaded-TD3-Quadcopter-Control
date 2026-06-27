"""
Position Controller Trainer V2 - OPTIMIZED
Key improvements:
1. Adaptive exploration per curriculum phase
2. Early stopping (don't waste time)
3. Curriculum-aware reward configuration
4. Better logging and monitoring
5. Episode length scaling
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_position_controller(env, agent, config, save_dir):
    """
    Train position controller with optimized curriculum
    
    Improvements:
    - Adaptive exploration per phase
    - Early stopping
    - Episode length scaling
    - Better logging
    """
    print("\n" + "="*70)
    print("POSITION CONTROLLER TRAINING V2 - OPTIMIZED CURRICULUM")
    print("="*70)
    print(f"Total phases: {len(config.CURRICULUM_PHASES)}")
    
    total_episodes = sum(p['episodes'] for p in config.CURRICULUM_PHASES)
    print(f"Total episodes: {total_episodes}")
    
    for i, phase in enumerate(config.CURRICULUM_PHASES):
        duration = phase['max_steps'] * env.sys_cfg.DT
        print(f"  Phase {i+1}: {phase['name']}")
        print(f"    Episodes: {phase['episodes']}, Duration: {duration:.1f}s")
        print(f"    Exploration: {phase['exploration_start']:.2f}→{phase['exploration_end']:.2f}")
    print("="*70 + "\n")
    
    all_stats = {
        'rewards': [],
        'pos_errors': [],
        'vel_errors': [],
        'successes': [],
        'episode_lengths': [],
        'phases': [],
        'q_values': [],
        'exploration_noise': []
    }
    
    for phase_idx, phase_config in enumerate(config.CURRICULUM_PHASES):
        print(f"\n{'='*70}")
        print(f"PHASE {phase_idx+1}/{len(config.CURRICULUM_PHASES)}: {phase_config['name'].upper()}")
        print(f"{'='*70}")
        print(f"Trajectory scale: {phase_config['trajectory_scale']*100:.0f}%")
        print(f"Episode length: {phase_config['max_steps']} steps ({phase_config['max_steps']*env.sys_cfg.DT:.1f}s)")
        print(f"Target error: < {phase_config['success_error']*1000:.0f}mm")
        print(f"Episodes: {phase_config['episodes']}")
        print(f"Required successes: {phase_config['required_successes']}")
        print(f"Wind: {phase_config['wind_std']}")
        print(f"Start radius: {phase_config['start_radius']}m")
        print(f"Exploration: {phase_config['exploration_start']:.2f} → {phase_config['exploration_end']:.2f}")
        print(f"{'='*70}\n")
        
        # Configure environment for this phase
        env.set_trajectory_scale(phase_config['trajectory_scale'])
        env.set_wind(phase_config.get('wind_std', [0, 0, 0]))
        env.set_start_radius(phase_config.get('start_radius', 0.0))
        
        # Set episode length for this phase (CRITICAL FIX!)
        env.max_steps = phase_config['max_steps']
        
        # Set reward configuration for this phase
        reward_config_key = phase_config.get('reward_config_key', 'helix_full')
        env.set_reward_config(reward_config_key)
        
        # Train this phase
        phase_stats = train_position_phase(
            env, agent, phase_config, phase_idx+1, save_dir
        )
        
        # Accumulate stats
        for key in ['rewards', 'pos_errors', 'vel_errors', 'successes', 
                    'episode_lengths', 'q_values', 'exploration_noise']:
            all_stats[key].extend(phase_stats[key])
        all_stats['phases'].extend([phase_idx+1] * len(phase_stats['rewards']))
        
        # Save phase checkpoint
        phase_path = f'{save_dir}/position_phase_{phase_idx+1}.pth'
        agent.save(phase_path)
        print(f"\n✓ Phase {phase_idx+1} complete! Saved to {phase_path}\n")
    
    # Final summary
    print("\n" + "="*70)
    print("POSITION CONTROLLER TRAINING COMPLETE")
    print("="*70)
    print(f"Total episodes: {len(all_stats['rewards'])}")
    print(f"Final success rate: {np.mean(all_stats['successes'][-100:]):.1%}")
    print(f"Final position error: {np.mean(all_stats['pos_errors'][-100:]):.1f}mm")
    print(f"Final Q-value: {np.mean(all_stats['q_values'][-100:]):.2f}")
    print("="*70 + "\n")
    
    return all_stats


def train_position_phase(env, agent, phase_config, phase_num, save_dir):
    """
    Train a single curriculum phase with adaptive exploration
    
    Key improvements:
    - Adaptive exploration decay
    - Early stopping
    - Better metrics tracking
    """
    from configs.config import TrainingConfig
    
    stats = {
        'rewards': [],
        'pos_errors': [],
        'vel_errors': [],
        'successes': [],
        'episode_lengths': [],
        'q_values': [],
        'exploration_noise': []
    }
    
    consecutive_successes = 0
    best_success_rate = 0
    best_mean_error = float('inf')
    episodes_without_improvement = 0
    
    # Adaptive exploration schedule for this phase
    exploration_start = phase_config.get('exploration_start', 0.2)
    exploration_end = phase_config.get('exploration_end', 0.05)
    total_episodes = phase_config['episodes']
    
    for episode in range(total_episodes):
        # Adaptive exploration noise (linear decay within phase)
        progress = episode / max(total_episodes - 1, 1)
        current_noise = exploration_start + (exploration_end - exploration_start) * progress
        agent.set_noise_scale(current_noise)
        
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_pos_errors = []
        episode_vel_errors = []
        episode_length = 0
        episode_q_values = []
        done = False
        
        while not done:
            # Select action
            if agent.total_steps < agent.warmup_steps:
                # Random exploration during warmup
                action = np.random.uniform(
                    -env.cfg.MAX_BODY_ACCELERATION, 
                    env.cfg.MAX_BODY_ACCELERATION, 
                    env.cfg.ACTION_DIM
                )
            else:
                action = agent.select_action(state, explore=True)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.replay_buffer.append((state, action, reward, next_state, done))
            
            # Update networks
            if agent.total_steps >= agent.warmup_steps:
                losses = agent.update(batch_size=agent.batch_size)
                if losses.get('q_value') is not None:
                    episode_q_values.append(losses['q_value'])
            
            episode_reward += reward
            episode_pos_errors.append(info['pos_error'])
            episode_vel_errors.append(info['vel_error'])
            episode_length += 1
            agent.total_steps += 1
            
            state = next_state
        
        agent.episode_count += 1
        
        # Episode statistics
        mean_pos_error = np.mean(episode_pos_errors)
        mean_vel_error = np.mean(episode_vel_errors)
        mean_q_value = np.mean(episode_q_values) if episode_q_values else 0.0
        
        # Success check
        success = (mean_pos_error < phase_config['success_error'] * 1000 and  
                  info.get('termination') == 'max_steps')
        
        if success:
            consecutive_successes += 1
        else:
            consecutive_successes = 0
        
        # Record stats
        stats['rewards'].append(episode_reward)
        stats['pos_errors'].append(mean_pos_error)
        stats['vel_errors'].append(mean_vel_error)
        stats['successes'].append(1 if success else 0)
        stats['episode_lengths'].append(episode_length)
        stats['q_values'].append(mean_q_value)
        stats['exploration_noise'].append(current_noise)
        
        # Compute running metrics
        window = min(100, len(stats['rewards']))
        recent_rewards = np.mean(stats['rewards'][-window:])
        recent_error = np.mean(stats['pos_errors'][-window:])
        recent_success = np.mean(stats['successes'][-window:])
        
        # Logging
        if (episode + 1) % TrainingConfig.LOG_FREQUENCY == 0 or episode < 5:
            print(f"Phase {phase_num}, Episode {episode+1}/{total_episodes}")
            print(f"  Reward: {episode_reward:.1f} (Avg: {recent_rewards:.1f})")
            print(f"  Pos Error: {mean_pos_error:.1f}mm (Avg: {recent_error:.1f}mm, Target: <{phase_config['success_error']*1000:.0f}mm)")
            print(f"  Vel Error: {mean_vel_error:.3f} m/s")
            print(f"  Length: {episode_length}/{env.max_steps}")
            print(f"  Success: {'✓' if success else '✗'} (Rate: {recent_success:.1%})")
            print(f"  Consecutive: {consecutive_successes}/{phase_config['required_successes']}")
            print(f"  Q-value: {mean_q_value:.2f}")
            print(f"  Exploration: {current_noise:.4f}")
            if 'thrust' in info:
                print(f"  Thrust: {info['thrust']:.2f}N")
            print()
        
        # Track best model
        if recent_error < best_mean_error:
            best_mean_error = recent_error
            episodes_without_improvement = 0
            if TrainingConfig.KEEP_BEST_ONLY:
                best_path = f'{save_dir}/position_phase_{phase_num}_best.pth'
                agent.save(best_path)
        else:
            episodes_without_improvement += 1
        
        if recent_success > best_success_rate:
            best_success_rate = recent_success
        
        # Periodic checkpoint
        if (episode + 1) % TrainingConfig.SAVE_FREQUENCY == 0:
            checkpoint_path = f'{save_dir}/position_phase_{phase_num}_ep{episode+1}.pth'
            agent.save(checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}\n")
        
        # EARLY STOPPING (if converged and past minimum)
        if (consecutive_successes >= phase_config['required_successes'] and 
            episode >= TrainingConfig.MIN_EPISODES_BEFORE_STOP):
            
            print(f"\n{'='*70}")
            print(f"✓✓✓ Phase {phase_num} CONVERGED at episode {episode+1}!")
            print(f"Achieved {consecutive_successes} consecutive successes")
            print(f"Final position error: {mean_pos_error:.1f}mm")
            print(f"Success rate: {recent_success:.1%}")
            print(f"{'='*70}\n")
            
            # Pad remaining episodes with final performance
            remaining = total_episodes - (episode + 1)
            for _ in range(remaining):
                stats['rewards'].append(stats['rewards'][-1])
                stats['pos_errors'].append(stats['pos_errors'][-1])
                stats['vel_errors'].append(stats['vel_errors'][-1])
                stats['successes'].append(1)
                stats['episode_lengths'].append(stats['episode_lengths'][-1])
                stats['q_values'].append(stats['q_values'][-1])
                stats['exploration_noise'].append(exploration_end)
            break
        
        # # EARLY STOPPING (if no improvement)
        # if (episodes_without_improvement >= TrainingConfig.EARLY_STOPPING_PATIENCE and
        #     episode >= TrainingConfig.MIN_EPISODES_BEFORE_STOP):
            
        #     print(f"\n{'='*70}")
        #     print(f"⚠ Phase {phase_num} EARLY STOP at episode {episode+1}")
        #     print(f"No improvement for {episodes_without_improvement} episodes")
        #     print(f"Best error: {best_mean_error:.1f}mm")
        #     print(f"Best success rate: {best_success_rate:.1%}")
        #     print(f"{'='*70}\n")
            
        #     # Pad remaining
        #     remaining = total_episodes - (episode + 1)
        #     for _ in range(remaining):
        #         stats['rewards'].append(stats['rewards'][-1])
        #         stats['pos_errors'].append(stats['pos_errors'][-1])
        #         stats['vel_errors'].append(stats['vel_errors'][-1])
        #         stats['successes'].append(stats['successes'][-1])
        #         stats['episode_lengths'].append(stats['episode_lengths'][-1])
        #         stats['q_values'].append(stats['q_values'][-1])
        #         stats['exploration_noise'].append(exploration_end)
        #     break
    
    return stats