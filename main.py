"""
Main Training Script for Cascaded TD3 Quadcopter Control
50-Second Expanding Helix Mission with Wind Training

Trains:
1. Attitude controller (inner loop) - 3,000 episodes
2. Position controller (outer loop) - 11,500 episodes (4 phases)
3. Fine-tuning (both together) - 3,000 episodes
Total: ~17,500 episodes
"""

import numpy as np
import scipy.io
import torch
import random
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (SystemConfig, AttitudeControllerConfig, 
                            PositionControllerConfig, FineTuningConfig, TrainingConfig)
from agents.td3_agent import TD3Agent
from environments.attitude_env import AttitudeEnv
from environments.position_env import PositionEnv
from training.attitude_trainer import train_attitude_controller
from training.position_trainer import train_position_controller
from training.fine_tuner import fine_tune_cascaded_system
from utils.visualization import plot_training_curves, plot_trajectory_3d
from utils.evaluation import evaluate_system


def set_seeds(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def print_system_info():
    """Print system configuration and trajectory info"""
    cfg = SystemConfig()
    
    print("\n" + "="*80)
    print("CASCADED TD3 QUADCOPTER TRAINING SYSTEM")
    print("="*80)
    print("\nMISSION PROFILE:")
    print(f"  Trajectory: Expanding Helix")
    print(f"  Formula: R(t) = {cfg.A} * (1 - exp(-{cfg.B}*t))")
    print(f"           x(t) = R(t) * cos({cfg.OMEGA}*t)")
    print(f"           y(t) = R(t) * sin({cfg.OMEGA}*t)")
    print(f"           z(t) = {cfg.VZ} * t")
    print(f"\n  Duration: {cfg.MAX_STEPS * cfg.DT:.0f} seconds")
    print(f"  Control Rate: {1/cfg.DT:.0f} Hz (dt={cfg.DT}s)")
    print(f"  Max Steps: {cfg.MAX_STEPS:,}")
    print(f"\n  Final Radius: ~3.86m (at t=50s)")
    print(f"  Final Height: 50m")
    print(f"  Total Loops: ~1.6 complete circles")
    print(f"\nQUADCOPTER SPECS:")
    print(f"  Mass: {cfg.MASS} kg")
    print(f"  Inertia: Ixx=0.3, Iyy=0.4, Izz=0.5 kg·m² (updated)")
    print(f"  Max Torque: {cfg.MAX_TORQUE:.4f} Nm  (= (2*m*g/4)*L*2)")
    print(f"\nTRAINING FEATURES:")
    print(f"  Wind Training: {'ENABLED' if cfg.WIND_TRAINING else 'DISABLED'}")
    print(f"  Random Start: {'ENABLED' if cfg.RANDOM_START else 'DISABLED'}")
    print(f"  Curriculum Phases: 4 (hover → 25% → 50% → 100%)")
    print("="*80)



def save_agent_as_mat(agent, filepath, agent_key='agent'):
    """
    Save a TD3 agent's actor/critic network weights as a MATLAB .mat file.
    
    The .mat file contains a struct-like dict with:
        <agent_key>.actor_layers  – list of (W, b) pairs for the actor network
        <agent_key>.critic1_layers – list of (W, b) pairs for critic 1
        <agent_key>.critic2_layers – list of (W, b) pairs for critic 2
        <agent_key>.max_action    – scalar action scaling
        <agent_key>.state_dim     – int
        <agent_key>.action_dim    – int
    
    To load in MATLAB:
        S = load('z_TD3_Attitude.mat');
        att = S.att_agent;
        % att.actor_weights_1, att.actor_biases_1, ... etc.
    """
    import torch
    import scipy.io as sio
    import numpy as np

    def extract_layers(model):
        """Extract (weight, bias) numpy arrays from all Linear layers."""
        layers = {}
        lin_idx = 1
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                layers[f'W{lin_idx}'] = module.weight.detach().cpu().numpy()
                layers[f'b{lin_idx}'] = module.bias.detach().cpu().numpy()
                lin_idx += 1
        return layers

    actor_layers   = extract_layers(agent.actor)
    critic1_layers = extract_layers(agent.critic.q1)
    critic2_layers = extract_layers(agent.critic.q2)

    mat_dict = {}
    for k, v in actor_layers.items():
        mat_dict[f'actor_{k}'] = v
    for k, v in critic1_layers.items():
        mat_dict[f'critic1_{k}'] = v
    for k, v in critic2_layers.items():
        mat_dict[f'critic2_{k}'] = v

    mat_dict['max_action'] = np.array([agent.max_action])
    mat_dict['state_dim']  = np.array([agent.actor.input_layer[0].in_features])
    mat_dict['action_dim'] = np.array([agent.actor.output_layers[-1].out_features])

    # Wrap in agent_key namespace so MATLAB access is: S.<agent_key>.*
    sio.savemat(filepath, {agent_key: mat_dict})
    print(f"  → .mat contains {len(mat_dict)} arrays under key '{agent_key}'")


def main():
    # Print system info
    print_system_info()
    
    # Set seeds
    set_seeds(SystemConfig.RANDOM_SEED)
    
    # Create directories
    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f'results/run_{timestamp}'
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f'{run_dir}/plots', exist_ok=True)
    
    print(f"\nResults will be saved to: {run_dir}")
    
    # =================================================================
    # PHASE 1: TRAIN ATTITUDE CONTROLLER (INNER LOOP)
    # =================================================================
    if TrainingConfig.TRAIN_ATTITUDE_FIRST:
        print("\n" + "="*80)
        print("PHASE 1: TRAINING ATTITUDE CONTROLLER (INNER LOOP)")
        print("="*80)
        
        # Create environment
        att_env = AttitudeEnv()
        
        # Create agent
        att_config = AttitudeControllerConfig()
        attitude_agent = TD3Agent(
            state_dim=att_config.STATE_DIM,
            action_dim=att_config.ACTION_DIM,
            max_action=att_config.MAX_TORQUE,
            hidden_dims=att_config.HIDDEN_DIMS,
            actor_lr=att_config.ACTOR_LR,
            critic_lr=att_config.CRITIC_LR,
            gamma=att_config.GAMMA,
            tau=att_config.TAU,
            policy_noise=att_config.POLICY_NOISE,
            noise_clip=att_config.NOISE_CLIP,
            policy_delay=att_config.POLICY_DELAY,
            buffer_size=att_config.BUFFER_SIZE
        )
        
        # Set warmup steps (CRITICAL - prevents NaN gradients)
        attitude_agent.warmup_steps = att_config.WARMUP_STEPS

        #attitude_agent.load(r'results\run_20260322_013752\attitude_checkpoint_ep7500.pth')
        
        # Train
        attitude_stats = train_attitude_controller(
            env=att_env,
            agent=attitude_agent,
            config=att_config,
            save_dir=run_dir
        )
        
        # Save final agent
        attitude_agent.save(f'{run_dir}/attitude_controller_final.pth')
        # Export attitude agent weights as .mat for MATLAB/Simulink
        save_agent_as_mat(attitude_agent, f'{run_dir}/z_TD3_Attitude.mat', 'att_agent')
        print(f"\n✓ Attitude agent saved as MATLAB .mat: {run_dir}/z_TD3_Attitude.mat")
        print(f"\n✓ Attitude controller saved to {run_dir}/attitude_controller_final.pth")
        
        # Plot training curves
        plot_training_curves(attitude_stats, f'{run_dir}/plots/attitude_training.png', 
                            'Attitude Controller Training')
    else:
        # Load pre-trained attitude controller
        print("\nLoading pre-trained attitude controller...")
        att_config = AttitudeControllerConfig()
        attitude_agent = TD3Agent(
            state_dim=att_config.STATE_DIM,
            action_dim=att_config.ACTION_DIM,
            max_action=att_config.MAX_TORQUE,
            hidden_dims=att_config.HIDDEN_DIMS
        )
        #attitude_agent.load(r'results\run_20260214_162109\attitude_best.pth')
        attitude_agent.load(r'results\run_20260323_191912\attitude_checkpoint_ep7500.pth')
    
    # =================================================================
    # PHASE 2: TRAIN POSITION CONTROLLER (OUTER LOOP)
    # =================================================================
    if TrainingConfig.TRAIN_POSITION_SECOND:
        print("\n" + "="*80)
        print("PHASE 2: TRAINING POSITION CONTROLLER (OUTER LOOP)")
        print("="*80)
        
        # Create environment with attitude controller
        pos_env = PositionEnv(attitude_agent=attitude_agent)
        
        # Create agent
        pos_config = PositionControllerConfig()
        position_agent = TD3Agent(
            state_dim=pos_config.STATE_DIM,
            action_dim=pos_config.ACTION_DIM,
            max_action=pos_config.MAX_BODY_ACCELERATION,
            hidden_dims=pos_config.HIDDEN_DIMS,
            actor_lr=pos_config.ACTOR_LR,
            critic_lr=pos_config.CRITIC_LR,
            gamma=pos_config.GAMMA,
            tau=pos_config.TAU,
            policy_noise=pos_config.POLICY_NOISE,
            noise_clip=pos_config.NOISE_CLIP,
            policy_delay=pos_config.POLICY_DELAY,
            buffer_size=pos_config.BUFFER_SIZE
        )

        position_agent.load(r'results\run_20260324_112629\position_phase_2_ep2000.pth')
        
        # Set warmup steps (CRITICAL - prevents NaN gradients)
        position_agent.warmup_steps = pos_config.WARMUP_STEPS
        
        # Train with curriculum
        position_stats = train_position_controller(
            env=pos_env,
            agent=position_agent,
            config=pos_config,
            save_dir=run_dir
        )
        
        # Save final agent
        position_agent.save(f'{run_dir}/position_controller_final.pth')
        # Export position agent weights as .mat for MATLAB/Simulink
        save_agent_as_mat(position_agent, f'{run_dir}/z_TD3_Position.mat', 'pos_agent')
        print(f"\n✓ Position agent saved as MATLAB .mat: {run_dir}/z_TD3_Position.mat")
        print(f"\n✓ Position controller saved to {run_dir}/position_controller_final.pth")
        
        # Plot training curves
        plot_training_curves(position_stats, f'{run_dir}/plots/position_training.png',
                            'Position Controller Training (4-Phase Curriculum)')
    else:
        # Load pre-trained position controller
        print("\nLoading pre-trained position controller...")
        pos_config = PositionControllerConfig()
        position_agent = TD3Agent(
            state_dim=pos_config.STATE_DIM,
            action_dim=pos_config.ACTION_DIM,
            max_action=pos_config.MAX_BODY_ACCELERATION,
            hidden_dims=pos_config.HIDDEN_DIMS
        )
        position_agent.load(r'results\run_20260326_112015\position_phase_3_ep3000.pth')
    
    # =================================================================
    # PHASE 3: FINE-TUNE CASCADED SYSTEM
    # =================================================================
    if TrainingConfig.FINE_TUNE_TOGETHER:
        print("\n" + "="*80)
        print("PHASE 3: FINE-TUNING CASCADED SYSTEM")
        print("="*80)
        
        finetune_stats = fine_tune_cascaded_system(
            position_agent=position_agent,
            attitude_agent=attitude_agent,
            config=FineTuningConfig(),
            save_dir=run_dir
        )
        
        # Save fine-tuned agents
        position_agent.save(f'{run_dir}/position_controller_finetuned.pth')
        attitude_agent.save(f'{run_dir}/attitude_controller_finetuned.pth')
        print(f"\n✓ Fine-tuned controllers saved")
        
        # Plot fine-tuning curves
        plot_training_curves(finetune_stats, f'{run_dir}/plots/finetuning.png',
                            'Cascaded System Fine-Tuning')
    
    # =================================================================
    # FINAL EVALUATION
    # =================================================================
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    # Create evaluation environment
    eval_env = PositionEnv(attitude_agent=attitude_agent)
    eval_env.set_trajectory_scale(1.0)  # Full trajectory
    eval_env.set_wind([0.05, 0.05, 0.3])  # Moderate wind for testing
    
    # Run evaluation
    eval_results = evaluate_system(
        env=eval_env,
        position_agent=position_agent,
        num_episodes=10,
        save_dir=run_dir
    )
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - FINAL METRICS")
    print("="*80)
    print(f"\nPosition Tracking Performance:")
    print(f"  Mean Position Error: {eval_results['mean_pos_error']:.1f} mm")
    print(f"  RMSE: {eval_results['rmse']:.1f} mm")
    print(f"  Max Error: {eval_results['max_error']:.1f} mm")
    print(f"  Success Rate: {eval_results['success_rate']:.1%}")
    print(f"  Mean Convergence Time: {eval_results['mean_conv_time']:.2f} s")
    
    print(f"\nAttitude Control Performance:")
    print(f"  Max Roll: {eval_results['max_roll']:.1f}°")
    print(f"  Max Pitch: {eval_results['max_pitch']:.1f}°")
    print(f"  Mean Roll: {eval_results['mean_roll']:.1f}°")
    print(f"  Mean Pitch: {eval_results['mean_pitch']:.1f}°")
    
    # Performance assessment (adjusted for 50m trajectory with 1 m/s climb)
    mean_error_m = eval_results['mean_pos_error'] / 1000
    if mean_error_m < 0.5:  # < 50cm
        print("\n✓✓✓ EXCELLENT PERFORMANCE!")
    elif mean_error_m < 0.8:  # < 80cm
        print("\n✓✓ GOOD PERFORMANCE!")
    elif mean_error_m < 1.2:  # < 120cm
        print("\n✓ ACCEPTABLE PERFORMANCE")
    else:
        print("\n⚠ NEEDS IMPROVEMENT")
    
    print(f"\nAll results saved to: {run_dir}")
    print("="*80)
    
    # Plot final trajectory
    plot_trajectory_3d(eval_env, f'{run_dir}/plots/final_trajectory.png')
    
    # Save evaluation results
    np.save(f'{run_dir}/evaluation_results.npy', eval_results)
    
    print("\n✓ Training pipeline complete!")
    print(f"Total training time: {timestamp}")


if __name__ == "__main__":
    main()