"""
Cascaded TD3 Config  –  FIXED VERSION
======================================
Fixes applied:
  - AttitudeControllerConfig.MAX_TORQUE = SystemConfig.MAX_TORQUE (was wrong value 5.0)
  - PositionControllerConfig: max_action = 1.0 (normalised), reward rewritten
  - Curriculum: starts with hover phase before helix
"""
 # To Prevent Original Research, this is not fully provided.

import numpy as np


class SystemConfig:
    MASS    = 1.0
    GRAVITY = 9.81
    DT      = 0.01

    IXX = 0.3; IYY = 0.4; IZZ = 0.5
    L   = 0.2

    MAX_THRUST     =  * MASS * GRAVITY
    MIN_THRUST     =  * MASS * GRAVITY
    MAX_TORQUE     = (MAX_THRUST / 4.0) * L * 2    
    MAX_TORQUE_YAW =  * MAX_TORQUE              

    BASE_MISSION_TIME = 50.0
    MAX_STEPS         = int(BASE_MISSION_TIME / DT)

    MAX_POSITION     = 
    MIN_ALTITUDE     = 
    MAX_ALTITUDE     = 
    CRASH_ANGLE      = np.deg2rad()
    MAX_VELOCITY     = 
    MAX_ANGULAR_RATE = np.deg2rad()

    WIND_TRAINING  = False
    RANDOM_START   = True
    RANDOM_SEED    = 
    WIND_GUST_PROB = 0.0
    WIND_GUST_MAG  = 0.0
    ATTITUDE_MAX_STEPS = 

    A = 9.81; B = 0.01; OMEGA = 0.2; VZ = 1.0


class AttitudeControllerConfig:
    STATE_DIM  = 9
    ACTION_DIM = 3

    ACTOR_LR  = 
    CRITIC_LR = 
    GAMMA = ; TAU = 
    POLICY_NOISE = 0.1; NOISE_CLIP = 0.2; POLICY_DELAY = 2
    HIDDEN_DIMS = 128
    BATCH_SIZE = 256; BUFFER_SIZE = 200_000; WARMUP_STEPS = 500
    EXPLORATION_NOISE = ; NOISE_DECAY = ; MIN_NOISE = 

    # FIX: use correct physical torque limit
    MAX_TORQUE = SystemConfig.MAX_TORQUE   # was 5.0, now 1.962 Nm

    EPISODES_PER_PHASE = 
    SUCCESS_THRESHOLD  = np.deg2rad()
    CONSECUTIVE_SUCCESSES = 
    MAX_EXPECTED_ERROR = np.deg2rad()
    MAX_EXPECTED_RATE  = 


class PositionControllerConfig:
    STATE_DIM  = 18
    ACTION_DIM = 3

    ACTOR_LR  = 
    CRITIC_LR = 
    GAMMA = ; TAU = 
    POLICY_NOISE = ; NOISE_CLIP = ; POLICY_DELAY = 
    HIDDEN_DIMS = 
    BATCH_SIZE = ; BUFFER_SIZE = ; WARMUP_STEPS = 
    EXPLORATION_NOISE =  NOISE_DECAY = ; MIN_NOISE = 

    # FIX: normalised action (body accelerations still, but actor outputs [-1,+1])
    MAX_BODY_ACCELERATION =   # physical limit
    ACTION_SCALE          =    # actor max_action

    MIN_THRUST     = SystemConfig.MIN_THRUST
    MAX_THRUST     = SystemConfig.MAX_THRUST
    MAX_TILT_ANGLE = np.deg2rad(45)

    ATTITUDE_PENALTY   = 
    ACTION_PENALTY     = 
    SMOOTHNESS_PENALTY = 
    PRECISION_BONUS    = 0.0   # FIX: removed – caused lazy-agent exploit

    REWARD_CONFIG = {
           # To Prevent Original Research, this is not fully provided.
    }

    CURRICULUM_PHASES = [
         # To Prevent Original Research, this is not fully provided.
    ]


class FineTuningConfig:
    EPISODES = 500
    FINE_TUNE_EPISODES = 500
    ATTITUDE_LR_SCALE = 0.1
    POSITION_LR_SCALE = 0.3
    NOISE_SCALE = 0.3
    SUCCESS_ERROR = 0.6
    SUCCESS_RATE  = 0.80


class TrainingConfig:
    TRAIN_ATTITUDE_FIRST  = False
    TRAIN_POSITION_SECOND = False
    FINE_TUNE_TOGETHER    = True
    SAVE_FREQUENCY  = 500
    KEEP_BEST_ONLY  = True
    LOG_FREQUENCY   = 5
    EVAL_EPISODES   = 5
    EVAL_FREQUENCY  = 100
    EARLY_STOPPING_PATIENCE  = 50
    MIN_EPISODES_BEFORE_STOP = 100
    NUM_WORKERS = 1


def get_trajectory_function():
    cfg = SystemConfig()

    def trajectory(t, scale=1.0):
        exp_term = np.exp(-cfg.B * t)
        radius   = cfg.A * (1 - exp_term) * scale
        omega_t  = cfg.OMEGA * t
        x = radius * np.cos(omega_t)
        y = radius * np.sin(omega_t)
        z = cfg.VZ * t * scale
        dR = cfg.A * cfg.B * exp_term * scale
        vx = dR*np.cos(omega_t) - radius*cfg.OMEGA*np.sin(omega_t)
        vy = dR*np.sin(omega_t) + radius*cfg.OMEGA*np.cos(omega_t)
        vz = cfg.VZ * scale
        d2R = -cfg.A * cfg.B**2 * exp_term * scale
        ax = d2R*np.cos(omega_t) - 2*dR*cfg.OMEGA*np.sin(omega_t) - radius*cfg.OMEGA**2*np.cos(omega_t)
        ay = d2R*np.sin(omega_t) + 2*dR*cfg.OMEGA*np.cos(omega_t) - radius*cfg.OMEGA**2*np.sin(omega_t)
        az = 0.0
        return np.array([x,y,z]), np.array([vx,vy,vz]), np.array([ax,ay,az])

    return trajectory
