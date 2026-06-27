"""
Position Controller Environment  –  FIXED VERSION
===================================================
Fixes:
  FIX 1: Action space is now normalised [-1,+1]. step() maps a[0..2] to
          body accelerations in [-MAX_BODY_ACCELERATION, +MAX_BODY_ACCELERATION].
          This ensures the actor can explore the full range without saturation.

  FIX 2: Reward rewritten to use dense exponential tracking + survival bonus.
          Precision bonus removed (caused lazy-agent exploit).

  FIX 3: Attitude obs fed to inner agent uses psi_des=0 consistently,
          not the current psi (which is meaningless as a setpoint).

  FIX 4: Warmup random actions are in [-1,+1] (normalised), not raw acc range.
"""

import numpy as np
import gym
from gym import spaces
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.dynamics import QuadcopterDynamics
from configs.config import SystemConfig, PositionControllerConfig, get_trajectory_function


def system_solve(u1, u2, u3, psi, m, g):
    """Python port of MATLAB system_solve.m"""
    if abs(psi) <= 0.001: psi = 0.001
    a = np.sin(psi); b = np.cos(psi)
    A = u1/u3; B = u2/u3
    C = (a*A - b*B) / (a**2 + b**2)
    roll  = np.arctan((B + b*C) / a)
    pitch = np.arctan(C * np.cos(roll))
    pitch = np.clip(pitch, -np.pi/4, np.pi/4)
    roll  = np.clip(roll,  -np.pi/4, np.pi/4)
    ft = (m / (np.cos(pitch)*np.cos(roll))) * (-u3)
    return ft, pitch, roll


class PositionEnv(gym.Env):

    def __init__(self, attitude_agent=None, config=None):
        super().__init__()
        self.sys_cfg = SystemConfig()
        self.cfg     = config if config is not None else PositionControllerConfig()

        self.dynamics = QuadcopterDynamics(
            mass=self.sys_cfg.MASS, gravity=self.sys_cfg.GRAVITY, dt=self.sys_cfg.DT)

        self.attitude_agent  = attitude_agent
        self.get_trajectory  = get_trajectory_function()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.STATE_DIM,), dtype=np.float32)

        # FIX 1: normalised action space [-1, +1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.ACTION_DIM,), dtype=np.float32)

        self.state      = None
        self.time       = 0.0
        self.steps      = 0
        self.max_steps  = self.sys_cfg.MAX_STEPS

        self.trajectory_scale  = 1.0
        self.wind_std          = [0.0, 0.0, 0.0]
        self.start_radius      = 0.0
        self.current_reward_config = self.cfg.REWARD_CONFIG['helix_full']
        self.prev_action       = None

        self.trajectory_history         = []
        self.desired_trajectory_history = []
        self.attitude_history           = []

    def set_trajectory_scale(self, s): self.trajectory_scale = s
    def set_start_radius(self, r):     self.start_radius = r

    def set_reward_config(self, key):
        if key in self.cfg.REWARD_CONFIG:
            self.current_reward_config = self.cfg.REWARD_CONFIG[key]

    def set_wind(self, std):
        self.wind_std = list(std) if hasattr(std,'__iter__') else [std,std,std*0.6]
        self.dynamics.set_wind(
            enabled=any(s>0 for s in self.wind_std), std=self.wind_std, mean=[0,0,0],
            gust_prob=self.sys_cfg.WIND_GUST_PROB, gust_mag=self.sys_cfg.WIND_GUST_MAG)

    def reset(self, initial_pos=None):
        self.time  = 0.0
        self.steps = 0

        if initial_pos is None:
            if self.sys_cfg.RANDOM_START and self.start_radius > 0:
                angle       = np.random.uniform(0, 2*np.pi)
                # beta = distribution parameter to control radial distribution (beta=1 uniform, beta>1 more towards edge, beta<1 more towards center)
                radius = np.random.beta(a=2.0, b=1.0) * self.start_radius
                #radius      = np.random.uniform(0.3, self.start_radius)
                z           = np.random.uniform(0, min(1.5, self.start_radius/2))
                initial_pos = np.array([radius*np.cos(angle), radius*np.sin(angle), z])
            else:
                initial_pos = np.random.uniform(-0.05, 0.05, 3)

        self.state       = np.zeros(12)
        self.state[0:3]  = initial_pos
        self.state[2]    = max(0.0, initial_pos[2])
        self.state[3:6]  = np.random.uniform(-0.05, 0.05, 3)
        self.state[6]    = np.random.uniform(-0.03, 0.03)
        self.state[7]    = np.random.uniform(-0.03, 0.03)
        self.state[8]    = np.random.uniform(-0.05, 0.05)

        self.set_wind(self.wind_std)
        self.prev_action = None
        self.trajectory_history = []; self.desired_trajectory_history = []
        self.attitude_history   = []
        return self._get_observation()

    def step(self, action):
        # FIX 1: map normalised [-1,+1] → physical body acceleration
        action = np.clip(np.array(action, dtype=np.float64), -1.0, 1.0)
        acc_body = action * self.cfg.MAX_BODY_ACCELERATION

        att   = self.state[6:9]
        rates = self.state[9:12]
        psi   = att[2]

        ux, uy, uz = acc_body
        u3 = uz - self.sys_cfg.GRAVITY
        if abs(u3) < 1e-4: u3 = -1e-4

        ft, phi_des, theta_des = system_solve(ux, uy, u3, psi,
                                              self.sys_cfg.MASS, self.sys_cfg.GRAVITY)
        ft = np.clip(ft, self.sys_cfg.MIN_THRUST, self.sys_cfg.MAX_THRUST)

        # FIX 3: always track yaw=0, not current psi
        att_des = np.array([phi_des, theta_des, 0.0])

        # clip the desired attitude to prevent extreme setpoints that cause instability and large negative rewards
        att_des[0] = np.clip(att_des[0], -np.deg2rad(45), np.deg2rad(45))
        att_des[1] = np.clip(att_des[1], -np.deg2rad(45), np.deg2rad(45))
        att_des[2] = np.clip(att_des[2], -np.deg2rad(45), np.deg2rad(45))   
        

        if self.attitude_agent is not None:

            # print("Raw att_des (deg):{:.2f}".format(np.rad2deg(att_des))
            #       , "ft:{:.2f}".format(ft), "acc_body:{:.2f}".format(acc_body[2])
            #       , "att:{:.2f}".format(np.rad2deg(att)), "rates:{:.2f}".format(np.rad2deg(rates)))
            #print("Raw att_des (deg):", np.rad2deg(att_des), "ft:", ft, "acc_body:", acc_body[2],    "att (deg):", np.rad2deg(att), "rates (rad/s):", rates)
            
            att_obs = np.concatenate([att, rates, att_des]).astype(np.float32)
            torques = self.attitude_agent.select_action(att_obs, explore=False)
            torques = np.clip(torques,
                              -self.sys_cfg.MAX_TORQUE, self.sys_cfg.MAX_TORQUE)
        else:
            att_error = self.dynamics.wrap_angles(att_des - att)
            kp, kd    = 8.0, 2.0
            torques   = kp * att_error - kd * rates
            torques   = np.clip(torques, -self.sys_cfg.MAX_TORQUE,
                                          self.sys_cfg.MAX_TORQUE)

        self.state = self.dynamics.rk4_step(self.state, ft, torques)
        self.time  += self.sys_cfg.DT
        self.steps += 1

        self.trajectory_history.append(self.state[0:3].copy())
        pos_des, _, _ = self.get_trajectory(self.time, self.trajectory_scale)
        self.desired_trajectory_history.append(pos_des.copy())
        self.attitude_history.append(np.rad2deg(self.state[6:9].copy()))

        reward, info = self._compute_reward(action, acc_body, ft, att_des)
        done, term   = self._check_done()

        if done and term in ('crash','out_of_bounds','altitude_limit',
                             'excessive_velocity','excessive_rate'):
            reward -= 50.0

        info['termination'] = term
        info['thrust']      = ft
        info['att_des']     = np.rad2deg(att_des)
        info['torques']     = torques

        self.prev_action = action.copy()
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        pos   = self.state[0:3]
        vel   = self.state[3:6]
        att   = self.state[6:9]
        rates = self.state[9:12]
        pos_des, vel_des, acc_des = self.get_trajectory(self.time, self.trajectory_scale)
        rel_pos  = (pos - pos_des) / self.sys_cfg.MAX_POSITION
        norm_vel = vel / self.sys_cfg.MAX_VELOCITY
        obs = np.concatenate([rel_pos, norm_vel, att, rates/10.0,
                              vel_des/self.sys_cfg.MAX_VELOCITY,
                              acc_des/self.cfg.MAX_BODY_ACCELERATION])
        return obs.astype(np.float32)

    def _compute_reward(self, norm_action, acc_body, ft, att_des):
        # FIX 2: dense exponential reward, no precision bonus
        pos     = self.state[0:3]
        vel     = self.state[3:6]
        att     = self.state[6:9]
        pos_des, vel_des, _ = self.get_trajectory(self.time, self.trajectory_scale)

        pos_err = np.linalg.norm(pos - pos_des)
        vel_err = np.linalg.norm(vel - vel_des)

        pos_r    = 10.0 * np.exp(-2.0 * pos_err)
        vel_p    = -0.5 * (vel_err / self.sys_cfg.MAX_VELOCITY)**2
        att_norm = np.deg2rad(15.0)
        att_p    = -1.0 * np.sum(np.square(att[0:2] / att_norm))
        thrust_err  = abs(ft - self.sys_cfg.MASS*self.sys_cfg.GRAVITY) / \
                      (self.sys_cfg.MASS*self.sys_cfg.GRAVITY)
        survival_r  = 1.0 * np.exp(-3.0 * thrust_err)

        smooth_p = 0.0
        if self.prev_action is not None:
            smooth_p = -0.05 * np.sum(np.square(norm_action - self.prev_action))

        total = 0.05*(pos_r + vel_p + att_p + survival_r + smooth_p)

        info = {
            'pos_error':   pos_err*1000,
            'vel_error':   vel_err,
            'pos_error_m': pos_err,
            'reward_components': {
                'position':   pos_r, 'velocity': vel_p,
                'attitude':   att_p, 'survival': survival_r,
                'smoothness': smooth_p, 'total': total
            }
        }
        return total, info

    def _check_done(self):
        pos   = self.state[0:3]
        att   = self.state[6:9]
        vel   = self.state[3:6]
        rates = self.state[9:12]
        if np.abs(att[0]) > self.sys_cfg.CRASH_ANGLE or \
           np.abs(att[1]) > self.sys_cfg.CRASH_ANGLE:
            return True, 'crash'
        if np.abs(pos[0]) > self.sys_cfg.MAX_POSITION or \
           np.abs(pos[1]) > self.sys_cfg.MAX_POSITION:
            return True, 'out_of_bounds'
        if pos[2] < self.sys_cfg.MIN_ALTITUDE or pos[2] > self.sys_cfg.MAX_ALTITUDE:
            return True, 'altitude_limit'
        if np.linalg.norm(vel) > self.sys_cfg.MAX_VELOCITY:
            return True, 'excessive_velocity'
        if np.max(np.abs(rates)) > self.sys_cfg.MAX_ANGULAR_RATE:
            return True, 'excessive_rate'
        if self.steps >= self.max_steps:
            return True, 'max_steps'
        return False, None

    def render(self, mode='human'): pass
