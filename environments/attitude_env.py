"""
Attitude Controller Environment  –  FIXED VERSION
===================================================
Fixes:
  FIX 1: MAX_TORQUE now uses correct physical value (1.962 Nm, not 5.0).
  FIX 2: Reward rewritten – exponential attitude reward + damping + stability.
         The reward clearly distinguishes good attitude from random.
  FIX 3: Observation unchanged (9-dim: [phi,theta,psi, p,q,r, phi_des,theta_des,psi_des]).
"""

import numpy as np
import gym
from gym import spaces
from environments.dynamics import QuadcopterDynamics
from configs.config import SystemConfig, AttitudeControllerConfig


class AttitudeEnv(gym.Env):

    def __init__(self, config=None):
        super().__init__()
        self.sys_cfg = SystemConfig()
        self.cfg     = config if config is not None else AttitudeControllerConfig()

        self.dynamics = QuadcopterDynamics(
            mass=self.sys_cfg.MASS, gravity=self.sys_cfg.GRAVITY, dt=self.sys_cfg.DT)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.STATE_DIM,), dtype=np.float32)

        # FIX 1: correct physical torque limit
        self.action_space = spaces.Box(
            low=-self.cfg.MAX_TORQUE, high=self.cfg.MAX_TORQUE,
            shape=(self.cfg.ACTION_DIM,), dtype=np.float32)

        self.state            = None
        self.desired_attitude = None
        self.steps            = 0
        self.max_steps        = self.sys_cfg.ATTITUDE_MAX_STEPS

    def reset(self, desired_attitude=None):
        if desired_attitude is None:
            self.desired_attitude = np.array([
                np.random.uniform(-np.pi/3, np.pi/3),
                np.random.uniform(-np.pi/3, np.pi/3),
                np.random.uniform(-np.pi, np.pi)
            ])
        else:
            self.desired_attitude = np.array(desired_attitude)

        # phi   = self.desired_attitude[0] + np.random.uniform(-0.05, 0.05)
        # theta = self.desired_attitude[1] + np.random.uniform(-0.05, 0.05)
        # psi   = self.desired_attitude[2] + np.random.uniform(-0.10, 0.10)

        max_offset = np.deg2rad(5 + 45 * np.random.rand())
        phi   = self.desired_attitude[0] + np.random.uniform(-max_offset, max_offset)
        theta = self.desired_attitude[1] + np.random.uniform(-max_offset, max_offset)
        psi   = self.desired_attitude[2] + np.random.uniform(-max_offset*1.5, max_offset*1.5)

        p = np.random.uniform(-0.3, 0.3)
        q = np.random.uniform(-0.3, 0.3)
        r = np.random.uniform(-0.3, 0.3)

        self.state        = np.zeros(12)
        self.state[6:9]   = [phi, theta, psi]
        self.state[9:12]  = [p, q, r]
        self.steps        = 0
        return self._get_observation()

    def step(self, action):
        action = np.clip(action, -self.cfg.MAX_TORQUE, self.cfg.MAX_TORQUE)
        thrust = self.sys_cfg.MASS * self.sys_cfg.GRAVITY  # hover thrust
        self.state = self.dynamics.rk4_step(self.state, thrust, action)
        self.steps += 1

        reward, info = self._compute_reward(action)
        done, term   = self._check_done()

        if done and term in ('crash', 'excessive_rate'):
            reward -= 50.0

        reward = np.clip(reward, -200.0, 15.0)
        info['termination'] = term
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        att   = self.state[6:9]
        rates = self.state[9:12]
        obs   = np.concatenate([att, rates, self.desired_attitude])
        return obs.astype(np.float32)

    def _compute_reward(self, action):
        att   = self.state[6:9]
        rates = self.state[9:12]

        att_error     = self.dynamics.wrap_angles(att - self.desired_attitude)
        att_error_mag = np.linalg.norm(att_error)

        # Exponential attitude reward (peak=+10)
        att_r = 10.0 * np.exp(-5.0 * att_error_mag)

        # Rate damping (stronger near target to prevent oscillation)
        rate_norm  = np.linalg.norm(rates)
        damp_mult  = 1.0 + 3.0 * np.exp(-5.0 * att_error_mag)
        rate_p     = -1.5 * rate_norm * damp_mult

        # Stability bonus
        is_level  = np.rad2deg(att_error_mag) < 5.0
        is_static = np.max(np.abs(rates)) < 0.05
        stab_b    = 3.0 if (is_level and is_static) else 0.0

        # Action effort penalty (normalised by physical limit)
        effort_p = -0.01 * np.sum((action / self.cfg.MAX_TORQUE)**2)

        reward = att_r + rate_p + stab_b + effort_p

        info = {
            'att_error':  np.rad2deg(att_error_mag),
            'rate_error': rate_norm,
            'reward_components': {
                'attitude':   att_r,
                'rate':       rate_p,
                'stability':  stab_b,
                'effort':     effort_p
            }
        }
        return reward, info

    def _check_done(self):
        att   = self.state[6:9]
        rates = self.state[9:12]
        if np.abs(att[0]) > self.sys_cfg.CRASH_ANGLE or \
           np.abs(att[1]) > self.sys_cfg.CRASH_ANGLE:
            return True, 'crash'
        if np.max(np.abs(rates)) > self.sys_cfg.MAX_ANGULAR_RATE:
            return True, 'excessive_rate'
        if self.steps >= self.max_steps:
            return True, 'max_steps'
        return False, None

    def render(self, mode='human'): pass
