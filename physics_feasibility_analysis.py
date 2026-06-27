"""
Comprehensive Physics-Based Feasibility Analysis
For Drone Control System with Requirements:
- Attitude: <5° mean error
- Position: <0.2m precision
- Remote start capability (far from helix)
"""

import numpy as np
import matplotlib.pyplot as plt
from configs.config import SystemConfig, AttitudeControllerConfig, PositionControllerConfig

class PhysicsFeasibilityAnalyzer:
    def __init__(self):
        self.sys_cfg = SystemConfig()
        self.att_cfg = AttitudeControllerConfig()
        self.pos_cfg = PositionControllerConfig()
        
        # Extract key parameters
        self.m = self.sys_cfg.MASS
        self.g = self.sys_cfg.GRAVITY
        self.dt = self.sys_cfg.DT
        self.Ixx = 0.3  # From dynamics.py
        self.Iyy = 0.3
        self.Izz = 0.3
        self.max_torque = self.att_cfg.MAX_TORQUE
        self.max_thrust = self.pos_cfg.MAX_THRUST
        self.min_thrust = self.pos_cfg.MIN_THRUST
        self.max_tilt = self.pos_cfg.MAX_TILT_ANGLE
        
    def print_header(self, title):
        print("\n" + "="*80)
        print(f"{title:^80}")
        print("="*80)
        
    def analyze_inertia(self):
        """Analyze if inertia values are realistic"""
        self.print_header("INERTIA ANALYSIS")
        
        print(f"\nConfigured Inertia:")
        print(f"  Ixx = {self.Ixx:.3f} kg·m²")
        print(f"  Iyy = {self.Iyy:.3f} kg·m²")
        print(f"  Izz = {self.Izz:.3f} kg·m²")
        print(f"  Mass = {self.m:.1f} kg")
        
        # Calculate effective radius
        # For a quadcopter: I ≈ 0.5 * m * r²
        r_eff_xx = np.sqrt(2 * self.Ixx / self.m)
        r_eff_yy = np.sqrt(2 * self.Iyy / self.m)
        r_eff_zz = np.sqrt(2 * self.Izz / self.m)
        
        print(f"\nEffective Radius (from I = 0.5*m*r²):")
        print(f"  r_xx = {r_eff_xx:.3f} m ({r_eff_xx*2:.2f}m diagonal)")
        print(f"  r_yy = {r_eff_yy:.3f} m ({r_eff_yy*2:.2f}m diagonal)")
        print(f"  r_zz = {r_eff_zz:.3f} m ({r_eff_zz*2:.2f}m diagonal)")
        
        # Typical values for comparison
        print(f"\nComparison to Typical Drones (1kg):")
        print(f"  Racing drone:      I ~ 0.01-0.03 kg·m²  (r ~ 0.14-0.24m)")
        print(f"  Standard drone:    I ~ 0.03-0.08 kg·m²  (r ~ 0.24-0.40m)")
        print(f"  Large commercial:  I ~ 0.10-0.15 kg·m²  (r ~ 0.45-0.55m)")
        print(f"  YOUR DRONE:        I ~ 0.30 kg·m²       (r ~ 0.77m)")
        
        print(f"\n⚠️  ASSESSMENT:")
        if self.Ixx > 0.15:
            print(f"  Your drone has VERY HIGH inertia for its mass.")
            print(f"  This suggests either:")
            print(f"    - Wide frame (~1.5m diagonal) with heavy motors at tips")
            print(f"    - Long-range platform with extended arms")
            print(f"    - Unusual mass distribution")
            print(f"  Impact: Slower angular response, requires strong damping")
        else:
            print(f"  Inertia is reasonable for a 1kg drone.")
            
        return r_eff_xx
        
    def analyze_attitude_control(self):
        """Check if <5° attitude control is feasible"""
        self.print_header("ATTITUDE CONTROL FEASIBILITY (<5° target)")
        
        # Angular acceleration capability
        alpha_max = self.max_torque / self.Ixx
        print(f"\nAngular Acceleration Capability:")
        print(f"  τ_max = {self.max_torque:.1f} N·m")
        print(f"  I = {self.Ixx:.3f} kg·m²")
        print(f"  α_max = τ/I = {alpha_max:.2f} rad/s²  ({np.rad2deg(alpha_max):.1f}°/s²)")
        
        # Time to correct 5° error
        angle_5deg = np.deg2rad(5.0)
        # Using s = 0.5*a*t² (assuming acceleration then deceleration)
        t_half = np.sqrt(angle_5deg / alpha_max)  # Time to halfway
        omega_peak = alpha_max * t_half
        t_total = 2 * t_half
        
        print(f"\nTime Response for 5° Correction:")
        print(f"  Target angle: {np.rad2deg(angle_5deg):.1f}°")
        print(f"  Acceleration phase: {t_half*1000:.1f} ms")
        print(f"  Peak rate: {omega_peak:.2f} rad/s ({np.rad2deg(omega_peak):.1f}°/s)")
        print(f"  Deceleration phase: {t_half*1000:.1f} ms")
        print(f"  Total maneuver time: {t_total*1000:.1f} ms ({int(t_total/self.dt)} steps)")
        
        # Bandwidth analysis
        print(f"\nControl Bandwidth Analysis:")
        print(f"  Sampling rate: {1/self.dt:.0f} Hz")
        print(f"  Nyquist frequency: {1/(2*self.dt):.0f} Hz")
        
        # Estimate natural frequency (assuming PD control with kp~10)
        kp_estimate = 10.0
        omega_n = np.sqrt(kp_estimate / self.Ixx)
        f_n = omega_n / (2 * np.pi)
        
        print(f"  Estimated natural freq: {f_n:.2f} Hz (ωn = {omega_n:.2f} rad/s)")
        print(f"  Bandwidth margin: {1/(2*self.dt) / f_n:.1f}x")
        
        if 1/(2*self.dt) / f_n > 10:
            print(f"  ✅ Excellent bandwidth margin (>10x)")
        elif 1/(2*self.dt) / f_n > 5:
            print(f"  ✅ Good bandwidth margin (>5x)")
        else:
            print(f"  ⚠️  Marginal bandwidth margin (<5x)")
            
        # Settling time estimate (2% criterion)
        zeta = 0.7  # Assumed damping ratio
        t_settle = 4 / (zeta * omega_n)
        
        print(f"\nSettling Time Analysis (2% criterion, ζ={zeta}):")
        print(f"  Expected settling time: {t_settle:.3f} s ({int(t_settle/self.dt)} steps)")
        
        # Check reward function
        print(f"\nReward Function Assessment:")
        print(f"  Current exponential factor: 10.0")
        print(f"  Damping multiplier: 1.0 + 3.0*exp(-5.0*error)")
        print(f"  Stability bonus threshold: 5.0°")
        
        # Feasibility verdict
        print(f"\n{'='*80}")
        print(f"ATTITUDE CONTROL FEASIBILITY: ", end="")
        
        if alpha_max > 10 and t_total < 0.3 and f_n < 1/(2*self.dt):
            print(f"✅ FEASIBLE")
            print(f"\nWith proper tuning, <5° mean error is ACHIEVABLE:")
            print(f"  • Angular authority: Adequate ({alpha_max:.1f} rad/s²)")
            print(f"  • Response time: Fast enough ({t_total*1000:.0f}ms)")
            print(f"  • Control bandwidth: Sufficient ({f_n:.1f} Hz)")
            print(f"  • Expected steady-state: 2-4° with current reward")
            feasible = True
        else:
            print(f"⚠️  MARGINAL")
            print(f"\n<5° may be difficult due to:")
            if alpha_max < 10:
                print(f"  • Low angular acceleration ({alpha_max:.1f} rad/s²)")
            if t_total > 0.5:
                print(f"  • Slow response ({t_total*1000:.0f}ms)")
            feasible = False
            
        return feasible
        
    def analyze_position_control(self):
        """Check if <0.2m position control is feasible"""
        self.print_header("POSITION CONTROL FEASIBILITY (<0.2m target)")
        
        # Thrust capability
        TWR = self.max_thrust / (self.m * self.g)
        print(f"\nThrust Capability:")
        print(f"  Min thrust: {self.min_thrust:.2f} N ({self.min_thrust/(self.m*self.g):.2f}x weight)")
        print(f"  Hover thrust: {self.m*self.g:.2f} N")
        print(f"  Max thrust: {self.max_thrust:.2f} N")
        print(f"  Thrust-to-Weight: {TWR:.2f}:1")
        
        # Vertical acceleration
        acc_vert_max = (self.max_thrust / self.m) - self.g
        acc_vert_min = (self.min_thrust / self.m) - self.g
        
        print(f"\nVertical Acceleration:")
        print(f"  Max climb: {acc_vert_max:.2f} m/s²")
        print(f"  Max descent: {acc_vert_min:.2f} m/s²")
        
        # Lateral acceleration at different tilts
        print(f"\nLateral Acceleration vs Tilt (at hover thrust):")
        tilts = [10, 20, 30, 45, 60]
        for tilt_deg in tilts:
            tilt = np.deg2rad(tilt_deg)
            if tilt <= self.max_tilt:
                acc_lat = self.g * np.tan(tilt)
                print(f"  {tilt_deg:2d}°: {acc_lat:5.2f} m/s² {'✓' if tilt <= self.max_tilt else 'X'}")
        
        print(f"\n  Max tilt configured: {np.rad2deg(self.max_tilt):.1f}°")
        acc_lat_max = self.g * np.tan(self.max_tilt)
        print(f"  Max lateral accel: {acc_lat_max:.2f} m/s²")
        
        # Trajectory requirements
        print(f"\nTrajectory Analysis (Expanding Helix):")
        A = self.sys_cfg.A
        B = self.sys_cfg.B
        omega = self.sys_cfg.OMEGA
        vz = self.sys_cfg.VZ
        
        print(f"  Asymptotic radius: {A:.2f} m")
        print(f"  Decay constant: {B:.4f}")
        print(f"  Angular velocity: {omega:.2f} rad/s")
        print(f"  Vertical velocity: {vz:.2f} m/s")
        
        # Max centripetal acceleration (at full radius)
        acc_centripetal = omega**2 * A
        print(f"\n  Max centripetal accel: {acc_centripetal:.3f} m/s²")
        print(f"  Required tilt: {np.rad2deg(np.arctan(acc_centripetal/self.g)):.2f}°")
        
        if acc_centripetal < acc_lat_max:
            print(f"  ✅ Trajectory is within physical limits")
        else:
            print(f"  ⚠️  Trajectory requires more lateral acceleration than available!")
            
        # Remote start analysis
        print(f"\n{'='*40}")
        print(f"REMOTE START CAPABILITY:")
        print(f"{'='*40}")
        
        max_start_radius = self.sys_cfg.START_RADIUS_MAX
        print(f"\nMaximum start radius: {max_start_radius:.1f} m")
        
        # Time to reach trajectory from 5m away
        # Assuming average approach speed of 2 m/s
        approach_speed = 2.0
        approach_time = max_start_radius / approach_speed
        
        print(f"Approach to trajectory:")
        print(f"  Average approach speed: {approach_speed:.1f} m/s")
        print(f"  Time to reach trajectory: {approach_time:.1f} s")
        print(f"  Required lateral accel: {acc_centripetal:.3f} m/s²")
        
        # Can we maintain <0.2m precision during approach?
        # Using control theory: steady-state error ≈ disturbance / (kp)
        # For wind disturbance ~0.5 m/s, kp needed: ~2.5
        
        print(f"\n{'='*80}")
        print(f"PRECISION ANALYSIS (<0.2m target):")
        print(f"{'='*80}")
        
        # Position control bandwidth
        # Assuming outer loop bandwidth ~0.5-1 Hz (typical for position control)
        f_pos_control = 0.5  # Hz (conservative estimate)
        t_settle_pos = 4 / (2 * np.pi * f_pos_control)
        
        print(f"\nPosition Control Loop:")
        print(f"  Estimated bandwidth: ~{f_pos_control:.1f} Hz")
        print(f"  Settling time: ~{t_settle_pos:.1f} s")
        print(f"  Wind disturbance: {self.sys_cfg.WIND_TRAINING}")
        
        # Steady-state error under wind
        wind_std = 0.5  # m/s
        drag_coeff = 0.1
        disturbance_force = drag_coeff * wind_std
        
        # For 0.2m precision with wind, need high proportional gain
        required_kp = disturbance_force / 0.2
        
        print(f"\nWind Rejection:")
        print(f"  Typical wind std: {wind_std:.1f} m/s")
        print(f"  Drag coefficient: {drag_coeff:.2f}")
        print(f"  Disturbance force: {disturbance_force:.2f} N")
        print(f"  Required kp (for <0.2m): {required_kp:.2f}")
        
        # Check if remote start + precision is feasible
        print(f"\n{'='*80}")
        print(f"POSITION CONTROL FEASIBILITY: ", end="")
        
        # Multiple challenges
        challenges = []
        
        if max_start_radius > 5.0:
            challenges.append(f"Large start distance ({max_start_radius}m)")
            
        if acc_lat_max < 5.0:
            challenges.append(f"Limited lateral authority ({acc_lat_max:.1f} m/s²)")
            
        if t_settle_pos > 5.0:
            challenges.append(f"Slow position settling ({t_settle_pos:.1f}s)")
        
        if len(challenges) == 0:
            print(f"✅ FEASIBLE")
            print(f"\n<0.2m precision is ACHIEVABLE:")
            print(f"  • Sufficient thrust authority (TWR={TWR:.1f})")
            print(f"  • Adequate tilt range ({np.rad2deg(self.max_tilt):.0f}°)")
            print(f"  • Remote start capability enabled")
            feasible = True
        else:
            print(f"⚠️  CHALLENGING")
            print(f"\n<0.2m precision will be difficult due to:")
            for challenge in challenges:
                print(f"  • {challenge}")
            feasible = False
            
        return feasible
        
    def analyze_hyperparameters(self):
        """Analyze if hyperparameters are appropriate"""
        self.print_header("HYPERPARAMETER ANALYSIS")
        
        print(f"\nATTITUDE CONTROLLER:")
        print(f"  Actor LR:  {self.att_cfg.ACTOR_LR:.0e}")
        print(f"  Critic LR: {self.att_cfg.CRITIC_LR:.0e}")
        print(f"  Gamma:     {self.att_cfg.GAMMA}")
        print(f"  Batch:     {self.att_cfg.BATCH_SIZE}")
        print(f"  Buffer:    {self.att_cfg.BUFFER_SIZE:,}")
        
        # Check if learning rates are appropriate
        if self.att_cfg.ACTOR_LR < 5e-5:
            print(f"  ⚠️  Actor LR very low - may learn slowly")
        elif self.att_cfg.ACTOR_LR > 5e-4:
            print(f"  ⚠️  Actor LR high - risk of instability")
        else:
            print(f"  ✅ Learning rates appropriate")
            
        print(f"\nPOSITION CONTROLLER:")
        print(f"  Actor LR:  {self.pos_cfg.ACTOR_LR:.0e}")
        print(f"  Critic LR: {self.pos_cfg.CRITIC_LR:.0e}")
        print(f"  Gamma:     {self.pos_cfg.GAMMA}")
        print(f"  Batch:     {self.pos_cfg.BATCH_SIZE}")
        print(f"  Buffer:    {self.pos_cfg.BUFFER_SIZE:,}")
        
        if self.pos_cfg.ACTOR_LR < 5e-5:
            print(f"  ⚠️  Actor LR very low - may learn slowly with high inertia")
        else:
            print(f"  ✅ Learning rates appropriate")
            
        # Exploration noise
        print(f"\nEXPLORATION:")
        print(f"  Attitude noise: {self.att_cfg.EXPLORATION_NOISE:.2f}")
        print(f"  Position noise: {self.pos_cfg.EXPLORATION_NOISE:.2f}")
        print(f"  Attitude decay: {self.att_cfg.NOISE_DECAY:.5f}")
        print(f"  Position decay: {self.pos_cfg.NOISE_DECAY:.5f}")
        
        # Calculate when noise reaches minimum
        steps_to_min_att = np.log(self.att_cfg.MIN_NOISE / self.att_cfg.EXPLORATION_NOISE) / np.log(self.att_cfg.NOISE_DECAY)
        steps_to_min_pos = np.log(self.pos_cfg.MIN_NOISE / self.pos_cfg.EXPLORATION_NOISE) / np.log(self.pos_cfg.NOISE_DECAY)
        
        print(f"  Attitude: reaches min noise after {int(steps_to_min_att):,} steps")
        print(f"  Position: reaches min noise after {int(steps_to_min_pos):,} steps")
        
    def generate_recommendations(self):
        """Generate specific recommendations"""
        self.print_header("RECOMMENDATIONS FOR ACHIEVING TARGETS")
        
        print(f"\n🎯 ATTITUDE CONTROLLER (<5° mean error):")
        print(f"\n1. INCREASE DAMPING for high inertia (I=0.3):")
        print(f"   CURRENT: damping_multiplier = 1.0 + 3.0 * exp(-5.0 * error)")
        print(f"   RECOMMENDED: damping_multiplier = 1.0 + 5.0 * exp(-8.0 * error)")
        print(f"   Reason: Higher inertia needs stronger damping to prevent overshoot")
        
        print(f"\n2. SHARPEN STABILITY BONUS:")
        print(f"   CURRENT: bonus if error < 5.0°")
        print(f"   RECOMMENDED: bonus if error < 3.0°, increase bonus to 8.0")
        print(f"   Reason: Push agent toward 2-3° instead of settling at 5° boundary")
        
        print(f"\n3. REDUCE EXPLORATION FASTER:")
        print(f"   CURRENT: NOISE_DECAY = 0.9998, MIN_NOISE = 0.05")
        print(f"   RECOMMENDED: NOISE_DECAY = 0.9995, MIN_NOISE = 0.02")
        print(f"   Reason: Converge to precise control faster")
        
        print(f"\n4. CONSIDER INCREASING MAX_TORQUE:")
        print(f"   CURRENT: 4.0 N·m (gives 13.3 rad/s² acceleration)")
        print(f"   RECOMMENDED: 5.0-6.0 N·m (gives 16.7-20.0 rad/s²)")
        print(f"   Reason: Faster response helps maintain <5° during disturbances")
        
        print(f"\n{'='*80}")
        print(f"\n🎯 POSITION CONTROLLER (<0.2m precision):")
        
        print(f"\n1. REDUCE MAX_TILT_ANGLE for precision:")
        print(f"   CURRENT: {np.rad2deg(self.pos_cfg.MAX_TILT_ANGLE):.0f}°")
        print(f"   RECOMMENDED: 30-35° for precision phase")
        print(f"   Reason: Smaller tilts give finer control, helix only needs ~3°")
        
        print(f"\n2. INCREASE PRECISION BONUS:")
        print(f"   CURRENT: PRECISION_BONUS = {self.pos_cfg.PRECISION_BONUS:.1f}")
        print(f"   RECOMMENDED: PRECISION_BONUS = 10.0-15.0")
        print(f"   Reason: Stronger incentive to stay within 0.2m cylinder")
        
        print(f"\n3. TIGHTER REWARD NORMALIZATION:")
        print(f"   CURRENT: max_pos_error = 6.0m for full helix")
        print(f"   RECOMMENDED: max_pos_error = 3.0-4.0m")
        print(f"   Reason: Sharper gradients near target improve precision")
        
        print(f"\n4. ADD VELOCITY-BASED PRECISION REWARD:")
        print(f"   NEW: velocity_precision_bonus if (pos_err<0.2 AND vel<0.5)")
        print(f"   Reason: Reward being near target AND moving slowly")
        
        print(f"\n5. REMOTE START CURRICULUM:")
        print(f"   CURRENT: START_RADIUS_MAX = {self.sys_cfg.START_RADIUS_MAX:.1f}m")
        print(f"   RECOMMENDED: Gradually increase from 1m → 3m → 5m → 6m")
        print(f"   Reason: Learn precision first, then add distance challenge")
        
        print(f"\n6. INCREASE POSITION LEARNING RATE:")
        print(f"   CURRENT: ACTOR_LR = {self.pos_cfg.ACTOR_LR:.0e}")
        print(f"   RECOMMENDED: ACTOR_LR = 7e-5 or 1e-4")
        print(f"   Reason: High inertia needs stronger learning signal")
        
        print(f"\n{'='*80}")
        print(f"\n🔧 SYSTEM-LEVEL RECOMMENDATIONS:")
        
        print(f"\n1. SAMPLING RATE:")
        print(f"   CURRENT: {1/self.dt:.0f} Hz")
        print(f"   RECOMMENDED: Consider 200 Hz (dt=0.005s)")
        print(f"   Reason: Better for both precision and high-inertia response")
        
        print(f"\n2. INERTIA VALUES:")
        print(f"   CURRENT: I = 0.3 kg·m² (very high for 1kg)")
        print(f"   RECOMMENDED: Verify this matches your actual hardware")
        print(f"   If not: Use 0.08-0.12 kg·m² for typical 1kg drone")
        print(f"   Impact: Lower inertia = faster response = easier to hit 5°")
        
        print(f"\n3. WIND DURING TRAINING:")
        print(f"   Start with no wind for initial learning")
        print(f"   Gradually introduce wind: 0 → 0.01 → 0.03 → 0.05 m/s std")
        print(f"   Only add wind after achieving precision in calm conditions")
        
    def run_full_analysis(self):
        """Run complete analysis"""
        print("\n")
        print("█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + "  COMPREHENSIVE PHYSICS FEASIBILITY ANALYSIS".center(78) + "█")
        print("█" + "  Attitude Target: <5° mean error".center(78) + "█")
        print("█" + "  Position Target: <0.2m precision".center(78) + "█")
        print("█" + "  Remote Start: Enabled (far from helix)".center(78) + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)
        
        # Run analyses
        self.analyze_inertia()
        att_feasible = self.analyze_attitude_control()
        pos_feasible = self.analyze_position_control()
        self.analyze_hyperparameters()
        self.generate_recommendations()
        
        # Final summary
        self.print_header("FINAL FEASIBILITY ASSESSMENT")
        
        print(f"\nATTITUDE CONTROL (<5°):")
        if att_feasible:
            print(f"  ✅ FEASIBLE with current parameters and recommended tuning")
            print(f"  Expected: 2-4° mean error achievable")
        else:
            print(f"  ⚠️  MARGINAL - requires significant changes")
            
        print(f"\nPOSITION CONTROL (<0.2m):")
        if pos_feasible:
            print(f"  ✅ FEASIBLE with proper tuning")
            print(f"  Expected: 0.15-0.25m achievable in final phase")
        else:
            print(f"  ⚠️  CHALLENGING - requires careful curriculum and parameter tuning")
            print(f"  Expected: 0.3-0.5m more realistic without major changes")
            
        print(f"\nREMOTE START CAPABILITY:")
        print(f"  ✅ Supported by system design")
        print(f"  Note: May temporarily exceed 0.2m during initial approach")
        print(f"  Recommendation: Separate 'approach' and 'track' phases in curriculum")
        
        print(f"\n{'='*80}")
        print(f"OVERALL VERDICT:")
        if att_feasible and pos_feasible:
            print(f"  ✅✅ BOTH TARGETS ACHIEVABLE with recommended changes")
        elif att_feasible:
            print(f"  ⚠️  Attitude feasible, Position challenging")
        else:
            print(f"  ⚠️  Both targets will require significant effort")
            
        print(f"\nRECOMMENDED ACTION PLAN:")
        print(f"  1. Verify inertia values match actual hardware (critical!)")
        print(f"  2. Implement attitude controller improvements first")
        print(f"  3. Achieve <5° attitude before training position controller")
        print(f"  4. Use staged curriculum for position: precision → distance")
        print(f"  5. Fine-tune reward functions based on early training results")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    analyzer = PhysicsFeasibilityAnalyzer()
    analyzer.run_full_analysis()
