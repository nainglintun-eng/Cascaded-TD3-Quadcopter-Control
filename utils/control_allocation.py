"""
Control Allocation Module
Converts desired body-frame accelerations to thrust and attitude commands
"""

import numpy as np


class ControlAllocator:
    """
    Converts desired accelerations to thrust and attitude references
    
    Input: [u_dot, v_dot, w_dot] - desired body frame accelerations (m/s^2)
    Output: [F_t, phi_des, theta_des, psi_des] - thrust and attitude commands
    
    This implements the geometric control allocation:
    - Total thrust from vertical acceleration requirement
    - Roll/pitch angles from horizontal acceleration requirements
    - Yaw angle maintained or set separately
    """
    
    def __init__(self, mass, gravity, max_thrust, max_tilt_angle):
        """
        Args:
            mass: quadcopter mass (kg)
            gravity: gravitational acceleration (m/s^2)
            max_thrust: maximum thrust (N)
            max_tilt_angle: maximum tilt angle (rad)
        """
        self.m = mass
        self.g = gravity
        self.max_thrust = max_thrust
        self.min_thrust = 0.3 * mass * gravity  # Minimum for stability
        self.max_tilt = max_tilt_angle
        
    def allocate(self, acc_body_des, current_attitude, yaw_des=0.0):
        """
        Allocate control from desired body accelerations
        
        Args:
            acc_body_des: [u_dot, v_dot, w_dot] desired body accelerations (m/s^2)
            current_attitude: [phi, theta, psi] current attitude (rad)
            yaw_des: desired yaw angle (rad), default 0
            
        Returns:
            thrust: total thrust command (N)
            att_des: [phi_des, theta_des, psi_des] desired attitude (rad)
        """
        u_dot_des, v_dot_des, w_dot_des = acc_body_des
        phi, theta, psi = current_attitude
        
        # Method 1: Direct geometric allocation
        # Total thrust needed (includes gravity compensation)
        # F_t / m = w_dot + g / (cos(phi)*cos(theta))
        # We want: a_z_world = w_dot_body * cos(phi)*cos(theta) - g = 0 (for hover)
        
        # Current rotation matrix (body to world)
        R = self._rotation_matrix(phi, theta, psi)
        
        # Desired total acceleration in world frame should fight gravity
        # a_world = R @ a_body - [0, 0, g]
        # For thrust: F_t = m * ||a_world + [0,0,g]||
        
        # Simplified approach: 
        # Thrust compensates for gravity and vertical acceleration
        thrust_vertical = self.m * (w_dot_des + self.g)
        
        # Additional thrust needed for tilting (to create horizontal forces)
        # F_t = thrust_vertical / (cos(phi)*cos(theta))
        # But we compute desired angles first
        
        # Desired tilt angles from horizontal accelerations
        # For small angles: u_dot ≈ -g*theta, v_dot ≈ g*phi
        # More accurately: we need the thrust vector to create these accelerations
        
        # Target total thrust magnitude (approximately)
        thrust_mag = np.sqrt(thrust_vertical**2 + 
                            (self.m * u_dot_des)**2 + 
                            (self.m * v_dot_des)**2)
        
        # Clip thrust
        thrust = np.clip(thrust_mag, self.min_thrust, self.max_thrust)
        
        # Desired attitude from acceleration requirements
        # phi_des ≈ arcsin(m * v_dot / F_t)
        # theta_des ≈ arcsin(-m * u_dot / F_t)
        
        # Avoid division by zero
        if thrust < 0.1:
            thrust = self.m * self.g
        
        # Compute desired tilt angles
        phi_des = np.arcsin(np.clip(self.m * v_dot_des / thrust, -0.9, 0.9))
        theta_des = np.arcsin(np.clip(-self.m * u_dot_des / thrust, -0.9, 0.9))
        
        # Clip to maximum tilt
        phi_des = np.clip(phi_des, -self.max_tilt, self.max_tilt)
        theta_des = np.clip(theta_des, -self.max_tilt, self.max_tilt)
        
        # Yaw handling (keep current or use desired)
        psi_des = yaw_des
        
        att_des = np.array([phi_des, theta_des, psi_des])
        
        return thrust, att_des
    
    def allocate_advanced(self, acc_body_des, current_attitude, current_rates, yaw_des=0.0):
        """
        Advanced allocation considering current state and rates
        
        This version accounts for:
        - Current attitude for better thrust vector computation
        - Angular rates for feedforward
        - More accurate small-angle approximation handling
        """
        u_dot_des, v_dot_des, w_dot_des = acc_body_des
        phi, theta, psi = current_attitude
        p, q, r = current_rates
        
        # Current rotation matrix
        R = self._rotation_matrix(phi, theta, psi)
        
        # Desired acceleration in world frame
        # We want: a_world = a_body_desired (rotated to world) + gravity compensation
        acc_body = np.array([u_dot_des, v_dot_des, w_dot_des])
        acc_world_desired = R @ acc_body
        
        # Add gravity to get required thrust acceleration
        thrust_acc_world = acc_world_desired + np.array([0, 0, self.g])
        
        # Thrust magnitude
        thrust = self.m * np.linalg.norm(thrust_acc_world)
        thrust = np.clip(thrust, self.min_thrust, self.max_thrust)
        
        # Desired thrust direction (unit vector)
        thrust_dir = thrust_acc_world / (np.linalg.norm(thrust_acc_world) + 1e-6)
        
        # Desired attitude from thrust direction
        # Thrust is along body z-axis, so we want: R_des @ [0,0,1] = thrust_dir
        
        # Yaw is independent, keep desired or current
        psi_des = yaw_des
        
        # From thrust direction, compute roll and pitch
        # thrust_dir = [tx, ty, tz]
        # For ZYX Euler: 
        # theta_des = arcsin(-tx)
        # phi_des = arctan2(ty, tz)
        
        tx, ty, tz = thrust_dir
        
        # Avoid singularities
        tz = max(tz, 0.1)  # Prevent looking too far down
        
        theta_des = np.arcsin(np.clip(-tx, -0.95, 0.95))
        phi_des = np.arctan2(ty, tz)
        
        # Clip to limits
        phi_des = np.clip(phi_des, -self.max_tilt, self.max_tilt)
        theta_des = np.clip(theta_des, -self.max_tilt, self.max_tilt)
        
        att_des = np.array([phi_des, theta_des, psi_des])
        
        return thrust, att_des
    
    @staticmethod
    def _rotation_matrix(phi, theta, psi):
        """Rotation matrix from body to world frame (ZYX Euler)"""
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        
        R = np.array([
            [ctheta*cpsi, sphi*stheta*cpsi - cphi*spsi, cphi*stheta*cpsi + sphi*spsi],
            [ctheta*spsi, sphi*stheta*spsi + cphi*cpsi, cphi*stheta*spsi - sphi*cpsi],
            [-stheta, sphi*ctheta, cphi*ctheta]
        ])
        
        return R


def test_control_allocator():
    """Test the control allocator"""
    allocator = ControlAllocator(mass=1.0, gravity=9.81, 
                                 max_thrust=18.0, max_tilt_angle=np.deg2rad(20))
    
    # Test 1: Hover (no acceleration)
    acc_des = np.array([0.0, 0.0, 0.0])
    current_att = np.array([0.0, 0.0, 0.0])
    thrust, att_des = allocator.allocate(acc_des, current_att)
    
    print("Test 1: Hover")
    print(f"  Desired acc: {acc_des}")
    print(f"  Thrust: {thrust:.2f} N (should be ~9.81 N)")
    print(f"  Attitude: phi={np.rad2deg(att_des[0]):.1f}°, theta={np.rad2deg(att_des[1]):.1f}°")
    print()
    
    # Test 2: Forward acceleration
    acc_des = np.array([2.0, 0.0, 0.0])  # 2 m/s^2 forward
    thrust, att_des = allocator.allocate(acc_des, current_att)
    
    print("Test 2: Forward acceleration (2 m/s^2)")
    print(f"  Desired acc: {acc_des}")
    print(f"  Thrust: {thrust:.2f} N")
    print(f"  Attitude: phi={np.rad2deg(att_des[0]):.1f}°, theta={np.rad2deg(att_des[1]):.1f}°")
    print(f"  (theta should be negative for forward)")
    print()
    
    # Test 3: Sideways acceleration
    acc_des = np.array([0.0, 1.5, 0.0])  # 1.5 m/s^2 right
    thrust, att_des = allocator.allocate(acc_des, current_att)
    
    print("Test 3: Sideways acceleration (1.5 m/s^2)")
    print(f"  Desired acc: {acc_des}")
    print(f"  Thrust: {thrust:.2f} N")
    print(f"  Attitude: phi={np.rad2deg(att_des[0]):.1f}°, theta={np.rad2deg(att_des[1]):.1f}°")
    print(f"  (phi should be positive for right)")
    print()
    
    # Test 4: Combined motion with vertical
    acc_des = np.array([1.0, 1.0, 2.0])
    thrust, att_des = allocator.allocate(acc_des, current_att)
    
    print("Test 4: Combined acceleration")
    print(f"  Desired acc: {acc_des}")
    print(f"  Thrust: {thrust:.2f} N")
    print(f"  Attitude: phi={np.rad2deg(att_des[0]):.1f}°, theta={np.rad2deg(att_des[1]):.1f}°")


if __name__ == "__main__":
    test_control_allocator()
