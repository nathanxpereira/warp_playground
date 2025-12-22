import torch

class PIDController():
    def __init__(self, kp=100.0, ki=0.0, kd=20.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        print(f"PID gains: kp={kp}, ki={ki}, kd={kd}")
    
    def reset(self):
        self.integral = 0.0
    
    def compute_control(self, observations, device):
        pole_rot = observations["pole_rotation"]
        pole_angle = 2 * torch.atan2(pole_rot[:, 1], pole_rot[:, 3])
        pole_angular_vel = observations["pole_velocity"][:, 4]
        self.integral += pole_angle.item() * 0.01
        return -self.kp * pole_angle - self.ki * self.integral - self.kd * pole_angular_vel