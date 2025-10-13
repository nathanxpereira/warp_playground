from abc import ABC, abstractmethod
from scipy.linalg import solve_continuous_are
import torch
import numpy as np


class Controller(ABC):
    @abstractmethod
    def compute_control(self, observations, device):
        pass

class LQRController(Controller):
    def __init__(self, m_cart=1.0, m_pole=0.1, l=0.5, g=9.81):
        M = m_cart + m_pole
        A = np.array([[0, 1, 0, 0], [0, 0, -m_pole*g/M, 0], [0, 0, 0, 1], [0, 0, g*M/(l*M), 0]])
        B = np.array([[0], [1/M], [0], [-1/(l*M)]])
        Q = np.diag([10.0, 1.0, 100.0, 10.0])
        R = np.array([[1.0]])
        P = solve_continuous_are(A, B, Q, R)
        self.K = (np.linalg.inv(R) @ B.T @ P).flatten()
        print(f"LQR gains: {self.K}")
    
    def compute_control(self, observations, device):
        cart_pos = observations["cart_position"][:, 0]
        cart_vel = observations["cart_velocity"][:, 0]
        pole_rot = observations["pole_rotation"]
        pole_angle = 2 * torch.atan2(pole_rot[:, 1], pole_rot[:, 3])
        pole_angular_vel = observations["pole_velocity"][:, 4]
        state = torch.stack([cart_pos, cart_vel, pole_angle, pole_angular_vel], dim=1)
        K_tensor = torch.tensor(self.K, device=device, dtype=state.dtype)
        return -(state * K_tensor).sum(dim=1)

class PIDController(Controller):
    def __init__(self, kp=100.0, ki=0.0, kd=20.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        print(f"PID gains: kp={kp}, ki={ki}, kd={kd}")
    
    def compute_control(self, observations, device):
        pole_rot = observations["pole_rotation"]
        pole_angle = 2 * torch.atan2(pole_rot[:, 1], pole_rot[:, 3])
        pole_angular_vel = observations["pole_velocity"][:, 4]
        self.integral += pole_angle.item() * 0.01
        return -self.kp * pole_angle - self.ki * self.integral - self.kd * pole_angular_vel