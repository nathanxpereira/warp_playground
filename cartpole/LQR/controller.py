from abc import ABC, abstractmethod
from scipy.linalg import solve_continuous_are
import torch
import numpy as np

g=9.81

class LQRController():
    def __init__(self, m_cart=1.0, m_pole=0.1, l=1.0):
        # Correct denominator from mathematical derivation
        M1 = m_cart + m_pole
        M2 = 4*m_cart+m_pole
        # Linearized system matrices from proper derivation
        A = np.array([[0, 1, 0, 0], 
                      [0, 0, -6*m_pole*g/M2, 0], 
                      [0, 0, 0, 1], 
                      [0, 0, (2*g*M1)/(l*M2), 0]])
        B = np.array([[0], [4/M2], [0], [-6/(l*M2)]])
        
        Q = np.diag([1.0, 5.0, 5.0, 5.0])
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