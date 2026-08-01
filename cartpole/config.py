"""Configuration classes for cartpole simulation."""

from dataclasses import dataclass
import numpy as np


class CartPolePhysicsConfig:
    def __init__(self, cart_mass, pole_mass, pole_length):

        if cart_mass is None: raise Exception("Cart Mass cannot be None")
        if pole_mass is None: raise Exception("Pole Mass cannot be None")
        if pole_length is None: 
            pole_length = 1.0 
            print("Pole length cannot be None. Setting as 1.0")

        # Cart parameters
        self.cart_mass: float = cart_mass

        # Pole parameters
        self.pole_length: float = pole_length
        self.pole_mass: float = pole_mass

        # Constants
        self.gravity: float = 9.81
    

    def compute_system_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute linearized state-space matrices A and B.

        The state vector is x = [cart_pos, cart_vel, pole_angle, pole_angular_vel]
        and the control input is u = [force_on_cart].

        Returns:
            Tuple of (A, B) matrices for linear system dx/dt = Ax + Bu
                A: State transition matrix (4x4)
                B: Control input matrix (4x1)
        """
        M1 = self.cart_mass + self.pole_mass
        M2 = 4 * self.cart_mass + self.pole_mass
        g = self.gravity
        l = self.pole_length
        m_p = self.pole_mass

        # Linearized system matrices from proper derivation
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, -6*m_p*g/M2, 0],
            [0, 0, 0, 1],
            [0, 0, (2*g*M1)/(l*M2), 0]
        ])

        B = np.array([
            [0],
            [4/M2],
            [0],
            [-6/(l*M2)]
        ])

        return A, B

