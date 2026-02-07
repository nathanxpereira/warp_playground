"""Configuration classes for cartpole simulation."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CartPolePhysicsConfig:
    """Physical parameters for the cartpole system."""

    # Cart parameters
    cart_length: float = 0.4
    cart_width: float = 0.3
    cart_height: float = 0.2
    cart_mass: float = 1.0

    # Pole parameters
    pole_radius: float = 0.02
    pole_length: float = 1.0
    pole_mass: float = 0.1

    # Initial conditions
    start_angle_degrees: float = 5.0

    # Constants
    gravity: float = 9.81

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


@dataclass(frozen=True)
class SimulationConfig:
    """Simulation control parameters."""

    warmup_steps: int = 3
    render: bool = True
    physics_dt: float = 1.0/60.0
