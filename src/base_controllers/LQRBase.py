from scipy.linalg import solve_continuous_are
import numpy as np
from src.base_controllers.Controller import Controller

class LQRBase(Controller):
    """Linear Quadratic Regulator controller for cartpole system.

    Computes optimal control law u = -Kx by solving the continuous-time
    Algebraic Riccati Equation with specified state and control weights.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> None:
        """Initialize LQR controller with system matrices and cost weights.

        Args:
            A: State transition matrix (4x4) for linearized system
            B: Control input matrix (4x1) for linearized system
            Q: State cost matrix (4x4), typically diagonal
            R: Control cost matrix (1x1), scalar weight on control effort
        """
        self.K = self.calculate_K(A, B, Q, R)
        print(f"LQR gains: {self.K}")

    def calculate_K(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Calculate LQR gain matrix K by solving Riccati equation.

        Args:
            A: State transition matrix
            B: Control input matrix
            Q: State cost matrix
            R: Control cost matrix

        Returns:
            Gain matrix K of shape (4,) such that u = -Kx
        """
        P = solve_continuous_are(A, B, Q, R)
        K = (np.linalg.inv(R) @ B.T @ P).flatten()
        return K
