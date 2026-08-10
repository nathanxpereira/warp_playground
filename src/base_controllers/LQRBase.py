from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import numpy as np
import torch

from src.base_controllers.Controller import Controller

class LQRBase(Controller):
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> None:
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = self.compute_gain(A, B, Q, R)
        print(f"LQR gains: {self.K}")

    def compute_gain(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P) 
        return K
    
    def simulate_closed_loop(
        self,
        x0: np.ndarray,
        dt: float = 1 / 240,
        duration: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        K = self.compute_gain(self.A,self.B,self.Q,self.R)
        lqr_sys = lambda t,x: (self.A-self.B@K)@x
        sol = solve_ivp(lqr_sys, t_span=[0,duration], y0=x0.ravel(), t_eval=np.arange(0,duration,dt))
        t = sol.t
        x = sol.y
        u = -K@x
        return t,x,u

    def is_stable(self) -> bool:
        eigs = np.linalg.eig(self.A-self.B@self.K)
        return np.all(np.real(eigs.eigenvalues)<0)
    
    def compute_control(self, observations: dict, device) -> torch.Tensor:
        raise NotImplementedError()