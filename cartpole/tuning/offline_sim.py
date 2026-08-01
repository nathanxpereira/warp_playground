import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are


def compute_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Solve the LQR Riccati equation for K such that u = -Kx.

    Deliberately mirrors src/base_controllers/LQRBase.calculate_K exactly rather
    than importing it, so this module stays free of LQRBase's torch dependency
    and can run without Isaac Sim/torch installed. Keep the two in sync.
    """
    pass


def simulate_closed_loop(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    x0: np.ndarray,
    dt: float = 1 / 60,
    duration: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate dx/dt = (A - B@K) x from x0 and recover u(t) = -K @ x(t).

    duration=5.0 is deliberately longer than the Isaac Sim debug run's
    num_steps=50 (~0.83s at the 60Hz assumed in analyze_run.py) — 50 steps
    is too short to observe settling at all.

    Returns:
        t: (N,) time points
        x: (N,4) state trajectory
        u: (N,) control force trajectory
    """
    pass


def is_stable(A: np.ndarray, B: np.ndarray, K: np.ndarray) -> bool:
    """True if every eigenvalue of the closed-loop system has negative real part."""
    pass
