"""Search LQR Q/R weights offline, seeded from Bryson's rule.

Run with: python Cartpole/tune_lqr.py  (or `cd Cartpole && python tune_lqr.py`)
No Isaac Sim / GPU required — validate the winning gains in Isaac Sim separately.
"""

import numpy as np
from scipy.optimize import minimize

from config import CartPolePhysicsConfig
from tuning.bryson import bryson_qr, DEFAULT_X_MAX, DEFAULT_U_MAX
from tuning.offline_sim import compute_gain, simulate_closed_loop, is_stable
from tuning.metrics import settling_time, overshoot, max_abs_force




def objective(log_weights: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    pass


def check_robustness(K: np.ndarray, spread: float = 0.2) -> list[str]:
    pass


def main() -> None:
    pass


if __name__ == "__main__":
    main()
