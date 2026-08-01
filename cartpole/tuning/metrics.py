"""Performance metrics for scoring a closed-loop cartpole trajectory.

Deliberately independent of the LQR design cost (x'Qx + u'Ru): for any (Q, R)
the Riccati solution is already optimal for that same cost, so scoring
candidates by their own design cost during a search would be circular.
"""

import numpy as np


def settling_time(t: np.ndarray, angle: np.ndarray, tol: float = 0.05) -> float:
    """Last time |angle| exceeds tol; np.inf if it never stays within tol."""
    pass


def overshoot(angle: np.ndarray, x0_angle: float) -> float:
    """Max |angle| reached beyond the initial deviation, as a fraction of x0_angle."""
    pass


def max_abs_force(u: np.ndarray) -> float:
    pass