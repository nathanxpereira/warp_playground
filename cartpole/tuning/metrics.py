import numpy as np

def settling_time(t: np.ndarray, angle: np.ndarray, tol: float = 0.05) -> float:
    angle_flag = np.abs(angle[1:]-angle[:-1]) < tol
    last_t = angle[1:][angle_flag]
    return last_t[-1] if len(last_t) else np.inf


def overshoot(angle: np.ndarray, x0_angle: float) -> float:
    """
    Max |angle| reached beyond the initial deviation, as a fraction of x0_angle.
    """
    return np.max(np.abs(angle))/x0_angle


def max_abs_force(u: np.ndarray) -> float:
    return np.max(np.abs(u))