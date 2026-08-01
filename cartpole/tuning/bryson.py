"""Bryson's rule: derive LQR Q/R weights from stated tolerances."""

import numpy as np

# Starting tolerances — replace with your actual track/actuator limits, not measured values.
DEFAULT_X_MAX = None
DEFAULT_U_MAX = None


def bryson_qr(x_max: list[float], u_max: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Compute diagonal LQR weights from maximum acceptable deviations.

    Q_ii = 1 / x_max[i]**2, R_jj = 1 / u_max[j]**2 — puts every state/control
    on a comparable footing in the cost regardless of physical units.

    Args:
        x_max: Max acceptable deviation per state, [cart_pos, cart_vel, pole_angle, pole_angular_vel].
        u_max: Max acceptable control effort, [force].

    Returns:
        (Q, R): Q is (4,4) diagonal, R is (1,1).
    """
    Q = None
    R = None
    return Q, R
