from src.base_controllers.Controller import Controller

class PIDBase(Controller):
    """PID controller for cartpole system.

    Applies proportional-integral-derivative control to stabilize the pole
    in the upright position by controlling cart acceleration.
    """

    def __init__(self, kp: float = 100.0, ki: float = 0.0, kd: float = 20.0) -> None:
        """Initialize PID controller with specified gains.

        Args:
            kp: Proportional gain (response to current angle error)
            ki: Integral gain (response to accumulated angle error)
            kd: Derivative gain (response to angular velocity)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0

    def reset(self) -> None:
        """Reset integral accumulator to zero."""
        self.integral = 0.0
