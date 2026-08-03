from scipy.linalg import solve_continuous_are
import torch
from src.base_controllers.LQRBase import LQRBase

class LQRController(LQRBase):
    def compute_control(self, observations: dict, device) -> torch.Tensor:
        """Compute LQR optimal control law u = -Kx.

        Args:
            observations: State dict with cart_position, cart_velocity,
                         pole_rotation, and pole_velocity
            device: Torch device for tensor operations

        Returns:
            Control forces of shape (num_envs,)
        """
        cart_pos = observations["cart_position"]
        cart_vel = observations["cart_velocity"]
        pole_ang = observations["pole_rotation"]
        pole_ang_vel = observations["pole_velocity"]
        state = torch.stack([cart_pos, cart_vel, pole_ang, pole_ang_vel], dim=1)
        K_tensor = torch.tensor(self.K, device=device, dtype=state.dtype)
        control_force = -state@K_tensor
        return control_force