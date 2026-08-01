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
        cart_pos = observations["cart_position"][:, 0]
        cart_vel = observations["cart_velocity"][:, 0]
        pole_rot = observations["pole_rotation"]
        pole_angle = 2 * torch.atan2(pole_rot[:, 1], pole_rot[:, 3])
        pole_angular_vel = observations["pole_velocity"][:, 4]
        state = torch.stack([cart_pos, cart_vel, pole_angle, pole_angular_vel], dim=1)
        K_tensor = torch.tensor(self.K, device=device, dtype=state.dtype)
        return -(state * K_tensor).sum(dim=1)