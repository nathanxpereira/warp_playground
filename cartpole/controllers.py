import torch

from src.base_controllers.LQRBase import LQRBase

class LQRController(LQRBase):
    def compute_control(self, observations: dict, device) -> torch.Tensor:
        cart_pos = observations["cart_position"]
        cart_vel = observations["cart_velocity"]
        pole_ang = observations["pole_rotation"]
        pole_ang_vel = observations["pole_velocity"]
        state = torch.stack([cart_pos, cart_vel, pole_ang, pole_ang_vel], dim=0)
        K_tensor = torch.tensor(self.K, device=device, dtype=state.dtype)
        control_force = -K_tensor@state
        return control_force
    
