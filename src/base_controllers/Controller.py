from abc import ABC, abstractmethod
import torch

class Controller(ABC):
    """Abstract base class for cartpole controllers.

    All controllers must implement compute_control() to calculate
    control forces based on observations. The reset() method is
    optional and should clear any internal state.
    """

    def reset(self) -> None:
        """Reset controller internal state. Override if needed."""
        pass

    @abstractmethod
    def compute_control(self, observations: dict, device) -> torch.Tensor:
        """Compute control force(s) for the cart.

        Args:
            observations: Dictionary containing system state observations
            device: Torch device for tensor operations

        Returns:
            Tensor of shape (num_envs,) with control forces
        """
        raise NotImplementedError()