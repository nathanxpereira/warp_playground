from abc import ABC, abstractmethod
import torch

class Controller(ABC):
    def reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def compute_control(self, observations: dict, device) -> torch.Tensor:
        raise NotImplementedError()