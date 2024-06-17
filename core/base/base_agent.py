from abc import ABC, abstractmethod
import torch


class BaseAgent(ABC):
    def __init__(self, _config) -> None:
        self.config = _config
        pass

    @abstractmethod
    def construct(self, stage) -> bool:
        pass

    @abstractmethod
    def prepare(self, _sim_app) -> None:
        pass

    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        pass
