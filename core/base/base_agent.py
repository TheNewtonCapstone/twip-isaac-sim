from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    def __init__(self, _config) -> None:
        self.config = _config
        pass

    @abstractmethod
    def construct(self, stage) -> bool:
        pass

    @abstractmethod
    def pre_physics(self, _sim_app) -> None:
        pass

    @abstractmethod
    def get_observations(self) -> np.array:
        pass
