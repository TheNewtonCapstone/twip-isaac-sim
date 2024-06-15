from abc import ABC, abstractmethod


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
