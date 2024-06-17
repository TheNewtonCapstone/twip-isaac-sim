import gc
from abc import ABC, abstractmethod
import torch
from typing import Dict, Tuple, Any

from core.base.base_agent import BaseAgent
from core.base.base_env import BaseEnv


class GenericEnv(BaseEnv):
    def __init__(self, _o_world_settings):
        super().__init__(_o_world_settings)

    def construct(self) -> bool:
        return super().construct()

    def step(
        self, _render
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        return super().step(_render)

    def reset(
        self,
    ) -> Dict[str, torch.Tensor]:
        return super().reset()

    def add_agent(self, _agent: BaseAgent) -> bool:
        return super().add_agent(_agent)

    def prepare(self, _sim_app) -> None:
        return super().prepare(_sim_app)
