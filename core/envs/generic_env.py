import gc
from abc import ABC, abstractmethod
import torch
from typing import Dict, Tuple, Any

from isaacsim import SimulationApp

from core.base.base_agent import BaseAgent
from core.base.base_env import BaseEnv


class GenericEnv(BaseEnv):
    def __init__(self, sim_app: SimulationApp, world_settings, num_envs):
        super().__init__(sim_app, world_settings, num_envs=num_envs)

    def construct(self, agent: BaseAgent) -> bool:
        return super().construct(agent)

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
