from abc import ABC, abstractmethod
from typing import Dict
import torch

from core.base.base_agent import BaseAgent
from core.terrain.terrain import TerrainBuilder

# TODO: separate into 3 classes: BaseEnv, BaseTask, BaseAgent
# BaseEnv: contains the world, agents and settings
# BaseTask: contains the world (with its agents) task settings and the task logic
# BaseAgent: contains the agent settings and the agent logic
# The goal is to make it so that any task can be run in any environment with any agent
# Need to figure out how abc works in python and how to use it to enforce the structure of the classes


class BaseEnv(ABC):
    def __init__(
        self,
        world_settings: Dict,
        num_envs: int,
    ) -> None:
        self.world = None
        self.agent = None
        self.world_settings = world_settings
        self.num_envs = num_envs

    @abstractmethod
    def construct(self, agent: BaseAgent, terrain: TerrainBuilder) -> str:
        self.agent = agent  # save the agent class for informative purposes (i.e. introspection/debugging)

        import omni.isaac.core
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import get_current_stage
        from pxr import Sdf, UsdLux

        stage = get_current_stage()

        self.world: World = World(
            physics_dt=self.world_settings["physics_dt"],
            rendering_dt=self.world_settings["rendering_dt"],
            stage_units_in_meters=self.world_settings["stage_units_in_meters"],
            backend=self.world_settings["backend"],
            device=self.world_settings["device"],
        )

        # Adjust physics scene settings (mainly for GPU memory allocation)
        phys_context = self.world.get_physics_context()
        phys_context.set_gpu_found_lost_aggregate_pairs_capacity(
            max(self.num_envs * 64, 1024)
        )  # 1024 is the default value, eyeballed the other value
        phys_context.set_gpu_total_aggregate_pairs_capacity(
            max(self.num_envs * 64, 1024)
        )  # 1024 is the default value, eyeballed the other value

        # Add sun
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/distantLight"))
        sun.CreateIntensityAttr(500)

        return None

    @abstractmethod
    def step(self, actions: torch.Tensor, render: bool) -> torch.Tensor:
        self.world.step(render=render)
        return None

    @abstractmethod
    def reset(
        self,
        indices: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        self.world.reset()

        return None
