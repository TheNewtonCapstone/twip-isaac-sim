import gc
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import torch

from omni.isaac.kit import SimulationApp

from core.base.base_agent import BaseAgent

# TODO: separate into 3 classes: BaseEnv, BaseTask, BaseAgent
# BaseEnv: contains the world, agents and settings
# BaseTask: contains the world (with its agents) task settings and the task logic
# BaseAgent: contains the agent settings and the agent logic
# The goal is to make it so that any task can be run in any environment with any agent
# Need to figure out how abc works in python and how to use it to enforce the structure of the classes


class BaseEnv(ABC):
    def __init__(self, world_settings, idx) -> None:
        from omni.isaac.core import World

        self.world = World(
            physics_dt=world_settings["physics_dt"],
            rendering_dt=world_settings["rendering_dt"],
            stage_units_in_meters=world_settings["stage_units_in_meters"],
            backend=world_settings["backend"],
            device=world_settings["device"],
        )
        self.world_settings = world_settings

        self.idx = idx

    @abstractmethod
    def construct(self, sim_app: SimulationApp) -> bool:
        from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics, PhysicsSchemaTools
        import omni.kit.commands

        self.sim_app = sim_app

        # Get stage handle
        stage = omni.usd.get_context().get_stage()

        # Add sun
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/distantLight"))
        sun.CreateIntensityAttr(500)

        # Add ground
        PhysicsSchemaTools.addGroundPlane(
            stage,
            "/world{}/groundPlane".format(self.idx),
            "Z",
            100,
            Gf.Vec3f(2.1 * self.idx, 0.0, 0.0),
            Gf.Vec3f(0.84, 0.40, 0.35),
        )

        return True

    @abstractmethod
    def step(
        self, _render
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        return self.world.step(render=_render)

    @abstractmethod
    def reset(
        self,
    ) -> Dict[str, torch.Tensor]:
        self.world.reset()

    @abstractmethod
    def add_agent(self, _agent: BaseAgent) -> bool:
        import omni.kit.commands

        # Get stage handle
        stage = omni.usd.get_context().get_stage()

        _agent.construct(stage)

        self.agent = _agent

    @abstractmethod
    def prepare(self) -> None:
        import omni.kit.commands

        # Ensure we start clean
        self.reset()

        # Start simulation
        omni.timeline.get_timeline_interface().play()

        # Do one step so that physics get loaded & dynamic control works
        self.sim_app.update()

        self.agent.prepare(self.sim_app)
