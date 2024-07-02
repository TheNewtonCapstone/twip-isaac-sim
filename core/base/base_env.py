import gc
from abc import ABC, abstractmethod
import time
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
    def __init__(self, sim_app: SimulationApp, world_settings, num_envs) -> None:
        from omni.isaac.core import World

        self.world_settings = world_settings
        self.world = World(
            physics_dt=self.world_settings["physics_dt"],
            rendering_dt=self.world_settings["rendering_dt"],
            stage_units_in_meters=self.world_settings["stage_units_in_meters"],
            backend=self.world_settings["backend"],
            device=self.world_settings["device"],
            set_defaults=False,
        )

        self.sim_app = sim_app
        self.num_envs = num_envs

    def big_bang(self) -> bool:
        from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics, PhysicsSchemaTools
        import omni.kit.commands

        # Get stage handle
        stage = omni.usd.get_context().get_stage()

        # Enable physics
        phys_scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))

        # Set gravity
        phys_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        phys_scene.CreateGravityMagnitudeAttr().Set(9.81)

        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
        physxSceneAPI.CreateEnableGPUDynamicsAttr(True)
        physxSceneAPI.CreateBroadphaseTypeAttr("GPU")

        # Add sun
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/distantLight"))
        sun.CreateIntensityAttr(500)

        return True

    @abstractmethod
    def construct(self, agent: BaseAgent) -> str:
        self.agent = agent

        from pxr import Gf, PhysicsSchemaTools
        import omni.kit.commands

        stage_path = "/envs/env_0"

        # Get stage handle
        stage = omni.usd.get_context().get_stage()

        PhysicsSchemaTools.addGroundPlane(
            stage,
            stage_path + "/world/groundPlane",
            "Z",
            2,
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(0.84, 0.40, 0.35),
        )

        self.agent.construct(stage_path)
        
        import omni.isaac.kit
        from omni.isaac.cloner import GridCloner
        from omni.isaac.core.articulations import ArticulationView
        from pxr import UsdGeom

        cloner = GridCloner(spacing=3)
        UsdGeom.Xform.Define(stage, stage_path)
        cloner.clone(
            source_prim_path=stage_path,
            base_env_path=stage_path, 
            copy_from_source=True,
            prim_paths=cloner.generate_paths("/envs/env", self.num_envs),
        )

        self.world.reset()

        #prims = ArticulationView(prim_paths_expr="/envs/env.*/twip/body", name="twip_view", reset_xform_properties=False)
        #prims.initialize()

        return stage_path

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
