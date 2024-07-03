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
    def __init__(
        self,
        sim_app: SimulationApp,
        world_settings: Dict,
        num_envs: int,
    ) -> None:
        self.world_settings = world_settings
        self.sim_app = sim_app
        self.num_envs = num_envs

    @abstractmethod
    def construct(self, agent: BaseAgent) -> str:
        self.agent = agent  # save the agent class for informative purposes

        import omni.isaac.core
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import create_new_stage, get_current_stage
        from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdLux, UsdPhysics

        # Make sure we have a brand new stage (we can also switch this out for a from-USD-loaded stage)
        create_new_stage()
        stage = get_current_stage()

        self.world = World(
            physics_dt=self.world_settings["physics_dt"],
            rendering_dt=self.world_settings["rendering_dt"],
            stage_units_in_meters=self.world_settings["stage_units_in_meters"],
            backend=self.world_settings["backend"],
            device=self.world_settings["device"],
            set_defaults=False,
        )
        self.world.initialize_physics()

        # Enable physics
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
        # Set gravity
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
        physxSceneAPI.CreateEnableCCDAttr(True)
        physxSceneAPI.CreateEnableStabilizationAttr(True)
        physxSceneAPI.CreateEnableGPUDynamicsAttr(True)
        physxSceneAPI.CreateBroadphaseTypeAttr("GPU")
        physxSceneAPI.CreateSolverTypeAttr("TGS")

        # Adjust physics scene settings (mainly for GPU memory allocation)
        phys_context = self.world.get_physics_context()
        phys_context.set_gpu_found_lost_aggregate_pairs_capacity(
            max(self.num_envs * 196, 1024)
        )  # 1024 is the default value, eyeballed the other value
        phys_context.set_gpu_total_aggregate_pairs_capacity(
            max(self.num_envs * 64, 1024)
        )  # 512 is the default value, eyeballed the other value

        # Add sun
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/distantLight"))
        sun.CreateIntensityAttr(500)

        self.agent_stage_path = "/envs/env_0"

        # Get stage handle
        stage = get_current_stage()

        PhysicsSchemaTools.addGroundPlane(
            stage,
            self.agent_stage_path + "/world/groundPlane",
            "Z",
            1,
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(0.84, 0.40, 0.35),
        )

        self.agent.construct(self.agent_stage_path)

        from omni.isaac.cloner import GridCloner
        from omni.isaac.core.articulations import ArticulationView
        from omni.isaac.core.robots import RobotView
        from pxr import UsdGeom

        cloner = GridCloner(spacing=3)
        UsdGeom.Xform.Define(stage, self.agent_stage_path)
        cloner.clone(
            source_prim_path=self.agent_stage_path,
            base_env_path=self.agent_stage_path,
            copy_from_source=True,
            prim_paths=cloner.generate_paths("/envs/env", self.num_envs),
        )

        self.world.reset()

        self.prims = RobotView(
            prim_paths_expr="/envs/env.*/twip/body",
            name="twip_art_view",
        )
        self.prims.initialize()

        return self.agent_stage_path

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
