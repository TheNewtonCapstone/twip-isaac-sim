import gc
from abc import abstractmethod

import isaacsim
from omni.isaac.kit import SimulationApp

config = {"headless": False}
sim_app = SimulationApp(config)

from omni.isaac.core import World
from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics

from base_agent import BaseAgent

# TODO: separate into 3 classes: BaseEnv, BaseTask, BaseAgent
# BaseEnv: contains the world, agents and settings
# BaseTask: contains the world (with its agents) task settings and the task logic
# BaseAgent: contains the agent settings and the agent logic
# The goal is to make it so that any task can be run in any environment with any agent
# Need to figure out how abc works in python and how to use it to enforce the structure of the classes


class BaseEnv(object):
    def __init__(self, _o_world_settings) -> None:
        self.o_world_settings = _o_world_settings
        self.o_world = World()
        return

    def construct(self):
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
        physxSceneAPI.CreateEnableCCDAttr(True)
        physxSceneAPI.CreateEnableStabilizationAttr(True)
        physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
        physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
        physxSceneAPI.CreateSolverTypeAttr("TGS")

        # Add sun
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
        sun.CreateIntensityAttr(500)

        # Add ground
        omni.kit.commands.execute(
            "AddGroundPlaneCommand",
            stage=stage,
            planePath="/planePath",
            axis="Z",
            size=1500.0,
            position=Gf.Vec3f(0, 0, -0.2),
            color=Gf.Vec3f(0.83, 0.4, 0.25),
        )

    def step(self, _render):
        self.o_world.step(render=_render)

    def reset(self):
        self.o_world.reset()

    def add_agent(self, _agent: BaseAgent) -> bool:
        import omni.kit.commands

        # Get stage handle
        stage = omni.usd.get_context().get_stage()

        _agent.construct(stage)

        self.agent = _agent

    def pre_play(self, _sim_app) -> None:
        import omni.kit.commands

        # Ensure we start clean
        self.reset()

        # Start simulation
        omni.timeline.get_timeline_interface().play()

        # Do one step so that physics get loaded & dynamic control works
        _sim_app.update()

        self.agent.pre_physics(_sim_app)


world_settings = {
    "physics_dt": 1.0 / 60.0,
    "stage_units_in_neters": 1.0,
    "rendering_dt": 1.0 / 60.0,
}

base_env = BaseEnv(world_settings)
base_env.construct()

agent_settings = {}

base_agent = BaseAgent(agent_settings)
base_env.add_agent(base_agent)
base_env.pre_play(sim_app)

while sim_app.is_running():
    base_env.step(_render=True)
