import gc
from abc import ABC, abstractmethod

from omni.isaac.kit import SimulationApp

from tasks.base.base_agent import BaseAgent

# TODO: separate into 3 classes: BaseEnv, BaseTask, BaseAgent
# BaseEnv: contains the world, agents and settings
# BaseTask: contains the world (with its agents) task settings and the task logic
# BaseAgent: contains the agent settings and the agent logic
# The goal is to make it so that any task can be run in any environment with any agent
# Need to figure out how abc works in python and how to use it to enforce the structure of the classes


class BaseEnv(ABC):
    def __init__(self, _o_world_settings) -> None:
        from omni.isaac.core import World

        self.o_world_settings = _o_world_settings
        self.o_world = World()
        return

    @abstractmethod
    def construct(self):
        from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics
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

    @abstractmethod
    def step(self, _render):
        self.o_world.step(render=_render)

    @abstractmethod
    def reset(self):
        self.o_world.reset()

    @abstractmethod
    def add_agent(self, _agent: BaseAgent) -> bool:
        import omni.kit.commands

        # Get stage handle
        stage = omni.usd.get_context().get_stage()

        _agent.construct(stage)

        self.agent = _agent

    @abstractmethod
    def pre_play(self, _sim_app: SimulationApp) -> None:
        import omni.kit.commands

        # Ensure we start clean
        self.reset()

        # Start simulation
        omni.timeline.get_timeline_interface().play()

        # Do one step so that physics get loaded & dynamic control works
        _sim_app.update()

        self.agent.pre_physics(_sim_app)
