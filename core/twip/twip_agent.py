import abc
from enum import Enum
import numpy as np
import torch

from core.base.base_agent import BaseAgent

from typing import Dict


class WheelDriveType(Enum):
    LEFT = 0
    RIGHT = 1


# when implementing ROS, check the following link: https://github.com/ros2/examples/blob/rolling/rclpy/topics/minimal_publisher
# probably a good idea to make separate wrapper classes for each joint (or rather, for each object that will publish/subscribe to ROS messages)


class TwipAgent(BaseAgent):
    def __init__(self, config, idx) -> None:
        super().__init__(config, idx)

    def construct(self, stage) -> bool:
        super().construct(stage)

        # these only work after SimulationApp is initialized (to be done in scripts that import this class)
        from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics
        import omni.kit.commands

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = False
        import_config.fix_base = False

        status, stage_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=self.config["twip_urdf_path"],
            import_config=import_config,
            get_articulation_root=True,
        )
        self.stage_path = stage_path

        self.lwd = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(self.stage_path + "/lwheel"), "angular"
        )
        self.rwd = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(self.stage_path + "/rwheel"), "angular"
        )

        self.set_damping(WheelDriveType.LEFT, 15000)
        self.set_damping(WheelDriveType.RIGHT, 15000)

        self.set_stiffness(WheelDriveType.LEFT, 0)
        self.set_stiffness(WheelDriveType.RIGHT, 0)

        # needs to be imported within the function because of import dependencies
        from omni.isaac.sensor import IMUSensor

        self.imu = IMUSensor(
            prim_path=self.stage_path + "/imu",
            name="imu",
            frequency=200,
            translation=np.array([0, 0, 0]),
            orientation=np.array([1, 0, 0, 0]),
            linear_acceleration_filter_size=10,
            angular_velocity_filter_size=10,
            orientation_filter_size=10,
        )

    def prepare(self, _sim_app) -> None:
        super().prepare(_sim_app)

        from omni.isaac.core.articulations import Articulation

        art = Articulation(
            prim_path=self.stage_path,
            position=np.array([2.1 * self.idx, 0, 0]),
        )
        art.initialize()

    def get_observations(self) -> Dict[str, torch.Tensor]:
        frame = self.imu.get_current_frame()

        lin_acc = frame["lin_acc"]
        ang_vel = frame["ang_vel"]
        orientation = frame["orientation"]

        return {"lin_acc": lin_acc, "ang_vel": ang_vel, "orientation": orientation}

    def set_damping(self, type: WheelDriveType, val) -> None:
        (self.lwd if type == WheelDriveType.LEFT else self.rwd).GetDampingAttr().Set(
            val
        )

    def set_stiffness(self, type: WheelDriveType, val) -> None:
        (self.lwd if type == WheelDriveType.LEFT else self.rwd).GetStiffnessAttr().Set(
            val
        )

    def set_target_velocity(self, type: WheelDriveType, val) -> None:
        (
            self.lwd if type == WheelDriveType.LEFT else self.rwd
        ).GetTargetVelocityAttr().Set(val)
