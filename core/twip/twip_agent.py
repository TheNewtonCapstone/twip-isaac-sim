import abc
from enum import Enum
import numpy as np

from core.base.base_agent import BaseAgent


class WheelDriveType(Enum):
    LEFT = 0
    RIGHT = 1


class TwipAgent(BaseAgent):
    def __init__(self, _config) -> None:
        super().__init__(_config)

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

        # needs to be imported within the function because of import dependencies
        from omni.isaac.sensor import IMUSensor

        self.imu = IMUSensor(
            prim_path=self.stage_path + "/imu",
            name="imu",
            frequency=60,
            translation=np.array([0, 0, 0]),
            orientation=np.array([1, 0, 0, 0]),
            linear_acceleration_filter_size=10,
            angular_velocity_filter_size=10,
            orientation_filter_size=10,
        )

    def pre_physics(self, _sim_app) -> None:
        super().pre_physics(_sim_app)

        from omni.isaac.core.articulations import Articulation

        art = Articulation(prim_path=self.stage_path)
        art.initialize()

    def get_observations(self) -> np.array:
        print(self.imu.get_current_frame())

        return np.array([])

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
