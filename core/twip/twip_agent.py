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


# this class describes how the agent will be constructed, nothing more
class TwipAgent(BaseAgent):
    def __init__(self, config) -> None:
        super().__init__(config)

    def construct(self, root_path: str) -> bool:
        super().construct(root_path)

        twip_prim_path = root_path + "/twip"

        # these only work after SimulationApp is initialized (to be done in scripts that import this class)
        import omni.isaac.core.utils.stage as stage_utils

        stage_utils.add_reference_to_stage(
            self.config["twip_usd_path"], prim_path=twip_prim_path
        )  # /envs/0/twip

        # needs to be imported within the function because of import dependencies
        from omni.isaac.sensor import IMUSensor

        self.imu = IMUSensor(
            prim_path=twip_prim_path + "/body/imu",
            name="imu",
            frequency=200,
            translation=np.array([0, 0, 0]),
            orientation=np.array([1, 0, 0, 0]),
            linear_acceleration_filter_size=10,
            angular_velocity_filter_size=10,
            orientation_filter_size=10,
        )
