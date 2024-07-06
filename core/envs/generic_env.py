import gc
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

from isaacsim import SimulationApp

from core.base.base_agent import BaseAgent
from core.base.base_env import BaseEnv


# TODO: should be called GenericTwipEnv
class GenericEnv(BaseEnv):
    def __init__(self, sim_app: SimulationApp, world_settings, num_envs):
        super().__init__(sim_app, world_settings, num_envs=num_envs)

    def construct(self, agent: BaseAgent) -> bool:
        super().construct(agent)

        import omni.isaac.core
        from omni.isaac.cloner import GridCloner
        from omni.isaac.core.articulations import ArticulationView
        from omni.isaac.core.utils.prims import define_prim

        # clone the agent
        cloner = GridCloner(spacing=1)
        cloner.define_base_env("/World/envs")
        self.base_agent_path = "/World/envs/e_0"
        define_prim(self.base_agent_path)

        self.agent.construct(self.base_agent_path, self.world)

        self.agent_paths = cloner.generate_paths("/World/envs/e", self.num_envs)
        self.agent_imu_paths = [f"{path}/twip/body/imu" for path in self.agent_paths]

        cloner.filter_collisions(
            physicsscene_path="/physicsScene",
            collision_root_path="/collisionGroups",
            prim_paths=self.agent_paths,
            global_paths=["/World/groundPlane"],
        )
        cloner.clone(
            source_prim_path=self.base_agent_path,
            prim_paths=self.agent_paths,
        )

        self.twip_art_view = ArticulationView(
            prim_paths_expr="/World/envs/e.*/twip/body",
            name="twip_art_view",
        )
        self.world.scene.add(self.twip_art_view)

        self.world.reset()

        return self.base_agent_path

    def step(self, actions: torch.Tensor, render: bool) -> torch.Tensor:
        # apply actions to the cloned agents

        super().step(actions, render)

        # get observations from the cloned agents
        obs = self._gather_imus_frame()

        return obs

    def reset(
        self,
        indices: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        super().reset()
        self.twip_art_view.initialize()
        return

    def _apply_actions(self, actions: torch.Tensor) -> None:
        # 0. add the option to choose between position, velocity or torque (effort) control
        # 1. create an articulation action for the view
        # 2. apply the actions to the view
        pass

    def _gather_imus_frame(self) -> torch.Tensor:
        from omni.isaac.sensor import _sensor

        i_imu = _sensor.acquire_imu_sensor_interface()

        imus_frame = torch.zeros(self.num_envs, 10)

        for i, imu_path in enumerate(self.agent_imu_paths):
            imu_data = i_imu.get_sensor_reading(imu_path)
            imus_frame[i, :] = torch.tensor(
                [
                    imu_data.lin_acc_x,
                    imu_data.lin_acc_y,
                    imu_data.lin_acc_z,
                    imu_data.ang_vel_x,
                    imu_data.ang_vel_y,
                    imu_data.ang_vel_z,
                    # the quaternion is stored in WXYZ format in rest of Isaac
                    imu_data.orientation[3],
                    imu_data.orientation[0],
                    imu_data.orientation[1],
                    imu_data.orientation[2],
                ],
            )

        return imus_frame
