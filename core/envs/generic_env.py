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

        # add a ground plane
        self.world.scene.add_default_ground_plane()

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
        self._apply_actions(actions)

        super().step(actions, render)

        # get observations from the cloned agents
        obs = self._gather_imus_frame()

        return obs

    def reset(
        self,
        indices: torch.Tensor = None,
    ) -> None:
        assert indices is None or indices.ndim == 1, "Indices must be a 1D tensor"

        # we assume it's a full reset
        if indices is None:
            super().reset()  # reset the world too, because we're doing a full reset

            indices = torch.arange(self.num_envs)

        num_to_reset = len(indices)

        self.twip_art_view.set_joint_velocity_targets(
            torch.zeros(num_to_reset, 2), indices=indices
        )
        # using set_velocities instead of individual methods (lin & ang), because it's the only method supported in the GPU pipeline
        self.twip_art_view.set_velocities(torch.zeros(num_to_reset, 6), indices=indices)
        self.twip_art_view.set_joint_positions(
            torch.zeros(num_to_reset, 2), indices=indices
        )
        self.twip_art_view.set_joint_efforts(
            torch.zeros(num_to_reset, 2), indices=indices
        )

        # orientations need to have the quaternion in WXYZ format, and 1 as the first element, the rest being zeros
        orientations = torch.tile(torch.tensor([1.0, 0, 0, 0]), (num_to_reset, 1))

        # from GridCloner
        # translations should arrange all agents in a grid, with a spacing of 1, even if it's not a perfect square
        # an agent should always be at the same position in the grid (same index as specified in indices)
        spacing = 1
        num_per_row = int(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_per_row)
        num_cols = np.ceil(self.num_envs / num_rows)

        row_offset = 0.5 * spacing * (num_rows - 1)
        col_offset = 0.5 * spacing * (num_cols - 1)

        translations = torch.zeros(num_to_reset, 3)

        for i, idx in enumerate(indices):
            row = idx // num_cols
            col = idx % num_cols
            x = row_offset - row * spacing
            y = col * spacing - col_offset

            translations[i, 0] = x
            translations[i, 1] = y
            translations[i, 2] = 0.115

        self.twip_art_view.set_local_poses(
            translations=translations,
            orientations=orientations,
            indices=indices,
        )

        return

    def _apply_actions(self, actions: torch.Tensor) -> None:
        self.twip_art_view.set_joint_efforts(actions)

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
