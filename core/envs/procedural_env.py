import math

import torch
import numpy as np

from core.base.base_agent import BaseAgent
from core.base.base_env import BaseEnv
from core.sensors.imu.imu import IMU
from core.terrain.terrain import TerrainBuilder


class ProceduralEnv(BaseEnv):
    def __init__(self, world_settings, num_envs, terrain_builders, randomization_settings) -> None:
        super().__init__(world_settings, num_envs, terrain_builders, randomization_settings)

        self.agent_positions = torch.zeros(self.num_envs, 3)

    def construct(self, agent: BaseAgent) -> bool:
        super().construct(agent)

        import omni.isaac.core
        from omni.isaac.cloner import Cloner
        from omni.isaac.core.articulations import ArticulationView
        from omni.isaac.core.utils.stage import get_current_stage
        from omni.isaac.core.utils.prims import define_prim

        stage = get_current_stage()
        num_terrains = len(self.terrain_builders)
        terrains_size = self.terrain_builders[0].size

        # generates a list of positions for each of the terrains, in a grid pattern
        perf_num_terrains_side = math.ceil(math.sqrt(num_terrains))
        terrain_positions = torch.tensor(
            [
                [
                    (i % perf_num_terrains_side) * terrains_size[0] - terrains_size[0] / 2,
                    (i // perf_num_terrains_side) * terrains_size[1] - terrains_size[1] / 2,
                    0
                ]
                for i in range(num_terrains)
            ]
        ).tolist()

        terrain_paths = []
        agent_batch_qty = int(math.ceil(self.num_envs / num_terrains))

        from core.utils.physics import raycast

        # build & add all given terrains
        for i, terrain_builder in enumerate(self.terrain_builders):
            terrain_spawn_position = terrain_positions[i]

            assert terrain_builder.size == terrains_size, "All terrains must have the same size"

            terrain = terrain_builder.build_from_self(stage, terrain_spawn_position)

            terrain_paths.append(terrain.path)

            # propagate physics changes
            self.world.reset()

            # from the raycast, we can get the desired position of the agent to avoid clipping with the terrain
            raycast_height = 5
            max_ray_dist = 0
            num_rays = 9
            rays_side = math.isqrt(num_rays)
            ray_separation = 0.1

            for j in range(num_rays):
                # we also want to cover a grid of rays on the xy-plane
                start_x = -ray_separation * (rays_side / 2)
                start_y = -ray_separation * (rays_side / 2)
                ray_x = ray_separation * (j % rays_side) + start_x
                ray_y = ray_separation * (j // rays_side) + start_y

                _, _, dist = raycast(
                    [terrain_spawn_position[0] + ray_x, terrain_spawn_position[1] + ray_y, raycast_height],
                    [0, 0, -1],
                    max_distance=10
                )

                max_ray_dist += max(dist, max_ray_dist)

            # we want all agents to be evenly split across all terrains
            agent_batch_start = i * agent_batch_qty
            agent_batch_end = i * agent_batch_qty + agent_batch_qty

            self.agent_positions[agent_batch_start:agent_batch_end, :] = torch.tensor(
                # TODO: make it dependent on the agent's contact point
                [terrain_spawn_position[0], terrain_spawn_position[1], raycast_height - max_ray_dist + 0.115]
            )

        # in some cases, ceil will give us more positions than we need
        if len(self.agent_positions) > self.num_envs:
            self.agent_positions = self.agent_positions[: self.num_envs]

        # clone the agent
        cloner = Cloner()
        cloner.define_base_env("/World/envs")
        base_agent_path = "/World/envs/e_0"
        define_prim(base_agent_path)

        self.agent.construct(base_agent_path, self.world)

        agent_paths = cloner.generate_paths("/World/envs/e", self.num_envs)

        cloner.filter_collisions(
            physicsscene_path="/physicsScene",
            collision_root_path="/collisionGroups",
            prim_paths=agent_paths,
            global_paths=["/World/groundPlane"] + terrain_paths,
        )
        cloner.clone(
            source_prim_path=base_agent_path,
            prim_paths=agent_paths,
            positions=self.agent_positions,
        )

        self.twip_art_view = ArticulationView(
            prim_paths_expr="/World/envs/e.*/twip/body",
            name="twip_art_view",
        )
        self.world.scene.add(self.twip_art_view)

        self.world.reset()

        self.imu = IMU(
            {
                "prim_path": "/World/envs/e.*/twip/body",
                "history_length": 0,
                "update_period": 0,
                "offset": {"pos": (0, 0, 0), "rot": (1.0, 0.0, 0.0, 0.0)},
            }
        )

        return base_agent_path

    def step(self, actions: torch.Tensor, render: bool) -> torch.Tensor:
        # apply actions to the cloned agents
        self._apply_actions(actions)

        # From IsaacLab (SimulationContext)
        # need to do one step to refresh the app
        # reason: physics has to parse the scene again and inform other extensions like hydra-delegate.
        # without this the app becomes unresponsive. If render is True, the world updates the app automatically.
        if not render:
            self.world.app.update()

        self.world.step(render=render)

        # get observations from the cloned agents
        self.imu.update(self.world.get_physics_dt())
        obs = self._gather_imus_frame()

        return obs

    def reset(
        self,
        indices: torch.LongTensor | None = None,
    ) -> None:
        assert indices is None or indices.ndim == 1, "Indices must be a 1D tensor"

        # we assume it's a full reset
        if indices is None:
            print("FULL RESET")

            self.world.reset()  # reset the world too, because we're doing a full reset

            indices = torch.arange(self.num_envs)

        num_to_reset = len(indices)

        self.twip_art_view.set_joint_velocity_targets(
            torch.zeros(num_to_reset, 2), indices=indices
        )
        # using set_velocities instead of individual methods (lin & ang),
        # because it's the only method supported in the GPU pipeline
        self.twip_art_view.set_velocities(torch.zeros(num_to_reset, 6), indices=indices)
        self.twip_art_view.set_joint_positions(
            torch.zeros(num_to_reset, 2), indices=indices
        )
        self.twip_art_view.set_joint_efforts(
            torch.zeros(num_to_reset, 2), indices=indices
        )

        # orientations need to have the quaternion in WXYZ format, and 1 as the first element, the rest being zeros
        orientations = torch.tile(torch.tensor([1.0, 0, 0, 0]), (num_to_reset, 1))

        # ensure that we're on the same device (since we don't know which one in advance)
        if self.agent_positions.device != indices.device:
            self.agent_positions = self.agent_positions.to(indices.device)

        translations = self.agent_positions[indices]

        self.twip_art_view.set_local_poses(
            translations=translations,
            orientations=orientations,
            indices=indices,
        )

        return

    def _apply_actions(self, actions: torch.Tensor) -> None:
        self.twip_art_view.set_joint_efforts(actions)

    def _gather_imus_frame(self) -> torch.Tensor:
        imu_data = self.imu.data
        return torch.cat(
            (imu_data.lin_acc_b, imu_data.ang_vel_b, imu_data.quat_w), dim=1
        )

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
