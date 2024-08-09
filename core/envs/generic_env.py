import torch
import numpy as np

from core.base.base_agent import BaseAgent
from core.base.base_env import BaseEnv
from core.sensors.imu.imu import IMU
from core.twip.twip_agent import TwipAgent
from core.domain_randomizer.domain_randomizer import DomainRandomizer

# TODO: should be called GenericTwipEnv
class GenericEnv(BaseEnv):
    def __init__(self, world_settings, num_envs, randomization_settings):
        super().__init__(
            world_settings,
            num_envs=num_envs,
            randomization_settings=randomization_settings,
        )
        self.randomize = randomization_settings.get("randomize", False)
        self.randomization_params = randomization_settings.get("randomization_params", {})

    def construct(self, agent: BaseAgent) -> bool:
        super().construct(agent)

        import omni.isaac.core
        from omni.isaac.cloner import GridCloner
        from omni.isaac.core.articulations import ArticulationView
        from omni.isaac.core.utils.prims import define_prim

        self.world.scene.add_default_ground_plane()

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

        self.imu = IMU(
            {
                "prim_path": "/World/envs/e.*/twip/body",
                "history_length": 0,
                "update_period": 0,
                "offset": {"pos": (0, 0, 0), "rot": (1.0, 0.0, 0.0, 0.0)},
            }
        )

        if self.randomize:
            self.domain_randomizer = DomainRandomizer(self.world, self.twip_art_view, self.randomization_params)
            self.domain_randomizer.apply_randomization()
        self.frame_idx = 0

        return self.base_agent_path

    def step(self, actions: torch.Tensor, render: bool) -> torch.Tensor:
        self._apply_actions(actions)

        if not render:
            self.world.app.update()

        if self.randomize:
            self.domain_randomizer.step_randomization()

        self.world.step(render=render)

        self.imu.update(self.world.get_physics_dt())
        obs = self._gather_imus_frame()

        self.frame_idx += 1

        return obs

    def reset(self, indices: torch.LongTensor | None = None) -> None:
        assert indices is None or indices.ndim == 1, "Indices must be a 1D tensor"

        if indices is None:
            self.world.reset()
            indices = torch.arange(self.num_envs)

        num_to_reset = len(indices)

        self.twip_art_view.set_joint_velocity_targets(
            torch.zeros(num_to_reset, 2), indices=indices
        )
        self.twip_art_view.set_velocities(torch.zeros(num_to_reset, 6), indices=indices)
        self.twip_art_view.set_joint_positions(
            torch.zeros(num_to_reset, 2), indices=indices
        )
        self.twip_art_view.set_joint_efforts(
            torch.zeros(num_to_reset, 2), indices=indices
        )

        orientations = torch.tile(torch.tensor([1.0, 0, 0, 0]), (num_to_reset, 1))

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

    def _apply_actions(self, actions: torch.Tensor) -> None:
        self.twip_art_view.set_joint_efforts(actions)

    def _gather_imus_frame(self) -> torch.Tensor:
        imu_data = self.imu.data
        return torch.cat(
            (imu_data.lin_acc_b, imu_data.ang_vel_b, imu_data.quat_w), dim=1
        )
