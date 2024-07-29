import torch
import numpy as np

from core.base.base_agent import BaseAgent
from core.base.base_env import BaseEnv
from core.sensors.imu.imu import IMU
from core.twip.twip_agent import TwipAgent




# TODO: should be called GenericTwipEnv
class GenericEnv(BaseEnv):
    def __init__(self, world_settings, num_envs, randomization_settings):
        super().__init__(
            world_settings,
            num_envs=num_envs,
            randomization_settings=randomization_settings,
        )

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

        self.imu = IMU(
            {
                "prim_path": "/World/envs/e.*/twip/body",
                "history_length": 0,
                "update_period": 0,
                "offset": {"pos": (0, 0, 0), "rot": (1.0, 0.0, 0.0, 0.0)},
            }
        )

        # Setting up domain randomization if enabled
        self.randomize = self.randomization_settings.get("randomize", False)
        self.randomization_settings = self.randomization_settings.get(
            "randomization_params", {}
        )

        if self.randomize:
            self.domain_randomization(randomization_params=self.randomization_settings)

        self.frame_idx = 0

        return self.base_agent_path

    def step(self, actions: torch.Tensor, render: bool) -> torch.Tensor:
        # apply actions to the cloned agents
        self._apply_actions(actions)

        # From IsaacLab (SimulationContext)
        # need to do one step to refresh the app
        # reason: physics has to parse the scene again and inform other extensions like hydra-delegate.
        # without this the app becomes unresponsive. If render is True, the world updates the app automatically.
        if not render:
            self.world.app.update()

        # domain randomization
        if self.randomize:
            self.step_randomization()

        # print(self.twip_art_view.get_joint_velocities())

        self.world.step(render=render)

        # get observations from the cloned agents
        self.imu.update(self.world.get_physics_dt())
        obs = self._gather_imus_frame()

        # if domain randomization is enabled, we need to update the frame index
        self.frame_idx += 1

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
        imu_data = self.imu.data
        return torch.cat(
            (imu_data.lin_acc_b, imu_data.ang_vel_b, imu_data.quat_w), dim=1
        )

    def domain_randomization(self, randomization_params) -> None:
        if not randomization_params:
            print("No domain randomization parameters provided.")
            return

        self.frequency = randomization_params["frequency"]
        self.domain_params = randomization_params["twip"]

        import omni.replicator.isaac as dr
        import omni.replicator.core as rep

        self.dr = dr
        self.rep = rep
        self.num_dof = self.twip_art_view.num_dof

        args = self.map_config_to_function_args(self.domain_params, rep)

        # domain randomization
        self.dr.physics_view.register_simulation_context(self.world)
        self.dr.physics_view.register_articulation_view(self.twip_art_view)

        with self.dr.trigger.on_rl_frame(num_envs=self.num_envs):
            with self.dr.gate.on_interval(interval=self.frequency):      
                dr.physics_view.randomize_articulation_view(
                    view_name=self.twip_art_view.name,
                    operation="direct",
                    **args,
                )
            with self.dr.gate.on_env_reset():
                dr.physics_view.randomize_articulation_view(
                    view_name=self.twip_art_view.name,
                    operation="additive",
                    **args,
                )

        rep.orchestrator.run()
        self.frame_idx = 0

    def step_randomization(self) -> None:
        reset_inds = []
        if self.frame_idx % 200 == 0:
            # triggers reset every 200 steps
            reset_inds = np.arange(self.num_envs)
            self.dr.physics_view.step_randomization(reset_inds)

    def map_config_to_function_args(self, config, rep):
        
        args = {}

        if "articulation_view_properties" not in config and "dof_properties" not in config:
            raise ValueError("Invalid randomization config: missing properties")
        
        self.articulation_view_properties = config["articulation_view_properties"]
        self.dof_properties = config["dof_properties"]


        for prop in self.articulation_view_properties:
            prop_range = self.get_randomization_range(self.articulation_view_properties[prop]["range"])

            print(prop)
            print(tuple(prop_range[0]))
            print(tuple(prop_range[1]))
            
            args[prop] = rep.distribution.normal(
                tuple(prop_range[0]), tuple(prop_range[1])
            )

        for prop in self.dof_properties:
            prop_range = self.get_randomization_range(self.dof_properties[prop]["range"])

            print([prop_range[0]] * self.num_dof)
            print([prop_range[1]] * self.num_dof)
            
            args[prop] = rep.distribution.uniform(
                tuple([prop_range[0]] * self.num_dof), tuple([prop_range[1]] * self.num_dof)
            )

        return args

    def get_randomization_range(self, prop_range):
        from_x = []
        to_y = []
        if isinstance(prop_range[0], list):
            for item in prop_range:
                from_x.append(item[0])
                to_y.append(item[1])
        else:
            from_x = [prop_range[0]]
            to_y = [prop_range[1]]

        return from_x, to_y
