import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from tasks.base.vec_task import VecTask


class Twip(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        # parameters
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # plane parameters
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # controller parameters
        self.control_mode = self.cfg["env"]["controlMode"] # "velocity" or "effort"
        if self.control_mode == "velocity":
            self.max_velocity = self.cfg["env"]["maxVelocity"]
            self.min_velocity = self.cfg["env"]["minVelocity"]
        if self.control_mode == "effort":
            self.max_effort = self.cfg["env"]["maxEffort"]
        self.free_dofs = self.cfg["env"]["freeDofs"] # indices of unactuated joints
        self.imu_frame_index = self.cfg["env"]["imuFrame"] # index of link corresponding to IMU frame

        # randomize parameters (on each reset)
        self.randomize_pitch = self.cfg["env"]["randomizePitch"]
        self.randomize_torque = self.cfg["env"]["randomizeTorque"]
        self.randomize_force = self.cfg["env"]["randomizeForce"]
        self.randomize_position = self.cfg["env"]["randomizePosition"]
        self.randomize_velocity = self.cfg["env"]["randomizeVelocity"]

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim) # access joint states
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim) # access link states
        self.body_state = gymtorch.wrap_tensor(body_state_tensor)
        self.body_pos = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self.body_ori = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self.body_linvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[...,7:10]
        self.body_angvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[...,10:13]

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim) # access actor root states
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()

        self.dt = self.cfg["sim"]["dt"]

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        assert "asset" in self.cfg["env"]
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),asset_root)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()

        # Set default control mode for joints
        if self.control_mode == "velocity":
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
        if self.control_mode == "effort":
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

        # Load assets
        twip_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)      
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.05, self.up_axis_idx))

        # Print names of links and joints
        self.num_bodies = self.gym.get_asset_rigid_body_count(twip_asset)
        body_names = [self.gym.get_asset_rigid_body_name(twip_asset, i) for i in range(self.num_bodies)]
        self.num_dof = self.gym.get_asset_dof_count(twip_asset)
        dof_names = [self.gym.get_asset_dof_name(twip_asset, i) for i in range(self.num_dof)]
        print("Bodies: ", body_names)
        print("Dofs: ", dof_names)

        if self.control_mode == "velocity":
            motor_vels = [self.max_velocity for i in range(self.num_dof-len(self.free_dofs))]
            self.motor_out = to_torch(motor_vels, device=self.device)
        if self.control_mode == "effort":
            motor_efforts = [self.max_effort for i in range(self.num_dof-len(self.free_dofs))]
            self.motor_out = to_torch(motor_efforts, device=self.device)
        
        self.twip_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            robot_handle = self.gym.create_actor(env_ptr, twip_asset, start_pose, "twip", i, 1, 0)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, robot_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.twip_handles.append(robot_handle)

            # set joint properties
            dof_prop = self.gym.get_actor_dof_properties(env_ptr, robot_handle)
            if self.control_mode == "velocity":
                dof_prop["driveMode"][:] = gymapi.DOF_MODE_VEL
                dof_prop["stiffness"].fill(0.0)
            if self.control_mode == "effort":
                dof_prop["driveMode"][:] = gymapi.DOF_MODE_EFFORT
                dof_prop["stiffness"].fill(0.0)
            if i==1:
                print("Damping: ", dof_prop["damping"]) # Make sure this is non-zero in velocity control mode
                print("Friction: ", dof_prop["friction"])
                print("Stiffness: ", dof_prop["stiffness"])

            # set up unactuated joints
            dof_prop["driveMode"][self.free_dofs] = gymapi.DOF_MODE_NONE
            dof_prop["stiffness"][self.free_dofs] = 0.0

            # set up joint limits
            for j in range(self.num_dof):
                if dof_prop['lower'][j] > dof_prop['upper'][j]:
                    self.dof_limits_lower.append(dof_prop['upper'][j])
                    self.dof_limits_upper.append(dof_prop['lower'][j])
                else:
                    self.dof_limits_lower.append(dof_prop['lower'][j])
                    self.dof_limits_upper.append(dof_prop['upper'][j])
            self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_prop)
        
    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_twip_reward(
            self.obs_buf, self.reset_buf, self.progress_buf, self.max_episode_length
        )
        # print("Reset: ", torch.sum(self.reset_buf))

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        
        #self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # NaN Check if you run into errors
        # print("NaN: ", torch.sum(torch.isnan(self.initial_root_states)))

        # Get pitch as first input
        quat = self.body_ori[env_ids,self.imu_frame_index].squeeze()
        ori_x = quat[:, 0]
        ori_y = quat[:, 1]
        ori_z = quat[:, 2]
        ori_w = quat[:, 3]
        x = torch.atan2(2*(ori_w*ori_x+ori_y*ori_z),1-2*(ori_x**2+ori_y**2)) #pitch (0 when vertical)
        self.obs_buf[env_ids,0] = x

        # Get last velocity action as second input
        self.obs_buf[env_ids,1] = self.actions[env_ids,0]

        # Get wheel position as third input (currently unused)
        dof_idx = list(range(self.num_dof))
        for i in self.free_dofs:
            dof_idx.remove(i)
        #self.obs_buf[env_ids,2] = torch.tanh(torch.mean(self.dof_pos[env_ids][:,dof_idx],dim=1))

        return self.obs_buf

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # Apply random pitch at start (called roll here because I adapted the code from Wikipedia)
        if self.randomize_pitch:
            roll_dir = (torch.randint(0,2,(len(env_ids), 1),dtype=torch.float32, device=self.device)*2-1)
            pitch = torch_rand_float(0, 0, (len(env_ids), 1), device=self.device)
            roll = torch_rand_float(0., 0.23, (len(env_ids), 1), device=self.device) * roll_dir
            yaw = torch_rand_float(0, 0, (len(env_ids), 1), device=self.device)

            cy = torch.cos(yaw*0.5)
            sy = torch.sin(yaw*0.5)
            cp = torch.cos(pitch*0.5)
            sp = torch.sin(pitch*0.5)
            cr = torch.cos(roll*0.5)
            sr = torch.sin(roll*0.5)

            xs = sr * cp * cy - cr * sp * sy
            ys = cr * sp * cy + sr * cp * sy
            zs = cr * cp * sy - sr * sp * cy
            ws = cr * cp * cy + sr * sp * sy
            oris = torch.cat([xs,ys,zs,ws], dim=1)
            oris = oris/torch.norm(oris,p='fro',dim=1).unsqueeze(dim=1)
            self.initial_root_states[env_ids,3:7] = oris

        # Set initial DOF position for wheels
        if self.randomize_position:
            pos_dir = (torch.randint(0,2,(len(env_ids), 1),dtype=torch.float32, device=self.device)*2-1).repeat(1,self.num_dof)
            positions = torch_rand_float(0., 10., (len(env_ids), 1), device=self.device).repeat(1,self.num_dof) * pos_dir
            positions[:, self.free_dofs] = 0
            self.dof_pos[env_ids] = positions
        else:
            positions[:, :] = 0
        
        # Set initial DOF velocity for wheels
        if self.randomize_velocity:
            vel_dir = (torch.randint(0,2,(len(env_ids), 1),dtype=torch.float32, device=self.device)*2-1).repeat(1,self.num_dof)
            velocities = torch_rand_float(0., 1., (len(env_ids), 1), device=self.device).repeat(1,self.num_dof) * vel_dir
            velocities[:, self.free_dofs] = 0
            self.dof_vel[env_ids] = velocities*self.max_velocity
        else:
            velocities[:, :] = 0
            self.dof_vel[env_ids] = velocities*self.max_velocity
        self.actions[env_ids,0] = velocities[:,0] # set last action since we use this as input

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # Apply random torque to body link
        if self.randomize_torque:
            torque_dir = (torch.randint(0,2,(len(env_ids),),dtype=torch.float32, device=self.device)*2-1)
            random_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float32)
            random_torques[env_ids,0,0] = torch_rand_float(0, 40., (len(env_ids),1), device=self.device)[:,0] * torque_dir
            random_torques_gym = gymtorch.unwrap_tensor(random_torques)
        else:
            random_torques_gym = None

        # Apply random force to body link
        if self.randomize_force:
            force_dir = (torch.randint(0,2,(len(env_ids),),dtype=torch.float32, device=self.device)*2-1)
            random_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float32)
            random_forces[env_ids,0,1] = torch_rand_float(0., 300., (len(env_ids),1), device=self.device)[:,0] * force_dir
            random_forces_gym = gymtorch.unwrap_tensor(random_forces)
        else:
            random_forces_gym = None

        self.gym.apply_rigid_body_force_tensors(self.sim, random_forces_gym, random_torques_gym, gymapi.ENV_SPACE)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        #print(self.actions)
        #self.actions[:] = 1
        dof_idx = list(range(self.num_dof))
        for i in self.free_dofs:
            dof_idx.remove(i)
        if self.control_mode == "velocity":
            vels = torch.zeros((self.actions.shape[0],self.num_dof)).to(self.device)
            #vels = torch.ones_like(self.actions) * self.motor_out
            vels[:,dof_idx] = self.actions[:,0].reshape(-1,1) * self.motor_out
            vels[torch.abs(vels)<=self.min_velocity] = 0. #if below min velocity set velocity to 0
            vel_tensor = gymtorch.unwrap_tensor(vels)
            self.gym.set_dof_velocity_target_tensor(self.sim, vel_tensor)
        if self.control_mode == "effort":
            forces = torch.zeros((self.actions.shape[0],self.num_dof)).to(self.device)
            forces[:, dof_idx] = self.actions * self.motor_out
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            points = []
            colors = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                # Add visualizations here
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])

            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_twip_reward(obs_buf, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    pole_angle = torch.abs(obs_buf[:,0])
    last_vel = torch.abs(obs_buf[:,1])
   
    # reward combines angle deviation from vertical and last action taken (velocity of wheels)
    #reward = (1.0 - 4.*pole_angle) * (1.0 - 0.25*last_vel)  #* (1.0 - 1. * pos)
    reward = 1.0 - torch.tanh(8*pole_angle) - 0.05 * torch.tanh(2*last_vel) 

    # adjust reward for reset agents
    reward = torch.where(torch.abs(pole_angle) > 0.5, torch.ones_like(reward) * -2.0, reward)

    # reset if angle is too far from vertical (TWIP has fallen over)
    reset = torch.where(torch.abs(pole_angle) > 0.5, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset
