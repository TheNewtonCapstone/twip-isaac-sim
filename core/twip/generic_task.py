import gym
import numpy as np
import torch
from typing import Dict, Any, Tuple, Callable

from core.base.base_task import BaseTask
from core.base.base_env import BaseEnv
from core.base.base_agent import BaseAgent


# should be called BalancingTwipTask
class GenericTask(BaseTask):
    def __init__(
        self,
        training_env_factory: Callable[..., BaseEnv],
        playing_env_factory: Callable[..., BaseEnv],
        agent_factory: Callable[..., BaseAgent],
    ):
        super().__init__(training_env_factory, playing_env_factory, agent_factory)

    def load_config(
        self,
        headless: bool,
        device: str,
        num_envs: int,
        playing: bool,
        config: Dict[str, Any],
        domain_rand: bool = False,
    ) -> None:
        config["device"] = device
        config["headless"] = headless

        config["num_envs"] = num_envs
        config["max_episode_length"] = 512

        config["num_agents"] = 1
        config["num_observations"] = 4
        config["num_actions"] = 2
        config["num_states"] = 0

        config["observation_space"] = gym.spaces.Box(
            low=np.array(
                [
                    -np.pi,
                    -np.Inf,
                    -1.0,
                    -1.0,
                ]
            ),
            high=np.array(
                [
                    np.pi,
                    np.Inf,
                    1.0,
                    1.0,
                ]
            ),
        )
        config["action_space"] = gym.spaces.Box(
            np.ones(config["num_actions"]) * -1.0,
            np.ones(config["num_actions"]) * 1.0,
        )
        config["state_space"] = gym.spaces.Box(
            np.ones(config["num_states"]) * -np.Inf,
            np.ones(config["num_states"]) * np.Inf,
        )

        # task-specific config
        config["domain_randomization"] = domain_rand
        self.frame_idx = 0

        print(f"{self.__class__.__name__} loaded config {config}")

        super().load_config(
            headless=headless,
            device=device,
            num_envs=num_envs,
            playing=playing,
            config=config,
        )

    def construct(self) -> bool:
        self.env = self.playing_env_factory() if self.playing else self.training_env_factory()
        self.agent = self.agent_factory()

        self.env.construct(self.agent)

        return True

    # RL-Games methods (required from IVecEnv)
    def step(
        self, actions: torch.Tensor = None
    ) -> Tuple[
        Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any] | None
    ]:
        # goes through a RL step (includes all the base RL things such as get obs, apply actions, etc.
        # args: actions to apply to the env
        # returns: obs, rewards, resets, info

        if actions is None:
            return (
                {"obs": self.obs_buf},
                self.rewards_buf,
                self.dones_buf,
                None,
            )

        env_info = self.get_env_info()
        self.progress_buf += 1

        # shape of twip_imu_obs: (num_envs, 10)
        # (:, 0:3) -> linear acceleration
        # (:, 3:6) -> angular velocity
        # (:, 6:10) -> quaternion (WXYZ)
        twip_imu_obs = self.env.step(actions, render=not self.headless).to(
            device=self.device
        )  # is on GPU, so all subsequent calculations will be on GPU

        # get the roll angle only
        twip_roll = roll_from_quat(twip_imu_obs[:, 6:10])

        self.obs_buf[:, 0] = twip_roll
        self.obs_buf[:, 1] = twip_imu_obs[:, 5]  # angular velocity on z-axis
        self.obs_buf[:, 2] = actions[:, 0]
        self.obs_buf[:, 3] = actions[:, 1]

        # the smaller the difference between current orientation and stable orientation, the higher the reward
        self.rewards_buf, episode_info = compute_rewards_twip(
            twip_roll, twip_imu_obs[:, 3], twip_imu_obs[:, 5], actions
        )

        env_info["episode"] = episode_info

        # process failures (when falling)
        self.dones_buf = torch.where(torch.abs(twip_roll) > 0.5, True, False)
        self.dones_buf = torch.where(
            self.progress_buf >= self.max_episode_length - 1, True, self.dones_buf
        )

        # creates a new tensor with only the indices of the environments that are done
        resets = self.dones_buf.nonzero(as_tuple=False).flatten()
        if len(resets) > 0:
            self.env.reset(resets)

        # clears the last 2 observations & the progress if the twips are reset
        self.obs_buf[resets, :] = 0.0
        self.progress_buf[resets] = 0

        return (
            {"obs": self.obs_buf},
            self.rewards_buf,
            self.dones_buf,
            env_info,
        )

    def reset(self) -> Dict[str, torch.Tensor]:
        # resets the entire task (i.e. beginning of training/playing)
        # returns: the observations

        super().reset()

        self.env.reset()

        obs = torch.zeros(
            self.num_envs,
            self.num_observations,
            device=self.device,
            dtype=torch.float32,
        )

        return {"obs": obs}


@torch.jit.script
def compute_rewards_twip(
    roll: torch.Tensor,
    ang_vel_x: torch.Tensor,
    ang_vel_z: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # computes the rewards based on the roll, angular velocity and actions
    # args: roll, angular velocity, actions
    # returns: rewards

    # Compute rate of change of roll
    roll_velocity = ang_vel_x

    # Normalized position (assume max distance is 5 units)
    # norm_position = torch.norm(position, dim=-1) / 5.0

    # Base reward for staying upright
    #upright_reward = 1.0 - torch.tanh(2 * torch.abs(roll))

    # Penalties
    roll_penalty = torch.tanh(6 * torch.abs(roll))
    ang_vel_penalty = torch.tanh(2 * torch.abs(ang_vel_z))
    action_penalty = torch.tanh(torch.sum(torch.abs(actions), dim=-1)) * 0.2
    # position_penalty = torch.tanh(norm_position) * 0.2
    #roll_velocity_penalty = torch.tanh(2 * torch.abs(roll_velocity)) * 0.3

    # Compute rewards
    rewards = (
        1.0
        - roll_penalty
        - ang_vel_penalty
        - action_penalty
        # - position_penalty
        # - roll_velocity_penalty
    )

    # Time factor to encourage longer balancing (assuming time_step is in seconds)
    # time_factor = torch.tanh(time_step / 10.0)  # Normalize to [0, 1] over 10 seconds
    # rewards *= 1.0 + time_factor

    # Harsh penalty for falling
    fall_penalty = torch.where(torch.abs(roll) > 0.5, -5.0, 0.0)
    rewards += fall_penalty

    episode = {
        "roll": torch.median(torch.abs(roll)),
        "roll_var": torch.var(torch.abs(roll)),
        "roll_penalty": torch.mean(roll_penalty),
        "ang_vel_z": torch.median(torch.abs(ang_vel_z)),
        "ang_vel_penalty": torch.mean(ang_vel_penalty),
        "action_magnitude": torch.median(torch.sum(torch.abs(actions), dim=-1)),
        "action_penalty": torch.mean(action_penalty),
        # "position": torch.median(norm_position),
        # "position_penalty": torch.mean(position_penalty),
        "roll_velocity": torch.median(torch.abs(roll_velocity)),
        # "roll_velocity_penalty": torch.mean(roll_velocity_penalty),
        # "time_factor": torch.mean(time_factor),
        # "upright_reward": torch.mean(upright_reward),
    }

    return rewards, episode


@torch.jit.script
def roll_from_quat(q: torch.Tensor) -> torch.Tensor:
    # extracts from a quaternion, the roll in rads, expects a 2dim tensor
    # args: quaternion
    # returns: euler angles

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    roll = torch.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    # pitch = torch.arcsin(2 * (w * y - z * x))
    # yaw = torch.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return roll
