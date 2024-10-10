from typing import Dict, Any, Tuple, Callable

import gym
import numpy as np
import torch
from core.base.base_agent import BaseAgent
from core.base.base_env import BaseEnv
from core.base.base_task import BaseTask
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs


class GenericCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=2)

    def _on_step(self) -> bool:
        task: GenericTask = self.training_env

        self.logger.record("rewards/mean", task.rewards_buf.mean().item())
        self.logger.record("rewards/median", torch.median(task.rewards_buf).item())

        self.logger.record("metrics/roll_mean", task.obs_buf[:, 0].mean().item())
        self.logger.record("metrics/ang_vel_mean", task.obs_buf[:, 1].mean().item())

        combined_actions = torch.sum(torch.abs(task.actions_buf), dim=-1)
        self.logger.record(
            "metrics/action_magnitude_mean", combined_actions.mean().item()
        )

        return True


# should be called BalancingTwipTask
class GenericTask(BaseTask):
    def __init__(
        self,
        headless: bool,
        device: str,
        num_envs: int,
        playing: bool,
        max_episode_length: int,
        domain_randomization: Dict[str, Any],
        training_env_factory: Callable[..., BaseEnv],
        playing_env_factory: Callable[..., BaseEnv],
        agent_factory: Callable[..., BaseAgent],
    ):
        observation_space = gym.spaces.Box(
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

        num_actions = 2
        action_space = gym.spaces.Box(
            low=np.ones(num_actions) * -1.0,
            high=np.ones(num_actions) * 1.0,
        )

        reward_space = gym.spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
        )

        super().__init__(
            headless=headless,
            device=device,
            num_envs=num_envs,
            playing=playing,
            max_episode_length=max_episode_length,
            observation_space=observation_space,
            action_space=action_space,
            reward_space=reward_space,
            training_env_factory=training_env_factory,
            playing_env_factory=playing_env_factory,
            agent_factory=agent_factory,
        )

    def construct(self) -> bool:
        self.env = (
            self.playing_env_factory() if self.playing else self.training_env_factory()
        )
        self.agent = self.agent_factory()

        self.env.construct(self.agent)

        return True

    # Gymnasium methods (required from VecEnv)

    def reset(self) -> VecEnvObs:
        # resets the entire task (i.e. beginning of training/playing)
        # returns: the observations

        super().reset()

        self.env.reset()

        # noinspection PyArgumentList
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations), dtype=torch.float32
        )

        return self.obs_buf.numpy()

    def step_wait(self) -> VecEnvStepReturn:
        if self.actions_buf is None:
            return (
                self.obs_buf.numpy(),
                self.rewards_buf.numpy(),
                self.dones_buf.numpy(),
                [],
            )

        self.progress_buf += 1

        import random

        # shape of twip_imu_obs: (num_envs, 10)
        # (:, 0:3) -> linear acceleration
        # (:, 3:6) -> angular velocity
        # (:, 6:10) -> quaternion (WXYZ)
        twip_imu_obs = self.env.step(
            actions_to_torque(self.actions_buf * random.gauss(1.0, 0.065)),
            render=not self.headless,
        )

        # get the roll angle only
        twip_roll = roll_from_quat(twip_imu_obs[:, 6:10])

        self.obs_buf[:, 0] = twip_roll * random.gauss(1.0, 0.065)
        self.obs_buf[:, 1] = twip_imu_obs[:, 5] * random.gauss(
            1.0, 0.065
        )  # angular velocity on z-axis
        self.obs_buf[:, 2] = self.actions_buf[:, 0]
        self.obs_buf[:, 3] = self.actions_buf[:, 1]

        # the smaller the difference between current orientation and stable orientation, the higher the reward
        self.rewards_buf, episode_info = compute_rewards_twip(
            twip_roll, twip_imu_obs[:, 3], twip_imu_obs[:, 5], self.actions_buf
        )

        # process failures (when falling)
        self.dones_buf = torch.where(torch.abs(twip_roll) > 0.5, True, False)
        self.dones_buf = torch.where(
            self.progress_buf >= self.max_episode_length - 1, True, self.dones_buf
        )

        # creates a new tensor with only the indices of the environments that are done
        resets = self.dones_buf.nonzero(as_tuple=False).flatten()
        if len(resets) > 0:
            self.env.reset(resets)

        # clears the last 2 observations & the progress if any twip is reset
        self.obs_buf[resets, :] = 0.0
        self.progress_buf[resets] = 0

        return (
            self.obs_buf.numpy(),
            self.rewards_buf.numpy(),
            self.dones_buf.numpy(),
            [{} for _ in range(self.num_envs)],
        )


@torch.jit.script
def pwm_percent_to_torque(pwm_percent: torch.Tensor) -> torch.Tensor:
    # if any pwm_percent is less than 0.375, return 0
    # otherwise, return the torque
    # this follows empirical data
    return torch.where(
        pwm_percent <= 0.375, 0.0, 0.653 + 0.507 * torch.log(pwm_percent)
    )


@torch.jit.script
def actions_to_torque(actions: torch.Tensor) -> torch.Tensor:
    return torch.sign(actions) * pwm_percent_to_torque(torch.abs(actions))


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
    # upright_reward = 1.0 - torch.tanh(2 * torch.abs(roll))

    # Penalties
    roll_penalty = torch.tanh(6 * torch.abs(roll))
    ang_vel_penalty = torch.tanh(2 * torch.abs(ang_vel_z))
    action_penalty = (
        torch.tanh(torch.sum(torch.abs(actions_to_torque(actions)), dim=-1)) * 0.2
    )
    # position_penalty = torch.tanh(norm_position) * 0.2
    # roll_velocity_penalty = torch.tanh(2 * torch.abs(roll_velocity)) * 0.3

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
        "action_magnitude": torch.median(
            torch.sum(torch.abs(actions_to_torque(actions)), dim=-1)
        ),
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

    roll = torch.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    # pitch = torch.arcsin(2 * (w * y - z * x))
    # yaw = torch.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return roll
