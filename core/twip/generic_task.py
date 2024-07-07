import gymnasium as gym
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
        env_factory: Callable[..., BaseEnv],
        agent_factory: Callable[..., BaseAgent],
    ):
        super().__init__(env_factory, agent_factory)

    def load_config(
        self,
        headless: bool,
        device: str,
        num_envs: int,
        config: Dict[Any, str] = {},
    ) -> None:
        config["device"] = device
        config["headless"] = headless

        config["num_envs"] = num_envs

        config["num_agents"] = 1
        config["num_observations"] = 4
        config["num_actions"] = 2
        config["num_states"] = 0

        config["observation_space"] = gym.spaces.Box(
            low=np.array(
                [
                    -np.pi,
                    -np.Inf,
                    -100.0,
                    -100.0,
                ]
            ),
            high=np.array(
                [
                    np.pi,
                    np.Inf,
                    100.0,
                    100.0,
                ]
            ),
        )
        config["action_space"] = gym.spaces.Box(
            np.ones(config["num_actions"]) * -100.0,
            np.ones(config["num_actions"]) * 100.0,
        )
        config["state_space"] = gym.spaces.Box(
            np.ones(config["num_states"]) * -np.Inf,
            np.ones(config["num_states"]) * np.Inf,
        )

        # task-specific config
        config["domain_randomization"] = {}

        print(f"{self.__class__.__name__} loaded config {config}")

        super().load_config(
            headless=headless,
            device=device,
            num_envs=num_envs,
            config=config,
        )

    def construct(self) -> bool:
        self.env = self.env_factory()
        self.agent = self.agent_factory()

        self.env.construct(self.agent)

        return True

    # RL-Games methods (required from IVecEnv)
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # goes through a RL step (includes all the base RL things such as get obs, apply actions, etc.
        # args: actions to apply to the env
        # returns: obs, rewards, resets, info

        super().step(actions)

        obs = torch.zeros(
            self.num_envs,
            self.num_observations,
            device=self.device,
            dtype=torch.float32,
        )
        rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        env_info = self.get_env_info()

        if actions is None:
            return (
                {"obs": obs},
                rewards,
                dones,
                env_info,
            )

        # shape of twip_imu_obs: (num_envs, 10)
        # (:, 0:3) -> linear acceleration
        # (:, 3:6) -> angular velocity
        # (:, 6:10) -> quaternion (WXYZ)
        twip_imu_obs = self.env.step(actions, render=not self.headless).to(
            device=self.device
        )  # is on GPU, so all subsequent calculations will be on GPU

        # get the roll angle only
        twip_roll = roll_from_quat(twip_imu_obs[:, 6:10])
        obs[:, 0] = twip_roll
        obs[:, 1] = twip_imu_obs[:, 5]  # angular velocity on z-axis
        obs[:, 2] = actions[:, 0]
        obs[:, 3] = actions[:, 1]

        # the smaller the difference between current orientation and stable orientation, the higher the reward
        rewards = compute_rewards_twip(twip_roll, twip_imu_obs[:, 3], actions)

        # process failures (when falling)
        dones = torch.where(torch.abs(twip_roll) > 0.2, True, False)

        # creates a new tensor with only the indices of the environments that are done
        resets = dones.nonzero(as_tuple=False).flatten()
        if len(resets) > 0:
            self.env.reset(resets)

        return (
            {"obs": obs},
            rewards,
            dones,
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
    ang_vel_z: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    # computes the rewards based on the roll, angular velocity and actions
    # args: roll, angular velocity, actions
    # returns: rewards

    combined_applied_vel = torch.abs(actions[:, 0]) + torch.abs(actions[:, 1])

    # the smaller the difference between current orientation and stable orientation, the higher the reward
    rewards = (
        1.0
        - torch.tanh(8 * torch.abs(roll))
        - 0.1 * torch.tanh(2 * torch.abs(ang_vel_z))
        - 0.1 * torch.tanh(combined_applied_vel)
    )

    # penalize for falling
    rewards = torch.where(torch.abs(roll) > 0.2, -4.0, rewards)

    return rewards


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
