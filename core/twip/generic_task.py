from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any, Tuple

from isaacsim import SimulationApp
from rl_games.common.ivecenv import IVecEnv

from core.base.base_task import BaseTask
from core.base.base_env import BaseEnv

from core.twip.twip_agent import WheelDriveType


class GenericTask(BaseTask):
    def __init__(self, _base_env: BaseEnv):
        super().__init__(_base_env)

    def load_config(self, headless=True):
        config = {}
        config["device"] = "cuda:0"
        config["headless"] = headless

        config["num_envs"] = 1

        config["num_agents"] = 1
        config["num_observations"] = 4
        config["num_actions"] = 2
        config["num_states"] = 0

        config["observation_space"] = gym.spaces.Box(
            low=np.array(
                [
                    -np.pi,
                    -np.Inf,
                    -400.0,
                    -400.0,
                ]
            ),
            high=np.array([np.pi, np.Inf, 400.0, 400.0]),
        )
        config["action_space"] = gym.spaces.Box(
            np.ones(config["num_actions"]) * -400.0,
            np.ones(config["num_actions"]) * 400.0,
        )
        config["state_space"] = gym.spaces.Box(
            np.ones(config["num_states"]) * -np.Inf,
            np.ones(config["num_states"]) * np.Inf,
        )

        # task-specific config
        config["domain_randomization"] = {}

        print(f"{self.__class__.__name__} loaded config {config}")

        return super().load_config(config)

    def construct(self, sim_app: SimulationApp) -> bool:
        res = self.base_env.construct(sim_app)

        if not res:
            return False

        self.base_env.prepare()

        return True

    # RL-Games methods (required from IVecEnv)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # goes through a RL step (includes all the base RL things such as get obs, apply actions, etc.
        # args: actions to apply to the env
        # returns: obs, rewards, resets, info

        super().step(actions)

        twip_agent = self.base_env.agent

        twip_agent.set_target_velocity(WheelDriveType.LEFT, actions[0, 0].item())
        twip_agent.set_target_velocity(WheelDriveType.RIGHT, actions[0, 1].item())

        self.base_env.step(_render=not self.headless)

        twip_obs_dict = twip_agent.get_observations()
        twip_roll = self._quaternion_to_euler(twip_obs_dict["orientation"])[0]

        twip_obs = torch.tensor(
            [
                twip_roll,
                twip_obs_dict["ang_vel"][0],
                actions[0, 0].item(),
                actions[0, 1].item(),
            ],
        )
        obs = {
            "obs": torch.tensor(
                twip_obs.unsqueeze(0), device=self.device, dtype=torch.float32
            )
        }

        combined_applied_vel = torch.sum(torch.abs(actions[0, :]))

        # the smaller the difference between current orientation and stable orientation, the higher the reward
        reward = torch.tensor(
            [
                1.0
                - torch.tanh(8 * torch.abs(twip_roll))
                - 0.6 * torch.tanh(2 * torch.abs(twip_obs_dict["ang_vel"][2]))
                - 0.1 * torch.tanh(combined_applied_vel)
            ]
        )

        if torch.abs(twip_roll) > 0.2:
            done = torch.tensor([True])
            reward = torch.tensor([-4.0])
            self.reset()
        else:
            done = torch.tensor([False])

        env_info = self.get_env_info()

        return obs, reward, done, env_info

    def reset(self) -> Dict[str, torch.Tensor]:
        # resets a single environment
        # returns: the observations

        super().reset()

        twip_agent = self.base_env.agent

        self.base_env.prepare()

        # makes sure that the twip is stable before starting
        for _ in range(4):
            self.base_env.step(_render=not self.headless)

        twip_obs_dict = twip_agent.get_observations()
        twip_obs = torch.tensor(
            [
                self._quaternion_to_euler(twip_obs_dict["orientation"])[0],
                twip_obs_dict["ang_vel"][0],
                0.0,
                0.0,
            ],
        )
        obs = {
            "obs": torch.tensor(
                twip_obs.unsqueeze(0), device=self.device, dtype=torch.float32
            )
        }

        return obs

    def _quaternion_to_euler(self, q: torch.Tensor) -> torch.Tensor:
        # converts a quaternion to euler angles
        # args: quaternion
        # returns: euler angles

        w, x, y, z = q[0], q[1], q[2], q[3]

        roll = torch.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = torch.arcsin(2 * (w * y - z * x))
        yaw = torch.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        return torch.tensor(
            [roll, pitch, yaw], device=q.get_device(), dtype=torch.float32
        )
