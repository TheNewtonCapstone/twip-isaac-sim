import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any, Tuple

from rl_games.common.ivecenv import IVecEnv

from core.base.base_env import BaseEnv


class BaseTask(IVecEnv):
    def __init__(self, _base_env: BaseEnv):
        self.config = {}

        self.base_env = _base_env

    def load_config(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config

        self.device: str = self.config["device"]
        self.headless: bool = self.config["headless"]

        self.num_envs: int = self.config["num_envs"]

        self.num_agents: int = self.config["num_agents"]
        self.num_observations: int = self.config["num_observations"]
        self.num_actions: int = self.config["num_actions"]
        self.num_states: int = self.config["num_states"]

        self.observation_space: gym.spaces.Box = self.config["observation_space"]
        self.action_space: gym.spaces.Box = self.config["action_space"]
        self.state_space: gym.spaces.Box = self.config["state_space"]

        # rest of config inside self.config

    def construct(self) -> bool:
        return self.base_env.construct()

    # RL-Games methods (required from IVecEnv)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # goes through a RL step (includes all the base RL things such as get obs, apply actions, etc.
        # args: actions to apply to the env
        # returns: obs, rewards, resets, info

        # print(f"{self.__class__.__name__} step")
        return

    def reset(self) -> Dict[str, torch.Tensor]:
        # resets a single environment
        # returns: the observations

        # print(f"{self.__class__.__name__} reset")
        return

    def seed(self, seed) -> None:
        pass

    def has_action_masks(self) -> bool:
        return False

    def get_number_of_agents(self) -> int:
        # we only support 1 agent in the env for now
        return 1

    def get_env_info(self) -> Dict:
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }

    def set_train_info(self, env_frames, *args, **kwargs):
        pass

    def get_env_state(self):
        return None

    def set_env_state(self, env_state) -> None:
        pass

    def get_observation_space(self) -> gym.spaces:
        return self.observation_space

    def get_action_space(self) -> gym.Space:
        return self.action_space

    def get_state_space(self) -> gym.spaces.Box:
        return self.state_space

    def get_num_envs(self) -> int:
        return self.num_envs

    def get_device(self) -> str:
        return self.device
