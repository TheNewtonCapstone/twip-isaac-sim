from abc import ABC, abstractmethod, property
import gymnasium as gm
import numpy as np
import torch
from typing import Dict, Any, Tuple

from rl_games.common.ivecenv import IVecEnv

from core.base.base_env import BaseEnv


class BaseTask(IVecEnv):
    def __init__(self, _base_env: BaseEnv):
        self.config = {}

        self.base_env = _base_env
        pass

    def load_config(self, config: Dict) -> None:
        self.config["device"] = "cuda:0"
        self.config["graphics_device_id"] = 0
        self.config["headless"] = False

        self.config["num_env"] = 1024

        self.config["num_agents"] = 1
        self.config["num_observations"] = 2
        self.config["num_actions"] = 1  # not sure why it's 1
        self.config["num_states"] = 0

        self.config["observation_space"] = gm.spaces.Box(
            np.ones(self.config["num_observations"]) * -np.Inf,
            np.ones(self.config["num_observations"]) * np.Inf,
        )
        self.config["action_space"] = gm.spaces.Box(
            np.ones(self.config["num_actions"]) * -1.0,
            np.ones(self.config["num_actions"]) * 1.0,
        )  # not sure why it's different from the other spaces
        self.config["state_space"] = gm.spaces.Box(
            np.ones(self.config["num_states"]) * -np.Inf,
            np.ones(self.config["num_states"]) * np.Inf,
        )  # still not sure what a state is

        # task-specific config
        self.config["domain_randomization"] = {}
        self.config["extern_actor_params"] = {}  # no se
        for i in range(self.config["num_env"]):
            self.config["extern_actor_params"][i] = None  # tampoco se

        return self.config

    def construct(self) -> bool:
        return self.base_env.construct()

    # RL-Games methods (required from IVecEnv)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # goes through a RL step (includes all the base RL things such as get obs, apply actions, etc.
        # args: actions to apply to the env
        # returns: obs, rewards, resets, info
        pass

    def reset(self) -> Dict[str, torch.Tensor]:
        # resets a single environment
        # returns: the observations
        pass

    def seed(self, seed) -> None:
        pass

    def has_action_masks(self) -> bool:
        return False

    def get_number_of_agents(self) -> int:
        # we only support 1 agent in the env for now
        return 1

    def get_env_info(self) -> Dict:
        return {
            "observation_space": self.config["observation_space"],
            "action_space": self.config["action_space"],
            "state_space": self.config["state_space"],
        }

    def set_train_info(self, env_frames, *args, **kwargs):
        pass

    def get_env_state(self):
        return None

    def set_env_state(self, env_state) -> None:
        pass

    def get_observation_space(self) -> gm.spaces:
        return self.config["observation_space"]

    def get_action_space(self) -> gm.Space:
        return self.config["action_space"]

    def get_state_space(self) -> gm.spaces.Box:
        return self.config["state_space"]

    def get_num_envs(self) -> int:
        return self.config["num_envs"]

    def get_device(self) -> str:
        return self.config["device"]


base_task = BaseTask()
print(base_task.load_config())
