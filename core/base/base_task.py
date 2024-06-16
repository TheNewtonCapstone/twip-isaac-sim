from abc import ABC, abstractmethod, property
import gymnasium as gm
import numpy as np
import torch
from typing import Dict, Any, Tuple
from core.base.base_env import BaseEnv


class BaseTask(object):
    def __init__(self, _base_env: BaseEnv):
        self.config = {}

        self.base_env = _base_env
        pass

    # @abstractmethod
    def load_config(self):
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

    # @abstractmethod
    def allocate_buffers(self) -> None:
        # allocates memory on the GPU
        self.obs_buf = torch.zeros(
            (self.config["num_envs"], self.config["num_observations"]),
            device=self.config["device"],
            dtype=torch.float,
        )
        self.states_buf = torch.zeros(
            (self.config["num_envs"], self.config["num_states"]),
            device=self.config["device"],
            dtype=torch.float,
        )
        self.rew_buf = torch.zeros(
            self.config["num_envs"], device=self.device, dtype=torch.float
        )
        self.reset_buf = torch.ones(
            self.config["num_envs"], device=self.device, dtype=torch.long
        )
        self.timeout_buf = torch.zeros(
            self.config["num_envs"], device=self.device, dtype=torch.long
        )
        self.progress_buf = torch.zeros(
            self.config["num_envs"], device=self.device, dtype=torch.long
        )
        self.randomize_buf = torch.zeros(
            self.config["num_envs"], device=self.device, dtype=torch.long
        )

    # @abstractmethod
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # goes through a RL step (includes all the base RL things such as get obs, apply actions, etc.
        # args: actions to apply to the env
        # returns: obs, rewards, resets, info
        pass

    # @abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]:
        # resets a single environment
        # returns: the observations
        pass

    # @abstractmethod
    def reset_idx(self, env_ids: torch.Tensor):
        # ?
        # args: ids of the environments to reset
        pass

    # @property
    def observation_space(self) -> gm.spaces:
        return self.config["observation_space"]

    # a bunch of getters were forgotten right about the time this was being written

    # this is the equivalent of create_sim() in the original class
    def construct(self):
        self.base_env.construct()  # aunque no es la buena cosa, cambiaremos despues

    # the following were included in the original class, but we've failed to see their utility
    # def set_sim_params_up_axis()

    # the following were included in the original class, but we don't know what they do exactly
    def get_state(self):
        # returns: observations (clamped apparently)
        pass


base_task = BaseTask()
print(base_task.load_config())
