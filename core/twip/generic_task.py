from abc import ABC, abstractmethod
import gymnasium as gm
import numpy as np
import torch
from typing import Dict, Any, Tuple

from isaacsim import SimulationApp
from rl_games.common.ivecenv import IVecEnv

from core.base.base_task import BaseTask
from core.base.base_env import BaseEnv


class GenericTask(BaseTask):
    def __init__(self, _base_env: BaseEnv):
        super().__init__(_base_env)

    def load_config(self):
        config = {}
        config["device"] = "cuda:0"
        config["headless"] = False

        config["num_envs"] = 1

        config["num_agents"] = 1
        config["num_observations"] = 2
        config["num_actions"] = 2
        config["num_states"] = 1

        config["observation_space"] = gm.spaces.Box(
            np.ones(config["num_observations"]) * -np.Inf,
            np.ones(config["num_observations"]) * np.Inf,
        )
        config["action_space"] = gm.spaces.Box(
            np.ones(config["num_actions"]) * -1.0,
            np.ones(config["num_actions"]) * 1.0,
        )  # not sure why it's different from the other spaces
        config["state_space"] = gm.spaces.Box(
            np.ones(config["num_states"]) * -np.Inf,
            np.ones(config["num_states"]) * np.Inf,
        )  # still not sure what a state is

        # task-specific config
        config["domain_randomization"] = {}

        print(f"{self.__class__.__name__} loaded config {config}")

        return super().load_config(config)

    def construct(self, sim_app: SimulationApp) -> bool:
        c_res = self.base_env.construct(sim_app)

        if not c_res:
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

        self.base_env.step(_render=not self.headless)
        obs = {"obs": twip_agent.get_observations()}

        print(obs)

        return obs, torch.zeros(1024), torch.zeros(1024), {}

    def reset(self) -> Dict[str, torch.Tensor]:
        # resets a single environment
        # returns: the observations

        super().reset()

        twip_agent = self.base_env.agent

        self.base_env.prepare()
        obs = {"obs": twip_agent.get_observations()}

        return obs
