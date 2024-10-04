from typing import Dict, Any, Callable, List, Optional, Sequence, Type

import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvStepReturn, VecEnvObs

import numpy as np
from core.base.base_agent import BaseAgent
from core.base.base_env import BaseEnv


class BaseTask(VecEnv):
    def __init__(self, headless: bool, device: str, num_envs: int, playing: bool, observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Box, reward_space: gym.spaces.Box, config: Dict[str, Any],
                 training_env_factory: Callable[..., BaseEnv], playing_env_factory: Callable[..., BaseEnv],
                 agent_factory: Callable[..., BaseAgent], ):
        super(BaseTask, self).__init__(num_envs=num_envs, observation_space=observation_space,
                                       action_space=action_space)

        self.config = {}

        self.training_env_factory = training_env_factory
        self.playing_env_factory = playing_env_factory
        self.agent_factory = agent_factory

        self.agent: BaseAgent | None = None
        self.envs: List[BaseEnv] = []

        self.config: Dict[str, Any] = config

        self.headless: bool = headless
        self.device: str = device
        self.playing: bool = playing

        self.observation_space: gym.spaces.Box = observation_space
        self.action_space: gym.spaces.Box = action_space
        self.reward_space: gym.spaces.Box = reward_space

        self.num_envs: int = num_envs
        self.max_episode_length: int = self.config["max_episode_length"]

        self.num_observations: int = self.config["num_observations"]
        self.num_actions: int = self.config["num_actions"]
        self.num_states: int = self.config["num_states"]
        self.actions: np.ndarray = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)

        self.domain_randomization: bool = self.config.get("domain_randomization", False)

        self._setup_buffers()

    def __str__(self):
        return f"{self.__class__.__name__} with {self.num_envs} environments, {self.num_observations} observations, {self.num_actions} actions, {self.num_states} states."

    def construct(self) -> bool:
        pass

    def _setup_buffers(self) -> None:
        self.obs_buf = np.zeros((self.num_envs, self.num_observations,), dtype=np.float32)
        self.rewards_buf = np.zeros((self.num_envs,), dtype=np.float32)
        self.dones_buf = np.zeros((self.num_envs,), dtype=np.bool_)
        self.progress_buf = np.zeros((self.num_envs,), dtype=np.ulonglong)

    # Gymnasium methods (required from VecEnv)

    def reset(self) -> VecEnvObs:
        pass

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        return

    def step_wait(self) -> VecEnvStepReturn:
        pass

    def close(self) -> None:
        pass

    def seed(self, seed: Optional[int] = None) -> Sequence[None | int]:
        pass

    # Helper methods (shouldn't need to be overridden)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        # noinspection PyTypeChecker
        return [self.envs[i] for i in indices]
