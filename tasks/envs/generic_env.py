import gc
from abc import ABC, abstractmethod

from tasks.base.base_agent import BaseAgent
from tasks.base.base_env import BaseEnv


class GenericEnv(BaseEnv):
    def __init__(self, _o_world_settings):
        super().__init__(_o_world_settings)

    def construct(self):
        super().construct()

    def step(self, _render):
        super().step(_render)

    def reset(self):
        super().reset()

    def add_agent(self, _agent: BaseAgent) -> bool:
        super().add_agent(_agent)

    def pre_play(self, _sim_app) -> None:
        super().pre_play(_sim_app)
