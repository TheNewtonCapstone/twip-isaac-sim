import gc
from abc import abstractmethod

import isaacsim
from omni.isaac.kit import SimulationApp

config = {"headless": False}
sim_app = SimulationApp(config)

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.stage import create_new_stage_async, update_stage_async

# TODO: separate into 3 classes: BaseEnv, BaseTask, BaseAgent
# BaseEnv: contains the world, tasks, agents, and settings (mother class, if you will)
# BaseTask: contains the task settings and the task logic
# BaseAgent: contains the agent settings and the agent logic
# The goal is to make it so that any task can be run in any environment with any agent
# Need to figure out how abc works in python and how to use it to enforce the structure of the classes


class BaseEnv(object):
    def __init__(self) -> None:
        self.world = None
        self._current_tasks = None
        self._world_settings = {
            "physics_dt": 1.0 / 60.0,
            "stage_units_in_neters": 1.0,
            "rendering_dt": 1.0 / 60.0,
        }
        return


world = World()
world.scene.add_default_ground_plane()

while sim_app.is_running():
    world.step(render=True)
