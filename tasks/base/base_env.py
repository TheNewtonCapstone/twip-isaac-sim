import gc
from abc import abstractmethod

from omni.isaac.kit import SimulationApp

config = {"headless": False}
sim_app = SimulationApp(config)

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.stage import create_new_stage_async, update_stage_async


class BaseEnv(object):
    def __init__(self) -> None:
        self.world = None
        self._current_tasks = None
        self._world_settings = {
            "physics_dt": 1.0 / 60.0,
            "stage_units_in_neters": 1.0,
            "rendereing_dt": 1.0 / 60.0,
        }
        return


world = World()
world.scene.add_default_ground_plane()

while True:
    world.step(render=True)
