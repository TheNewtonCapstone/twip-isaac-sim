import os
import hydra
import torch

from isaacsim import SimulationApp
from rl_games.torch_runner import Runner

from core.envs.generic_env import GenericEnv
from core.twip.twip_agent import TwipAgent, WheelDriveType
from core.twip.generic_task import GenericTask


def get_current_path() -> str:
    return os.path.dirname(os.path.realpath(__file__))


app_settings = {
    "headless": False,
}

world_settings = {
    "physics_dt": 1.0 / 60.0,
    "stage_units_in_meters": 1.0,
    "rendering_dt": 1.0 / 60.0,
}

twip_settings = {
    "twip_urdf_path": os.path.join(get_current_path(), "assets/twip.urdf"),
}

if __name__ == "__main__":
    sim_app = SimulationApp(app_settings)

    env = GenericEnv(world_settings)
    twip = TwipAgent(twip_settings)

    env.add_agent(twip)

    task = GenericTask(env)
    task.load_config()
    task.construct(sim_app)

    twip.set_target_velocity(WheelDriveType.LEFT, 0)
    twip.set_target_velocity(WheelDriveType.RIGHT, 1200)

    runner = Runner()

    runner.run({"train": True, "play": False})

    while False and sim_app.is_running():
        task.step(torch.zeros(1))

        if env.o_world.current_time % 10 <= 5:
            twip.set_target_velocity(WheelDriveType.LEFT, 0)
        else:
            twip.set_target_velocity(WheelDriveType.LEFT, 1400)
