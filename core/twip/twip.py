import os
import hydra

from isaacsim import SimulationApp

from core.envs.generic_env import GenericEnv
from core.twip.twip_agent import TwipAgent, WheelDriveType


def get_current_path() -> str:
    return os.path.dirname(os.path.realpath(__file__))


world_settings = {
    "physics_dt": 1.0 / 60.0,
    "stage_units_in_meters": 1.0,
    "rendering_dt": 1.0 / 60.0,
}

twip_settings = {
    "twip_urdf_path": os.path.join(get_current_path(), "assets/twip.urdf"),
}

if __name__ == "__main__":
    config = {"headless": False}
    sim_app = SimulationApp(config)

    env = GenericEnv(world_settings)
    env.construct()

    # task = BaseTask()
    # task.construct()
    # task.add_agent()
    # task.prepare()

    twip = TwipAgent(twip_settings)
    env.add_agent(twip)
    env.pre_play(sim_app)

    twip.set_target_velocity(WheelDriveType.LEFT, 0)
    twip.set_target_velocity(WheelDriveType.RIGHT, 1200)

    while sim_app.is_running():
        env.step(_render=True)

        twip.get_observations()

        if env.o_world.current_time % 10 <= 5:
            twip.set_target_velocity(WheelDriveType.LEFT, 0)
        else:
            twip.set_target_velocity(WheelDriveType.LEFT, 1400)
