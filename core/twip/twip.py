import os
import hydra
import torch
from rl_games.common import env_configurations, vecenv


from isaacsim import SimulationApp
from rl_games.torch_runner import Runner

from core.envs.generic_env import GenericEnv
from core.twip.twip_agent import TwipAgent, WheelDriveType
from core.twip.generic_task import GenericTask

from omegaconf import DictConfig, OmegaConf
from typing import Dict


@hydra.main(config_name="config", config_path="./../../configs/")
def load_config(cfg: DictConfig) -> DictConfig:
    return cfg


def omegaconf_to_dict(d: DictConfig) -> Dict:
    ret = {}
    exit(1)
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


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
    # config_dict = omegaconf_to_dict(load_config())

    # vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: GenericTask(env))
    # runner = Runner()
    # runner.load(config_dict)
    # runner.reset()

    # runner.run({"train": True, "play": False})

    while sim_app.is_running():
        task.step(torch.zeros(1))

        if env.o_world.current_time % 10 <= 5:
            twip.set_target_velocity(WheelDriveType.LEFT, 0)
        else:
            twip.set_target_velocity(WheelDriveType.LEFT, 1400)
