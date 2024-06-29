import os
import hydra
import torch

from isaacsim import SimulationApp
from rl_games.torch_runner import Runner
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.common import env_configurations, vecenv

from core.envs.generic_env import GenericEnv
from core.twip.twip_agent import TwipAgent, WheelDriveType
from core.twip.generic_task import GenericTask
from core.utils.config import omegaconf_to_dict
from core.utils.env import base_task_architect

from omegaconf import DictConfig, OmegaConf
from typing import Dict
from copy import deepcopy


def get_current_path() -> str:
    return os.path.dirname(os.path.realpath(__file__))


def to_absolute_path(rel_path: str) -> str:
    return os.path.abspath(rel_path)


rl_settings = {
    "load_checkpoint": True,
    "checkpoint": to_absolute_path("checkpoints/runs/twip_balancing/nn/twip.pth"),
    "train": True,
}

app_settings = {
    "headless": False,
    "sim_only": False,
}

world_settings = {
    "physics_dt": 1.0 / 200.0,
    "stage_units_in_meters": 1.0,
    "rendering_dt": 1.0 / 60.0,
    "backend": "torch",
    "device": "cuda:0",
}

twip_settings = {
    "twip_urdf_path": os.path.join(get_current_path(), "assets/twip.urdf"),
}


@hydra.main(config_name="config", config_path="./configs")
def train(cfg: DictConfig) -> Dict:
    sim_app = SimulationApp(app_settings)

    env = GenericEnv(world_settings)
    twip = TwipAgent(twip_settings)

    env.add_agent(twip)

    twip.set_target_velocity(WheelDriveType.LEFT, 0)
    twip.set_target_velocity(WheelDriveType.RIGHT, 0)

    if app_settings["sim_only"]:
        env.construct(sim_app)
        env.prepare()

        while sim_app.is_running():
            if env.world.current_time % 10 <= 5:
                twip.set_target_velocity(WheelDriveType.LEFT, 0)
            else:
                twip.set_target_velocity(WheelDriveType.LEFT, 1400)

            env.step(not app_settings["headless"])

        return

    task_architect = base_task_architect(
        env,
        sim_app,
        GenericTask,
        app_settings["headless"],
    )

    runner_config = OmegaConf.to_container(cfg, resolve=True)

    # registers task creation function with RL Games
    env_configurations.register(
        "generic",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: task_architect(),
        },
    )
    vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: task_architect())

    runner = Runner(IsaacAlgoObserver())
    runner.load(runner_config)
    runner.reset()

    runner.run(
        {
            "train": rl_settings["train"],
            "play": not rl_settings["train"],
            "checkpoint": rl_settings["checkpoint"]
            if rl_settings["load_checkpoint"]
            else None,
        }
    )


if __name__ == "__main__":
    train()
