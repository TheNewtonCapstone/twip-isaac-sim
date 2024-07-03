import os
import hydra
import torch
import numpy as np

from isaacsim import SimulationApp
from rl_games.torch_runner import Runner
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.common import env_configurations, vecenv

from core.envs.generic_env import GenericEnv
from core.base.base_agent import BaseAgent
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
    "sim_only": True,
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
    "twip_usd_path": os.path.join(get_current_path(), "assets/twip.usd"),
}


@hydra.main(config_name="config", config_path="./configs")
def train(cfg: DictConfig) -> Dict:
    sim_app = SimulationApp(app_settings)

    num_envs = 4

    if app_settings["sim_only"]:
        env = GenericEnv(sim_app=sim_app, world_settings=world_settings, num_envs=num_envs)
        twip = TwipAgent(twip_settings)

        env.construct(twip)
        world = env.world

        twip_view = env.prims
        num_dof = twip_view.num_dof

        import omni.replicator.isaac as dr
        import omni.replicator.core as rep

        dr.physics_view.register_simulation_context(world)
        dr.physics_view.register_articulation_view(twip_view)
        dr.physics_view.torch.set_default_device("cuda:0")


        with dr.trigger.on_rl_frame(num_envs=num_envs):
            # with dr.gate.on_interval(interval=10):
            #     dr.physics_view.randomize_simulation_context(
            #         operation="scaling",
            #         gravity=rep.distribution.uniform((1, 1, -1.5), (1, 1, 2.0)),
            #     )
    
            with dr.gate.on_interval(interval=100):
                dr.physics_view.randomize_articulation_view(
                    view_name=twip_view.name,
                    operation="direct",
                    joint_velocities=rep.distribution.uniform(tuple([-200]*num_dof), tuple([200]*num_dof)),
                )
            
        rep.orchestrator.run()

        frame_idx = 0
        while sim_app.is_running():
            if world.is_playing():
                reset_inds = list()
                if frame_idx % 200 == 0:
                    # triggers reset every 200 steps
                    reset_inds = np.arange(num_envs)
                dr.physics_view.step_randomization(reset_inds)
                world.step(render=True)
                frame_idx += 1

            env.step(not app_settings["headless"])

        return

    runner_config = OmegaConf.to_container(cfg, resolve=True)

    def generic_env_factory() -> GenericEnv:
        return GenericEnv(
            sim_app=sim_app,
            world_settings=world_settings,
            num_envs=runner_config["params"]["config"]["num_actors"],
        )

    def twip_agent_factory() -> TwipAgent:
        return TwipAgent(twip_settings)

    task_architect = base_task_architect(
        generic_env_factory,
        twip_agent_factory,
        sim_app,
        GenericTask,
        app_settings["headless"],
    )

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
