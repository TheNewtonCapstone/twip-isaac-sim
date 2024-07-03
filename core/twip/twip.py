import os
import yaml
import torch

import argparse
import numpy as np

from isaacsim import SimulationApp
from rl_games.torch_runner import Runner
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.common import env_configurations, vecenv

from core.envs.generic_env import GenericEnv
from core.base.base_agent import BaseAgent
from core.twip.twip_agent import TwipAgent, WheelDriveType
from core.twip.generic_task import GenericTask
from core.utils.env import base_task_architect
from core.utils.path import get_current_path
from core.utils.config import load_config

twip_settings = {
    "twip_urdf_path": os.path.join(get_current_path(__file__), "assets/twip.urdf"),
    "twip_usd_path": os.path.join(get_current_path(__file__), "assets/twip.usd"),
}


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="core/twip/twip.py", description="Entrypoint for any TWIP-related actions."
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode.", default=False
    )
    parser.add_argument(
        "--sim-only",
        action="store_true",
        help="Run the simulation only (no RL).",
        default=False,
    )
    parser.add_argument(
        "--rl-config",
        type=str,
        help="Path to the configuration file for RL.",
        default="configs/twip.yaml",
    )
    parser.add_argument(
        "--world-config",
        type=str,
        help="Path to the configuration file for the world.",
        default="configs/world.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the checkpoint to load for RL.",
        default=None,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the agent using RL. Set to False to play.",
        default=True,
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        help="Number of environments to run (will be read from the rl-config if not specified).",
        default=-1,
    )

    return parser


if __name__ == "__main__":
    parser = setup_argparser()

    cli_args = parser.parse_args()
    rl_config = load_config(cli_args.rl_config)
    world_config = load_config(cli_args.world_config)

    # override config with CLI args & vice versa
    if cli_args.num_envs == -1:
        cli_args.num_envs = rl_config["params"]["num_actors"]
    else:
        rl_config["params"]["num_actors"] = cli_args.num_envs

    sim_app = SimulationApp({"headless": cli_args.headless})

    # ---------- #
    # SIMULATION #
    # ---------- #

    if cli_args.sim_only:
        env = GenericEnv(
            sim_app=sim_app,
            world_settings=world_config,
            num_envs=cli_args.num_envs,
        )

        twip = TwipAgent(twip_settings)

        env.construct(twip)

        while sim_app.is_running():
            env.step(True) # Fix this :)

    # ----------- #
    # RL TRAINING #
    # ----------- #

    # override config with CLI args

    def generic_env_factory() -> GenericEnv:
        return GenericEnv(
            sim_app=sim_app,
            world_settings=world_config,
            num_envs=cli_args.num_envs,
        )

    def twip_agent_factory() -> TwipAgent:
        return TwipAgent(twip_settings)

    task_architect = base_task_architect(
        generic_env_factory,
        twip_agent_factory,
        sim_app,
        GenericTask,
    )

    # registers task creation function with RL Games
    env_configurations.register(
        "generic",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: task_architect(kwargs),
        },
    )
    vecenv.register(
        "RLGPU",
        lambda config_name, num_actors, **kwargs: task_architect(
            cli_args.headless, rl_config["device"], num_actors
        ),
    )

    runner = Runner(IsaacAlgoObserver())
    runner.load(rl_config)
    runner.reset()

    runner.run(
        {
            "train": cli_args.train,
            "play": not cli_args.train,
            "checkpoint": cli_args.checkpoint,
        }
    )
