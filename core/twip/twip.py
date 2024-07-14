from datetime import datetime
import os
import torch

import argparse
import numpy as np

from isaacsim import SimulationApp
from rl_games.torch_runner import Runner
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.common import env_configurations, vecenv

from core.envs.generic_env import GenericEnv
from core.twip.twip_agent import TwipAgent
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
        "--play",
        action="store_true",
        help="Play the agent using a trained checkpoint.",
        default=False,
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
        cli_args.num_envs = rl_config["params"]["config"]["num_actors"]
    else:
        rl_config["params"]["config"]["num_actors"] = cli_args.num_envs

    print(
        f"Updated the following parameters: num_envs={rl_config['params']['config']['num_actors']}"
    )

    if cli_args.play and cli_args.checkpoint is None:
        print("Please provide a checkpoint to play the agent.")
        exit(1)

    sim_app = SimulationApp(
        {"headless": cli_args.headless}, experience="./apps/omni.isaac.sim.twip.kit"
    )

    # ---------- #
    # SIMULATION #
    # ---------- #

    if cli_args.sim_only:
        env = GenericEnv(
            world_settings=world_config,
            num_envs=cli_args.num_envs,
        )

        twip = TwipAgent(twip_settings)

        env.construct(twip)
        env.reset()

        world = env.world
        num_envs = env.num_envs
        twip_view = env.twip_art_view
        num_dof = twip_view.num_dof

        # set up randomization with omni.replicator.isaac, imported as dr
        import omni.replicator.isaac as dr
        import omni.replicator.core as rep

        dr.physics_view.register_simulation_context(world)
        dr.physics_view.register_articulation_view(twip_view)

        with dr.trigger.on_rl_frame(num_envs=num_envs):
            with dr.gate.on_interval(interval=100):
                dr.physics_view.randomize_articulation_view(
                    view_name=twip_view.name,
                    operation="direct",
                    joint_efforts=rep.distribution.uniform(
                        tuple([-10] * num_dof), tuple([10] * num_dof)
                    ),
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
                env.step(torch.zeros(num_envs, 2), render=cli_args.headless)
                frame_idx += 1

    # ----------- #
    # RL TRAINING #
    # ----------- #

    # override config with CLI args

    def generic_env_factory() -> GenericEnv:
        return GenericEnv(
            world_settings=world_config,
            num_envs=cli_args.num_envs,
        )

    def twip_agent_factory() -> TwipAgent:
        return TwipAgent(twip_settings)
    
    rl_config["params"]["config"]["full_experiment_name"] = rl_config["params"]["config"]["full_experiment_name"] + "_" + datetime.now().strftime("%d%m%y%H%M%S")

    task_architect = base_task_architect(
        generic_env_factory,
        twip_agent_factory,
        GenericTask,
    )

    # registers task creation function with RL Games
    env_configurations.register(
        "generic",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: task_architect(
                cli_args.headless,
                rl_config["device"],
                cli_args.num_envs,
            ),
        },
    )
    vecenv.register(
        "RLGPU",
        lambda config_name, num_actors, **kwargs: task_architect(
            cli_args.headless,
            rl_config["device"],
            num_actors,
        ),
    )

    runner = Runner(IsaacAlgoObserver())
    runner.load(rl_config)
    runner.reset()

    runner.run(
        {
            "train": not cli_args.play,
            "play": cli_args.play,
            "checkpoint": cli_args.checkpoint,
        }
    )
