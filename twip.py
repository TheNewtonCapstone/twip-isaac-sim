import argparse
import os

import torch
from core.envs.generic_env import GenericEnv
from core.envs.procedural_env import ProceduralEnv
from core.terrain.flat_terrain import FlatTerrainBuilder
from core.terrain.perlin_terrain import PerlinTerrainBuilder
from core.twip.generic_task import GenericTask
from core.twip.twip_agent import TwipAgent
from core.utils.config import load_config
from core.utils.path import get_current_path, build_child_path_with_prefix
from isaacsim import SimulationApp
from stable_baselines3 import PPO

twip_settings = {
    "twip_urdf_path": os.path.join(
        get_current_path(__file__), "core/twip/assets/twip.urdf"
    ),
    "twip_usd_path": os.path.join(
        get_current_path(__file__), "core/twip/assets/twip.usd"
    ),
}


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twip.py", description="Entrypoint for any TWIP-related actions."
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
        default="configs/task_twip.yaml",
    )
    parser.add_argument(
        "--world-config",
        type=str,
        help="Path to the configuration file for the world.",
        default="configs/world.yaml",
    )
    parser.add_argument(
        "--randomization-config",
        type=str,
        help="Enable domain randomization.",
        default="configs/randomization.yaml",
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
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Exports checkpoint as ONNX model.",
        default=False,
    )

    return parser


if __name__ == "__main__":
    parser = setup_argparser()

    cli_args = parser.parse_args()
    rl_config = load_config(cli_args.rl_config)
    world_config = load_config(cli_args.world_config)
    randomization_config = load_config(cli_args.randomization_config)

    # override config with CLI args & vice versa
    if cli_args.num_envs != -1:
        rl_config["n_envs"] = cli_args.num_envs

        print(
            f"Updated the following parameters: num_envs={rl_config['params']['config']['num_actors']}"
        )

    if cli_args.play and cli_args.checkpoint is None:
        print("Please provide a checkpoint to play the agent.")
        exit(1)

    # if we're exporting, don't show the GUI
    cli_args.headless = cli_args.headless or cli_args.export_onnx

    sim_app = SimulationApp(
        {"headless": cli_args.headless}, experience="./apps/omni.isaac.sim.twip.kit"
    )

    # ---------- #
    # SIMULATION #
    # ---------- #

    if cli_args.sim_only:
        env = ProceduralEnv(
            world_settings=world_config,
            num_envs=1,
            terrain_builders=[PerlinTerrainBuilder(), FlatTerrainBuilder()],
            randomization_settings=randomization_config,
        )

        twip = TwipAgent(twip_settings)

        env.construct(twip)
        env.reset()

        while sim_app.is_running():
            if env.world.is_playing():
                env.step(torch.zeros(env.num_envs, 2), render=not cli_args.headless)

    # ----------- #
    #     RL      #
    # ----------- #

    def generic_env_factory() -> GenericEnv:
        return GenericEnv(
            world_settings=world_config,
            num_envs=rl_config["n_envs"],
            terrain_builders=[FlatTerrainBuilder(size=[10, 10])],
            randomization_settings=randomization_config,
        )

    def procedural_env_factory() -> ProceduralEnv:
        terrains_size = [10, 10]
        terrains_resolution = [20, 20]

        return ProceduralEnv(
            world_settings=world_config,
            num_envs=rl_config["n_envs"],
            terrain_builders=[
                FlatTerrainBuilder(size=terrains_size),
                PerlinTerrainBuilder(
                    size=terrains_size,
                    resolution=terrains_resolution,
                    height=0.05,
                    octave=4,
                    noise_scale=2,
                ),
                PerlinTerrainBuilder(
                    size=terrains_size,
                    resolution=terrains_resolution,
                    height=0.03,
                    octave=8,
                    noise_scale=4,
                ),
                PerlinTerrainBuilder(
                    size=terrains_size,
                    resolution=terrains_resolution,
                    height=0.02,
                    octave=16,
                    noise_scale=8,
                ),
            ],
            randomization_settings=randomization_config,
        )

    def twip_agent_factory() -> TwipAgent:
        return TwipAgent(twip_settings)

    task_runs_directory = "runs"
    task_name = build_child_path_with_prefix(
        rl_config["task_name"], task_runs_directory
    )

    task = GenericTask(
        headless=cli_args.headless,
        device=rl_config["device"],
        num_envs=rl_config["n_envs"],
        playing=cli_args.play,
        max_episode_length=rl_config["ppo"]["n_steps"],
        domain_randomization=randomization_config,
        training_env_factory=procedural_env_factory,
        playing_env_factory=generic_env_factory,
        agent_factory=twip_agent_factory,
    )
    task.construct()

    model = PPO(
        rl_config["policy"],
        task,
        verbose=2,
        device=rl_config["device"],
        seed=rl_config["seed"],
        learning_rate=float(rl_config["base_lr"]),
        n_steps=rl_config["ppo"]["n_steps"],
        batch_size=rl_config["ppo"]["batch_size"],
        n_epochs=rl_config["ppo"]["n_epochs"],
        gamma=rl_config["ppo"]["gamma"],
        gae_lambda=rl_config["ppo"]["gae_lambda"],
        clip_range=float(rl_config["ppo"]["clip_range"]),
        clip_range_vf=(
            None
            if rl_config["ppo"]["clip_range_vf"] == "None"
            else float(rl_config["ppo"]["clip_range_vf"])
        ),
        ent_coef=rl_config["ppo"]["ent_coef"],
        vf_coef=rl_config["ppo"]["vf_coef"],
        max_grad_norm=rl_config["ppo"]["max_grad_norm"],
        use_sde=rl_config["ppo"]["use_sde"],
        sde_sample_freq=rl_config["ppo"]["sde_sample_freq"],
        target_kl=(
            None
            if rl_config["ppo"]["target_kl"] == "None"
            else float(rl_config["ppo"]["target_kl"])
        ),
        tensorboard_log=task_runs_directory,
    )

    # we're not exporting nor purely simulating, so we're training
    if not cli_args.export_onnx:
        model.learn(
            total_timesteps=rl_config["timesteps_per_env"] * rl_config["n_envs"],
            tb_log_name=task_name,
            progress_bar=True,
        )
        model.save(f"{task_runs_directory}/{task_name}/model.zip")

        exit(1)

    # ----------- #
    #    ONNX     #
    # ----------- #

    # Load model from checkpoint
    model = PPO.load(cli_args.checkpoint)

    # Create dummy observations tensor for tracing torch model
    obs_shape = model.observation_space.shape
    dummy_input = torch.rand((1, *obs_shape))

    # Simplified network for actor inference
    # Tested for continuous_a2c_logstd
    class OnnxablePolicy(torch.nn.Module):
        def __init__(self, actor: torch.nn.Module):
            super().__init__()
            self.actor = actor

        def forward(
            self,
            observation: torch.Tensor,
        ):
            return self.actor(observation, deterministic=True)

    onnxable_model = OnnxablePolicy(model.policy.actor)
    torch.onnx.export(
        onnxable_model,
        dummy_input,
        f"{cli_args.checkpoint}.onnx",
        verbose=True,
        input_names=["observations"],
        output_names=["actions"],
    )  # outputs are mu (actions), sigma, value

    print(f"Exported to {cli_args.checkpoint}.onnx!")
