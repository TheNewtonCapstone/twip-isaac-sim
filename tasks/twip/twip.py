import isaacsim
from omni.isaac.kit import SimulationApp

from tasks.envs.generic_env import GenericEnv
from tasks.twip.twip_agent import TwipAgent

world_settings = {
    "physics_dt": 1.0 / 60.0,
    "stage_units_in_neters": 1.0,
    "rendering_dt": 1.0 / 60.0,
}

twip_settings = {}

if __name__ == "__main__":
    config = {"headless": False}
    sim_app = SimulationApp(config)

    base_env = GenericEnv(world_settings)
    base_env.construct()

    agent_settings = {}

    base_agent = TwipAgent(agent_settings)
    base_env.add_agent(base_agent)
    base_env.pre_play(sim_app)

    while sim_app.is_running():
        base_env.step(_render=True)
