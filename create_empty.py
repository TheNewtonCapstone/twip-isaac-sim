import argparse

from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.sim import SimulationCfg, SimulationContext

parser = argparse.ArgumentParser(
    description="Tutorial on creating an empty stage")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
    sim_cfg = SimulationCfg(dt=0.01, substeps=1)
    sim_ctx = SimulationContext(sim_cfg)

    sim_ctx.set_camera_view([2.5, 2.5, 2.5], [0, 0, 0])

    sim_ctx.reset()

    while simulation_app.is_running():
        sim_ctx.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
