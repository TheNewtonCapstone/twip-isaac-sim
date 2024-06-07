from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": False})

while sim_app.is_running():
    sim_app.update()

sim_app.close()
