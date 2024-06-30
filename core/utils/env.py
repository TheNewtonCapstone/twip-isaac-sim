from isaacsim import SimulationApp

from core.base.base_env import BaseEnv

from typing import Callable, Type


def base_task_architect(
    env_factory: Callable,
    agent_factory: Callable,
    sim_app: SimulationApp,
    task_class: Type,
    headless: bool,
    post_create_hook: Callable = None,
):
    def base_task_creator():
        task = task_class(env_factory, agent_factory)
        task.load_config(headless=headless)
        task.construct(sim_app)

        print("Task created")

        if post_create_hook is not None:
            post_create_hook()

        return task

    return base_task_creator
