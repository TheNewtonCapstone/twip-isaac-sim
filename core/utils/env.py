from isaacsim import SimulationApp

from core.base.base_env import BaseEnv

from typing import Callable, Type


def base_task_architect(
    env: BaseEnv,
    sim_app: SimulationApp,
    task_class: Type,
    post_create_hook: Callable = None,
):
    def base_task_creator():
        task = task_class(env)
        task.load_config()
        task.construct(sim_app)

        print("Task created")

        if post_create_hook is not None:
            post_create_hook()

        return task

    return base_task_creator
