from typing import Callable, Type


def base_task_architect(
    env_factory: Callable,
    agent_factory: Callable,
    task_class: Type,
):
    def base_task_creator(
        headless: bool, device: str, num_envs: int, domain_rand: bool = False
    ):
        task = task_class(env_factory, agent_factory)
        task.load_config(headless, device, num_envs, domain_rand)
        task.construct()

        print("Created", task)

        return task

    return base_task_creator
