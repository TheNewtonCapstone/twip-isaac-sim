from typing import Callable, Type


def base_task_architect(
    training_env_factory: Callable,
    playing_env_factory: Callable,
    agent_factory: Callable,
    task_class: Type,
):
    def base_task_creator(headless: bool, device: str, num_envs: int, playing: bool, domain_rand: bool, config: dict):
        task = task_class(training_env_factory, playing_env_factory, agent_factory)
        task.load_config(headless, device, num_envs, playing, domain_rand, config=config)
        task.construct()

        print("Created", task)

        return task

    return base_task_creator
