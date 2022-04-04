# Gym2Real_isaacgym
This repo provides a custom task to be used with Isaac Gym as well as scripts for training and exporting models.

## TWIP
`tasks/twip/`: A custom task for our two-wheeled inverted pendulum robot. The task is set up to balance the robot.

## Adding a New Task
Create a new folder called `tasks/{task_name}`. The following files will be needed:
- `tasks/{task_name}/{task_name}.py`: Class defining the task
- `tasks/{task_name}/assets/`: Put your URDF and meshes here
- `cfg/task/{TaskName}.yaml`: Environment parameters and domain randomization
- `cfg/train/{TaskName}PPO.yaml`: Hyperparameters for training

## Training
`python train.py task={task_name}`

## Exporting ONNX
`python export.py task={task_name} checkpoint="runs/{TaskName}/nn/{checkpoint_name}.pth"`

## IsaacGymEnvs
This repo was designed to be used after installing IsaacGymEnvs (`https://github.com/NVIDIA-Omniverse/IsaacGymEnvs`) as a Python library.
IsaacGymEnvs comes close to being usable as a library, but requires adding your custom task to a map.

The following files are taken from IsaacGymEnvs and modified so that this isn't necessary:
 - `utils/rlgames_utils.py` (minimal changes)
 - `base/vec_task.py` (minimal changes)
 - `train.py`

Instead of looking for tasks in a static map, this code uses pydoc.locate to dynamically load the task class.

Tested with IsaacGymEnvs commit `9656bac7e59b96382d2c5040b90d2ea5c227d56d`.