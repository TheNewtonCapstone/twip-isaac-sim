import os


def get_current_path(file_name: str) -> str:
    return os.path.dirname(os.path.realpath(file_name))


def to_absolute_path(rel_path: str) -> str:
    return os.path.abspath(rel_path)


def get_cwd() -> str:
    return os.getcwd()
