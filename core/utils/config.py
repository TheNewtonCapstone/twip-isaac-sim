from omegaconf import DictConfig
from typing import Dict


def omegaconf_to_dict(d: DictConfig) -> Dict:
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret
