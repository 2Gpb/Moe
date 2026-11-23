"""YAML configs. A config can inherit another one with `base: other.yaml`."""

import copy
from pathlib import Path

import yaml


def merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(open(path)) or {}
    if "base" in cfg:
        cfg = merge(load_config(path.parent / cfg.pop("base")), cfg)
    return cfg
