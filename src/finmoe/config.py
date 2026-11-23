"""YAML configs with single inheritance through a ``base:`` key."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(open(path)) or {}
    if "base" in cfg:
        parent = load_config(path.parent / cfg.pop("base"))
        cfg = deep_merge(parent, cfg)
    for item in overrides or []:
        key, value = item.split("=", 1)
        set_by_path(cfg, key, yaml.safe_load(value))
    return cfg


def set_by_path(cfg: dict, dotted: str, value) -> None:
    node = cfg
    *parents, last = dotted.split(".")
    for k in parents:
        node = node.setdefault(k, {})
    node[last] = value
