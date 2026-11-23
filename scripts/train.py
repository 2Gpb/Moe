"""Fine-tune a MoE model with one adaptation method.

    uv run python scripts/train.py --config configs/lora_attn.yaml --seed 1
"""

import argparse
import json
from pathlib import Path

from finmoe.config import load_config
from finmoe.training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="runs")
    parser.add_argument("--name", default=None, help="run name, defaults to <config>_s<seed>")
    parser.add_argument("--set", nargs="*", default=[], help="overrides such as optim.lr=1e-4")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    name = args.name or f"{Path(args.config).stem}_s{cfg['seed']}"
    summary = train(cfg, Path(args.out) / name)
    print(json.dumps({"run": name, "test": summary["test"].get("mean"), "trainable": summary["trainable_params"]}))


if __name__ == "__main__":
    main()
