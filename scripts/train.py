"""Fine-tune a MoE model with one adaptation method.

    uv run python scripts/train.py --config configs/lora_attn.yaml --seed 1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

from finmoe.config import load_config
from finmoe.training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="runs")
    parser.add_argument("--name", default=None, help="run name, defaults to <config>_s<seed>")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    name = args.name or f"{Path(args.config).stem}_s{cfg['seed']}"
    summary = train(cfg, Path(args.out) / name)
    print(json.dumps({"run": name, "test": summary["test"].get("mean"), "trainable": summary["trainable_params"]}))


if __name__ == "__main__":
    main()
