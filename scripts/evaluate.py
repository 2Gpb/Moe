"""Evaluate the frozen base model (zero-/few-shot) or a trained run on the test splits.

    uv run python scripts/evaluate.py --config configs/base.yaml --shots 0
    uv run python scripts/evaluate.py --config configs/base.yaml --shots 2
    uv run python scripts/evaluate.py --run runs/lora_attn_s0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

import torch

from finmoe.build import build_model, load_trained
from finmoe.config import load_config
from finmoe.data import Collator, load_split, pick_demos
from finmoe.modeling import VerbalizerHead, pick_device, pick_dtype
from finmoe.training import evaluate_split, load_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--run", default=None)
    parser.add_argument("--shots", type=int, default=0, help="in-context examples per label")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    if args.run:
        device = pick_device()
        dtype = pick_dtype(device)
        model, tokenizer, cfg = load_trained(args.run, device, dtype)
    else:
        cfg = load_config(args.config, args.set)
        device = pick_device(cfg.get("device", "auto"))
        dtype = pick_dtype(device, cfg.get("dtype", "auto"))
        model, tokenizer, _ = build_model(cfg, device, dtype)

    data = cfg["data"]
    tasks = args.tasks or data["tasks"]
    demos = None
    max_length = data.get("max_length", 192)
    if args.shots:
        demos = {t: pick_demos(load_split(data["root"], t, "train"), args.shots, seed=cfg.get("seed", 0)) for t in tasks}
        max_length = 1024

    collator = Collator(tokenizer, max_length=max_length, demos=demos)
    head = VerbalizerHead(model, tokenizer)
    rows = load_rows(data["root"], tasks, args.split)
    batch_size = cfg["optim"].get("eval_batch_size", 32) // (4 if args.shots else 1)
    records, metrics, seconds = evaluate_split(head, rows, collator, batch_size)

    name = Path(args.run).name if args.run else f"base_{args.shots}shot"
    out = Path(args.out or f"results/eval/{name}_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"name": name, "shots": args.shots, "split": args.split, "metrics": metrics,
              "ms_per_example": 1000 * seconds / len(records), "device": device.type}
    json.dump(result, open(out, "w"), indent=2)
    with open(out.with_suffix(".predictions.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    for task, m in metrics.items():
        print(f"{task:10s} acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
