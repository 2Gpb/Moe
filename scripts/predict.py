"""Classify a single text with a trained adapter (or the frozen base model).

    uv run python scripts/predict.py --run runs/lora_attn_s0 --task fpb \
        "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007."
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import torch

from finmoe.build import build_model, load_trained
from finmoe.config import load_config
from finmoe.data import TASKS, Collator
from finmoe.modeling import VerbalizerHead, pick_device, pick_dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--task", default="fpb", choices=sorted(TASKS))
    parser.add_argument("--run", default=None, help="run directory; omit to use the zero-shot base model")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    device = pick_device()
    dtype = pick_dtype(device)
    if args.run:
        model, tokenizer, _ = load_trained(args.run, device, dtype)
    else:
        model, tokenizer, _ = build_model(load_config(args.config), device, dtype)

    batch = Collator(tokenizer)([{"task": args.task, "text": args.text, "label": 0}])
    with torch.no_grad():
        probs = VerbalizerHead(model, tokenizer)(batch)[0].softmax(-1)
    labels = TASKS[args.task].labels
    for label, p in sorted(zip(labels, probs.tolist()), key=lambda x: -x[1]):
        print(f"{label:>9s}  {p:.3f}")


if __name__ == "__main__":
    main()
