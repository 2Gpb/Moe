"""Record how the router distributes tokens over experts for every task.

Produces ``<out>/<name>.pt`` with one routing summary per task and, for the
base model, ``<out>/esft_stats.pt`` with relevance scores aggregated over the
financial *training* data (used to pick experts for ESFT; no test data is involved).

    uv run python scripts/analyze_routing.py --config configs/base.yaml
    uv run python scripts/analyze_routing.py --run runs/esft_gate_s0 --split test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from finmoe.build import build_model, load_trained
from finmoe.config import load_config
from finmoe.data import FIN_TASKS, Collator, load_split, subsample
from finmoe.modeling import VerbalizerHead, pick_device, pick_dtype
from finmoe.moe import RoutingRecorder, find_moe_layers


@torch.no_grad()
def record(head, recorder, rows, collator, batch_size):
    recorder.reset()
    for batch in tqdm(DataLoader(rows, batch_size=batch_size, collate_fn=collator), leave=False):
        recorder.token_mask = batch["attention_mask"].flatten().bool()
        head(batch)
    return recorder.summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--run", default=None)
    parser.add_argument("--split", default="train", help="split used for the financial tasks")
    parser.add_argument("--max-examples", type=int, default=1000, help="0 = use the whole split")
    parser.add_argument("--out", default="results/routing")
    args = parser.parse_args()

    if args.run:
        device = pick_device()
        model, tokenizer, cfg = load_trained(args.run, device, pick_dtype(device))
        name = Path(args.run).name
    else:
        cfg = load_config(args.config)
        device = pick_device(cfg.get("device", "auto"))
        model, tokenizer, _ = build_model(cfg, device, pick_dtype(device, cfg.get("dtype", "auto")))
        name = "base"

    root = cfg["data"]["root"]
    head = VerbalizerHead(model, tokenizer)
    collator = Collator(tokenizer, max_length=cfg["data"].get("max_length", 192))
    recorder = RoutingRecorder(find_moe_layers(model))

    per_task = {}
    for task in FIN_TASKS:
        rows = subsample(load_split(root, task, args.split), args.max_examples or None, seed=0)
        per_task[task] = record(head, recorder, rows, collator, 32)
        print(f"{task}: {int(per_task[task]['tokens'])} tokens, mean entropy {per_task[task]['entropy'].mean():.3f}")
    per_task["sst2"] = record(head, recorder, load_split(root, "sst2", "test"), collator, 32)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"split": args.split, "tasks": per_task}, out / f"{name}_{args.split}.pt")

    if not args.run and args.split == "train":
        # token-weighted average over the financial tasks
        weights = torch.tensor([float(per_task[t]["tokens"]) for t in FIN_TASKS])
        weights = weights / weights.sum()
        stats = {
            key: sum(w * per_task[t][key] for w, t in zip(weights, FIN_TASKS))
            for key in ("selection_ratio", "mean_prob")
        }
        torch.save(stats, out / "esft_stats.pt")
        print(f"saved ESFT relevance scores to {out / 'esft_stats.pt'}")


if __name__ == "__main__":
    main()
