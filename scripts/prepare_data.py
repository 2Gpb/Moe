"""Download the datasets and freeze the train/val/test splits as jsonl files."""

import argparse
import collections
from pathlib import Path

import yaml

from finmoe.data import build_splits, save_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    splits = build_splits(seed=cfg["seed"], caps=cfg.get("caps"))
    save_splits(splits, Path(cfg["root"]))

    for task, parts in splits.items():
        for split, rows in parts.items():
            dist = collections.Counter(r["label"] for r in rows)
            print(f"{task:10s} {split:5s} n={len(rows):5d} labels={dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
