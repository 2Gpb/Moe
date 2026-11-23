"""Download the datasets and save fixed train/val/test splits to data/splits/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import collections
import json
import random

import yaml
from datasets import load_dataset

from finmoe.data import TASKS, subsample


def load_flare(repo, task):
    ds = load_dataset(repo)
    out = {}
    for split, name in (("train", "train"), ("valid", "val"), ("test", "test")):
        rows = []
        for r in ds[split]:
            answer = r["answer"].strip()
            if task != "headlines":
                answer = answer.lower()
            text = r["query"] if task == "headlines" else r["text"]
            rows.append({"id": r["id"], "text": text.strip(), "label": TASKS[task].labels.index(answer)})
        out[name] = rows
    return out


def load_tfns(seed):
    # 0 bearish, 1 bullish, 2 neutral -> negative / positive / neutral
    remap = {0: 0, 1: 2, 2: 1}
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    train = [{"id": f"tfns-tr{i}", "text": r["text"].strip(), "label": remap[int(r["label"])]} for i, r in enumerate(ds["train"])]
    test = [{"id": f"tfns-te{i}", "text": r["text"].strip(), "label": remap[int(r["label"])]} for i, r in enumerate(ds["validation"])]
    # no public test split: official validation -> test, 10% of train -> validation
    train, val = stratified_split(train, 0.1, seed)
    return {"train": train, "val": val, "test": test}


def load_sst2():
    ds = load_dataset("stanfordnlp/sst2")
    return {"test": [{"id": f"sst2-{r['idx']}", "text": r["sentence"].strip(), "label": int(r["label"])} for r in ds["validation"]]}


def headline_of(query):
    return query.split("\nText:", 1)[-1].strip().lower()


def dedupe_headlines(splits):
    # the same headline comes with several questions, so drop train headlines that occur in val/test
    test = {headline_of(r["text"]) for r in splits["test"]}
    val = [r for r in splits["val"] if headline_of(r["text"]) not in test]
    held = test | {headline_of(r["text"]) for r in val}
    train = [r for r in splits["train"] if headline_of(r["text"]) not in held]
    return {"train": train, "val": val, "test": splits["test"]}


def stratified_split(rows, frac, seed):
    rng = random.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    keep, held = [], []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n = round(len(group) * frac)
        held += group[:n]
        keep += group[n:]
    return keep, held


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed, caps = cfg["seed"], cfg.get("caps", {})

    splits = {
        "fpb": load_flare("ChanceFocus/flare-fpb", "fpb"),
        "fiqasa": load_flare("ChanceFocus/flare-fiqasa", "fiqasa"),
        "tfns": load_tfns(seed),
        "headlines": dedupe_headlines(load_flare("ChanceFocus/flare-headlines", "headlines")),
        "sst2": load_sst2(),
    }

    root = Path(cfg["root"])
    root.mkdir(parents=True, exist_ok=True)
    for task, parts in splits.items():
        for split, rows in parts.items():
            rows = subsample(rows, caps.get(task, {}).get(split), seed)
            with open(root / f"{task}_{split}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            dist = collections.Counter(r["label"] for r in rows)
            print(f"{task:10s} {split:5s} n={len(rows):5d} labels={dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
