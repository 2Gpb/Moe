"""Financial classification tasks, fixed splits and prompt formatting.

Every task is cast as next-token classification: the model reads an instruction
plus the input text and we compare the logits of one verbalizer token per class
at the final position. The same formulation is used for zero-shot evaluation
and for fine-tuning, so the numbers are directly comparable.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset

SENTIMENT = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class Task:
    name: str
    labels: tuple[str, ...]
    instruction: str | None  # None -> the source already provides a per-example instruction


TASKS: dict[str, Task] = {
    "fpb": Task(
        "fpb",
        SENTIMENT,
        "Classify the sentiment of the sentence from a financial news article as negative, neutral or positive.",
    ),
    "fiqasa": Task(
        "fiqasa",
        SENTIMENT,
        "Classify the sentiment of the financial social media post as negative, neutral or positive.",
    ),
    "tfns": Task(
        "tfns",
        SENTIMENT,
        "Classify the sentiment of the financial tweet as negative, neutral or positive.",
    ),
    "headlines": Task("headlines", ("No", "Yes"), None),
    # general-domain control, used only for routing analysis
    "sst2": Task("sst2", ("negative", "positive"), "Classify the sentiment of the movie review as negative or positive."),
}

FIN_TASKS = ("fpb", "fiqasa", "tfns", "headlines")


def format_prompt(task: Task, text: str) -> str:
    if task.instruction is None:
        return f"{text}\nAnswer:"
    return f"{task.instruction}\nText: {text}\nAnswer:"


def _flare(repo: str, task: str) -> dict[str, list[dict]]:
    ds = load_dataset(repo)
    out = {}
    for split, name in (("train", "train"), ("valid", "val"), ("test", "test")):
        rows = []
        for r in ds[split]:
            label = TASKS[task].labels.index(r["answer"].strip().lower() if task != "headlines" else r["answer"].strip())
            text = r["query"] if task == "headlines" else r["text"]
            rows.append({"id": r["id"], "text": text.strip(), "label": label})
        out[name] = rows
    return out


def _tfns(seed: int) -> dict[str, list[dict]]:
    # original labels: 0 bearish, 1 bullish, 2 neutral
    remap = {0: 0, 1: 2, 2: 1}
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    train = [{"id": f"tfns-tr{i}", "text": r["text"].strip(), "label": remap[int(r["label"])]} for i, r in enumerate(ds["train"])]
    test = [{"id": f"tfns-te{i}", "text": r["text"].strip(), "label": remap[int(r["label"])]} for i, r in enumerate(ds["validation"])]
    # the public release has no test split: the official validation set becomes our test set
    # and a stratified 10% of train becomes validation
    train, val = stratified_split(train, 0.1, seed)
    return {"train": train, "val": val, "test": test}


def _sst2() -> dict[str, list[dict]]:
    ds = load_dataset("stanfordnlp/sst2")
    rows = [{"id": f"sst2-{r['idx']}", "text": r["sentence"].strip(), "label": int(r["label"])} for r in ds["validation"]]
    return {"test": rows}


def headline_of(query: str) -> str:
    return query.split("\nText:", 1)[-1].strip().lower()


def _dedupe_headlines(splits: dict[str, list[dict]]) -> dict[str, list[dict]]:
    # FLARE asks several questions about the same headline; a headline seen in
    # training must not reappear in validation or test under a different question
    test = {headline_of(r["text"]) for r in splits["test"]}
    val = [r for r in splits["val"] if headline_of(r["text"]) not in test]
    held = test | {headline_of(r["text"]) for r in val}
    train = [r for r in splits["train"] if headline_of(r["text"]) not in held]
    return {"train": train, "val": val, "test": splits["test"]}


def stratified_split(rows: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = {}
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


def subsample(rows: list[dict], n: int | None, seed: int) -> list[dict]:
    if n is None or len(rows) <= n:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)


def build_splits(seed: int = 42, caps: dict[str, dict[str, int]] | None = None) -> dict[str, dict[str, list[dict]]]:
    caps = caps or {}
    raw = {
        "fpb": _flare("ChanceFocus/flare-fpb", "fpb"),
        "fiqasa": _flare("ChanceFocus/flare-fiqasa", "fiqasa"),
        "tfns": _tfns(seed),
        "headlines": _flare("ChanceFocus/flare-headlines", "headlines"),
        "sst2": _sst2(),
    }
    raw["headlines"] = _dedupe_headlines(raw["headlines"])
    for task, splits in raw.items():
        for split in splits:
            splits[split] = subsample(splits[split], caps.get(task, {}).get(split), seed)
    return raw


def save_splits(splits: dict[str, dict[str, list[dict]]], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for task, parts in splits.items():
        for split, rows in parts.items():
            with open(root / f"{task}_{split}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_split(root: Path, task: str, split: str) -> list[dict]:
    with open(Path(root) / f"{task}_{split}.jsonl") as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        r["task"] = task
    return rows


class Collator:
    """Tokenises prompts with right padding and records the last real position."""

    def __init__(self, tokenizer, max_length: int = 192):
        self.tok = tokenizer
        self.max_length = max_length
        self.label_ids = {name: verbalizer_ids(tokenizer, t) for name, t in TASKS.items()}

    def __call__(self, batch: list[dict]) -> dict:
        prompts = [format_prompt(TASKS[r["task"]], r["text"]) for r in batch]
        # truncate the text from the left side of the prompt so that "Answer:" is never cut off
        enc = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        enc = [ids[-self.max_length:] for ids in enc]
        length = max(len(ids) for ids in enc)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        input_ids = torch.full((len(enc), length), pad, dtype=torch.long)
        attention_mask = torch.zeros((len(enc), length), dtype=torch.long)
        for i, ids in enumerate(enc):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attention_mask[i, : len(ids)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "last_index": torch.tensor([len(ids) - 1 for ids in enc]),
            "labels": torch.tensor([r["label"] for r in batch]),
            "tasks": [r["task"] for r in batch],
        }


def verbalizer_ids(tokenizer, task: Task) -> list[int]:
    ids = []
    for word in task.labels:
        tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        ids.append(tokens[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"verbalizer tokens collide for task {task.name}: {ids}")
    return ids
