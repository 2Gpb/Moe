"""Tasks, prompts and batching.

Every task is solved as next-token prediction: the prompt ends with "Answer:"
and we compare the logits of one label word per class.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

SENTIMENT = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class Task:
    name: str
    labels: tuple[str, ...]
    instruction: str | None  # headlines already contain the question


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


def subsample(rows: list[dict], n: int | None, seed: int) -> list[dict]:
    if n is None or len(rows) <= n:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)


def load_split(root: Path, task: str, split: str) -> list[dict]:
    with open(Path(root) / f"{task}_{split}.jsonl") as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        r["task"] = task
    return rows


class Collator:
    """Tokenises prompts with right padding and records the last real position."""

    def __init__(self, tokenizer, max_length: int = 192, demos: dict[str, list[dict]] | None = None):
        self.tok = tokenizer
        self.max_length = max_length
        self.demos = demos or {}
        # pad with an ordinary token, not <|endoftext|>: in Granite it gets huge
        # activations, padded positions overflow in fp16 and the gradients become NaN
        self.pad_id = tokenizer(" .", add_special_tokens=False)["input_ids"][-1]

    def prompt(self, row: dict) -> str:
        task = TASKS[row["task"]]
        shots = "".join(
            f"{format_prompt(task, d['text'])} {task.labels[d['label']]}\n\n" for d in self.demos.get(row["task"], [])
        )
        return shots + format_prompt(task, row["text"])

    def __call__(self, batch: list[dict]) -> dict:
        prompts = [self.prompt(r) for r in batch]
        # cut long prompts from the left so "Answer:" stays at the end
        enc = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        enc = [ids[-self.max_length:] for ids in enc]
        length = max(len(ids) for ids in enc)
        input_ids = torch.full((len(enc), length), self.pad_id, dtype=torch.long)
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


def pick_demos(rows: list[dict], per_label: int, seed: int) -> list[dict]:
    """per_label examples of each class, interleaved."""
    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    picked = {label: rng.sample(group, per_label) for label, group in sorted(by_label.items())}
    demos = [picked[label][i] for i in range(per_label) for label in picked]
    return demos


def verbalizer_ids(tokenizer, task: Task) -> list[int]:
    ids = []
    for word in task.labels:
        tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        ids.append(tokens[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"verbalizer tokens collide for task {task.name}: {ids}")
    return ids
