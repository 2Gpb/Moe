from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from finmoe.modeling import VerbalizerHead


@torch.no_grad()
def predict(head: VerbalizerHead, loader: DataLoader, recorder=None, desc: str = "eval") -> tuple[list[dict], float]:
    """Returns per-example predictions and the wall-clock time of the pass in seconds."""
    head.model.eval()
    records = []
    start = time.perf_counter()
    for batch in tqdm(loader, desc=desc, leave=False):
        if recorder is not None:
            recorder.token_mask = batch["attention_mask"].flatten().bool()
        logits = head(batch)
        probs = logits.softmax(-1).cpu()
        for i, task in enumerate(batch["tasks"]):
            records.append(
                {
                    "task": task,
                    "label": int(batch["labels"][i]),
                    "pred": int(probs[i].argmax()),
                    "probs": [round(float(x), 5) for x in probs[i]],
                }
            )
    return records, time.perf_counter() - start


def compute_metrics(records: list[dict]) -> dict[str, dict[str, float]]:
    """Accuracy and F1 per task, plus the unweighted mean over financial tasks."""
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    out = {}
    for task, rows in sorted(by_task.items()):
        y = np.array([r["label"] for r in rows])
        p = np.array([r["pred"] for r in rows])
        out[task] = {
            "n": len(rows),
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro")),
            "weighted_f1": float(f1_score(y, p, average="weighted")),
        }
    fin = [m for t, m in out.items() if t != "sst2"]
    if fin:
        out["mean"] = {k: float(np.mean([m[k] for m in fin])) for k in ("accuracy", "macro_f1", "weighted_f1")}
    return out
