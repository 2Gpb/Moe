"""Multi-task fine-tuning loop shared by every adaptation method."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from finmoe.adapters import adapter_state_dict, count_parameters
from finmoe.build import build_model
from finmoe.data import Collator, load_split
from finmoe.evaluation import compute_metrics, predict
from finmoe.modeling import VerbalizerHead, pick_device, pick_dtype
from finmoe.moe import RoutingRecorder, find_moe_layers, load_balancing_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_rows(root: str, tasks: list[str], split: str) -> list[dict]:
    return [r for t in tasks for r in load_split(root, t, split)]


def param_groups(model: torch.nn.Module, optim: dict) -> list[dict]:
    """One group per learning rate; ``lr_overrides`` maps a name fragment to its lr."""
    overrides = optim.get("lr_overrides", {})
    groups: dict[float, list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr = next((v for k, v in overrides.items() if k in name), optim["lr"])
        groups.setdefault(float(lr), []).append(p)
    return [{"params": ps, "lr": lr, "initial_lr": lr} for lr, ps in groups.items()]


def lr_factor(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def peak_memory_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return float("nan")


def release_cache(device: torch.device) -> None:
    # batches have different lengths, so the MPS/CUDA caching allocators keep growing otherwise
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def evaluate_split(head, rows, collator, batch_size, recorder=None, desc="eval"):
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=collator)
    records, seconds = predict(head, loader, recorder=recorder, desc=desc)
    release_cache(head.model.lm_head.weight.device)
    return records, compute_metrics(records), seconds


def train(cfg: dict, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 0)
    seed_everything(seed)
    device = pick_device(cfg.get("device", "auto"))
    dtype = pick_dtype(device, cfg.get("dtype", "auto"))

    model, tokenizer, info = build_model(cfg, device, dtype, seed=seed)
    head = VerbalizerHead(model, tokenizer)
    trainable, total = count_parameters(model)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)
    if "esft_experts" in info:
        json.dump(info["esft_experts"], open(run_dir / "esft_experts.json", "w"))

    data, optim = cfg["data"], cfg["optim"]
    train_rows = load_rows(data["root"], data["tasks"], "train")
    val_rows = load_rows(data["root"], data["tasks"], "val")
    collator = Collator(tokenizer, max_length=data.get("max_length", 192))
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_rows, batch_size=optim["batch_size"], shuffle=True, collate_fn=collator, generator=g, drop_last=True)

    accum = optim.get("grad_accum", 1)
    total_steps = optim["epochs"] * len(loader) // accum
    warmup = int(optim.get("warmup_ratio", 0.06) * total_steps)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups(model, optim), weight_decay=optim.get("weight_decay", 0.0))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16 and device.type == "cuda")

    aux_coef = cfg.get("adapt", {}).get("aux_loss_coef", 0.0)
    layers = find_moe_layers(model, required=aux_coef > 0)
    recorder = RoutingRecorder(layers, keep_logits=True) if aux_coef > 0 else None
    if recorder is not None:
        recorder.enabled = False

    eval_every = optim.get("eval_every", max(1, total_steps // 2))
    log = open(run_dir / "log.jsonl", "w")
    best = {"score": -1.0, "step": -1}
    step, t0 = 0, time.perf_counter()
    print(f"trainable {trainable:,} / {total:,} params ({100 * trainable / total:.3f}%), {total_steps} steps", flush=True)

    model.train()
    for epoch in range(optim["epochs"]):
        for i, batch in enumerate(loader):
            if recorder is not None:
                recorder.token_mask = batch["attention_mask"].flatten().bool()
            logits = head(batch)
            loss = F.cross_entropy(logits, batch["labels"].to(logits.device))
            aux = torch.zeros((), device=logits.device)
            if recorder is not None:
                aux = load_balancing_loss(recorder.pop_logits(), layers[0].top_k)
            scaler.scale((loss + aux_coef * aux) / accum).backward()
            if (i + 1) % accum:
                continue

            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lr_factor(step, total_steps, warmup)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, optim.get("max_grad_norm", 1.0))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % optim.get("log_every", 20) == 0:
                release_cache(device)
                entry = {"step": step, "loss": loss.item(), "aux": aux.item(), "grad_norm": float(grad_norm),
                         "lr": optimizer.param_groups[0]["lr"], "elapsed": time.perf_counter() - t0}
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(entry, flush=True)

            if step % eval_every == 0 or step == total_steps:
                if recorder is not None:
                    recorder.keep_logits = False
                _, metrics, _ = evaluate_split(head, val_rows, collator, optim.get("eval_batch_size", 32), desc="val")
                if recorder is not None:
                    recorder.keep_logits = True
                score = metrics["mean"]["macro_f1"]
                log.write(json.dumps({"step": step, "val": metrics}) + "\n")
                log.flush()
                print(f"step {step} val macro-F1 {score:.4f}", flush=True)
                if score > best["score"]:
                    # only the trainable tensors are saved, the frozen backbone is reloaded from the hub
                    best = {"score": score, "step": step, "val": metrics}
                    torch.save(adapter_state_dict(model), run_dir / "adapter.pt")
                model.train()

    train_seconds = time.perf_counter() - t0
    if recorder is not None:
        recorder.remove()
    best_state = torch.load(run_dir / "adapter.pt", map_location=device)
    model.load_state_dict(best_state, strict=False)
    summary = test_and_profile(cfg, model, head, collator, run_dir)
    summary.update({
        "trainable_params": trainable,
        "total_params": total,
        "train_seconds": train_seconds,
        "train_steps": total_steps,
        "best_step": best["step"],
        "val": best["val"],
        "peak_memory_gb": peak_memory_gb(device),
        "device": device.type,
        "dtype": str(dtype),
    })
    json.dump(summary, open(run_dir / "summary.json", "w"), indent=2)
    return summary


def test_and_profile(cfg, model, head, collator, run_dir: Path) -> dict:
    """Test-set metrics with routing statistics recorded on the same pass."""
    data = cfg["data"]
    tasks = data.get("test_tasks", data["tasks"])
    rows = load_rows(data["root"], tasks, "test")
    layers = find_moe_layers(model, required=False)
    recorder = RoutingRecorder(layers) if layers else None
    records, metrics, seconds = evaluate_split(head, rows, collator, cfg["optim"].get("eval_batch_size", 32), recorder, desc="test")
    with open(run_dir / "test_predictions.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    if recorder is not None:
        recorder.remove()
        torch.save(recorder.summary(), run_dir / "routing_test.pt")
    return {"test": metrics, "test_ms_per_example": 1000 * seconds / len(records)}
