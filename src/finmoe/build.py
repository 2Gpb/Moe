"""Builds a model with the adaptation method described by a config dict.

Example config section::

    adapt:
      attn_lora: {rank: 16, alpha: 32, targets: [q_proj, k_proj, v_proj, o_proj]}
      expert_lora: {rank: 8, alpha: 16}
      router: true
      esft: {score: selection_ratio, threshold: 0.2, control: none}

Any subset of the components can be combined; an empty section means the
frozen base model (zero-shot / few-shot evaluation).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from finmoe.adapters import wrap_experts
from finmoe.modeling import load_model
from finmoe.moe import find_moe_layers
from finmoe.selection import select_by_threshold, select_least_used_like, select_random_like, select_specific_like


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.dropout(x).to(self.lora_a.dtype) @ self.lora_a.t()) @ self.lora_b.t()
        return self.base(x) + (delta * self.scale).to(x.dtype)


def add_attention_lora(model: nn.Module, rank: int, alpha: float, targets: list[str], dropout: float = 0.0) -> int:
    replaced = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha, dropout))
                replaced += 1
    return replaced


def _cast_to_weight_dtype(module: nn.Module, args):
    hidden = args[0]
    return (hidden.to(module.weight.dtype),) + tuple(args[1:])


def resolve_esft_experts(esft: dict, num_experts: int, seed: int) -> dict[int, list[int]]:
    stats = torch.load(esft["stats"], weights_only=False)
    scores = stats[esft.get("score", "selection_ratio")]
    chosen = select_by_threshold(scores, esft["threshold"], esft.get("max_per_layer"))
    control = esft.get("control", "none")
    if control == "random":
        chosen = select_random_like(chosen, num_experts, seed)
    elif control == "least_used":
        chosen = select_least_used_like(chosen, scores)
    elif control == "domain_specific":
        chosen = select_specific_like(chosen, stats["selection_ratio"], stats["general_selection_ratio"])
    return chosen


def build_model(cfg: dict, device: torch.device, dtype: torch.dtype, seed: int = 0):
    model, tokenizer = load_model(cfg["model"], device, dtype)
    adapt = cfg.get("adapt") or {}
    info: dict = {}

    if "attn_lora" in adapt:
        a = adapt["attn_lora"]
        info["attn_lora_modules"] = add_attention_lora(
            model, a["rank"], a.get("alpha", 2 * a["rank"]), a.get("targets", ["q_proj", "k_proj", "v_proj", "o_proj"]), a.get("dropout", 0.0)
        )

    if "expert_lora" in adapt or "esft" in adapt:
        num_experts = find_moe_layers(model)[0].num_experts
        delta = None
        if "esft" in adapt:
            delta = resolve_esft_experts(adapt["esft"], num_experts, seed)
            info["esft_experts"] = delta
        e = adapt.get("expert_lora", {})
        wrap_experts(model, lora_rank=e.get("rank", 0), lora_alpha=e.get("alpha", 16.0), delta_experts=delta)

    if adapt.get("router"):
        for layer in find_moe_layers(model):
            layer.router.weight.requires_grad_(True)
            layer.router.register_forward_pre_hook(_cast_to_weight_dtype)

    # keep every trainable tensor in fp32 even if the frozen backbone is bf16/fp16
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model, tokenizer, info


def load_trained(run_dir: str | Path, device: torch.device, dtype: torch.dtype):
    import yaml

    run_dir = Path(run_dir)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    model, tokenizer, info = build_model(cfg, device, dtype, seed=cfg.get("seed", 0))
    state = torch.load(run_dir / "adapter.pt", map_location=device)
    missing = set(state) - {n for n, _ in model.named_parameters()}
    if missing:
        raise KeyError(f"adapter tensors without a matching parameter: {sorted(missing)[:5]}")
    model.load_state_dict(state, strict=False)
    return model, tokenizer, cfg
