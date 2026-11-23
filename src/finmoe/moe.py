"""Model-agnostic access to the sparse MoE blocks of a Hugging Face model.

Recent transformers versions implement OLMoE, Granite-MoE, Qwen-MoE, Mixtral, ...
with the same two-part layout: a top-k router that returns
``(router_logits, top_k_weights, top_k_index)`` and an experts module that keeps
all expert weights in two 3D tensors, ``gate_up_proj`` [E, 2I, H] and
``down_proj`` [E, H, I]. Everything in this project relies only on that layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MoELayer:
    index: int
    block: nn.Module
    router: nn.Module
    experts: nn.Module
    experts_name: str  # attribute name of the experts module inside the block

    @property
    def num_experts(self) -> int:
        return self.router.weight.shape[0]

    @property
    def top_k(self) -> int:
        return self.router.top_k


def _is_experts(module: nn.Module) -> bool:
    return isinstance(getattr(module, "gate_up_proj", None), nn.Parameter) and isinstance(
        getattr(module, "down_proj", None), nn.Parameter
    )


def find_moe_layers(model: nn.Module, required: bool = True) -> list[MoELayer]:
    layers = []
    for block in model.modules():
        experts_name = next((n for n, m in block.named_children() if _is_experts(m) or _is_adapted(m)), None)
        if experts_name is None:
            continue
        router = next(
            (m for n, m in block.named_children() if n != experts_name and hasattr(m, "top_k") and hasattr(m, "weight")),
            None,
        )
        if router is None:
            continue
        layers.append(MoELayer(len(layers), block, router, getattr(block, experts_name), experts_name))
    if not layers and required:
        raise ValueError("no sparse MoE blocks found in the model")
    return layers


def _is_adapted(module: nn.Module) -> bool:
    return hasattr(module, "base") and _is_experts(module.base)


def unpack_router_output(output: tuple, num_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (router_logits [N, E], top_k_index [N, k]).

    The order of the router outputs changed between transformers releases
    (5.20 returns index first, 5.24 logits first), so the tensors are told
    apart by their content: the index tensor is integer, the logits have E columns.
    """
    index = next(t for t in output if not t.is_floating_point())
    logits = next(t for t in output if t.is_floating_point() and t.shape[-1] == num_experts)
    return logits, index


class RoutingRecorder:
    """Accumulates per-layer routing statistics through forward hooks on the routers.

    Set ``recorder.token_mask`` (bool, [batch*seq]) before each forward pass so that
    padding positions are ignored.
    """

    def __init__(self, layers: list[MoELayer]):
        self.layers = layers
        self.token_mask: torch.Tensor | None = None
        self.reset()
        self._handles = [l.router.register_forward_hook(self._hook(l.index)) for l in layers]

    def reset(self) -> None:
        n_layers, n_exp = len(self.layers), self.layers[0].num_experts
        self.counts = torch.zeros(n_layers, n_exp, dtype=torch.float64)  # times an expert is in the top-k
        self.probs = torch.zeros(n_layers, n_exp, dtype=torch.float64)  # summed softmax affinity
        self.entropy = torch.zeros(n_layers, dtype=torch.float64)  # summed router entropy
        self.tokens = 0

    def _hook(self, idx: int):
        n_exp = self.layers[idx].num_experts

        def hook(module, inputs, output):
            logits, top_idx = unpack_router_output(output, n_exp)
            with torch.no_grad():
                if self.token_mask is not None:
                    mask = self.token_mask.to(logits.device)
                    logits, top_idx = logits[mask], top_idx[mask]
                p = logits.float().softmax(-1)
                self.probs[idx] += p.sum(0).cpu().double()
                self.entropy[idx] += (-(p * p.clamp_min(1e-12).log()).sum(-1)).sum().cpu().double()
                self.counts[idx] += torch.bincount(top_idx.flatten(), minlength=p.shape[-1]).cpu().double()
                if idx == 0:
                    self.tokens += logits.shape[0]

        return hook

    def summary(self) -> dict[str, torch.Tensor]:
        n = max(self.tokens, 1)
        top_k = self.layers[0].top_k
        return {
            "selection_ratio": self.counts / (n * top_k),  # rows sum to 1
            "mean_prob": self.probs / n,  # rows sum to 1
            "entropy": self.entropy / n,
            "tokens": torch.tensor(self.tokens),
        }

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

