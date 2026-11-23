"""Parameter-efficient adapters for the MoE blocks.

`AdaptedExperts` replaces a model's experts module and adds two optional kinds
of trainable parameters on top of the frozen expert weights W_e:

* expert LoRA:   W_e + (alpha / r) * B_e A_e   for every expert e (or a subset)
* expert deltas: W_e + D_e                      full-rank, only for selected experts
                                                (expert-specialised fine-tuning, ESFT)

Tokens are sorted by expert and each expert processes its contiguous slice,
which is the same idea as grouped GEMM but written as a plain loop so it runs
unchanged on CUDA, MPS and CPU.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from finmoe.moe import MoELayer, find_moe_layers


class AdaptedExperts(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_experts: list[int] | None = None,
        delta_experts: list[int] | None = None,
    ):
        super().__init__()
        self.base = base
        self.num_experts = base.gate_up_proj.shape[0]
        self.act_fn = base.act_fn
        n_exp, two_i, hidden = base.gate_up_proj.shape
        inter = base.down_proj.shape[-1]

        self.lora_rank = lora_rank
        self.lora_scale = lora_alpha / lora_rank if lora_rank else 0.0
        lora_experts = list(range(n_exp)) if lora_experts is None else sorted(lora_experts)
        self.lora_slot = {e: j for j, e in enumerate(lora_experts)} if lora_rank else {}
        if lora_rank:
            k = len(lora_experts)
            self.lora_gu_a = nn.Parameter(torch.empty(k, lora_rank, hidden))
            self.lora_gu_b = nn.Parameter(torch.zeros(k, two_i, lora_rank))
            self.lora_d_a = nn.Parameter(torch.empty(k, lora_rank, inter))
            self.lora_d_b = nn.Parameter(torch.zeros(k, hidden, lora_rank))
            for a in (self.lora_gu_a, self.lora_d_a):
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))

        delta_experts = sorted(delta_experts or [])
        self.delta_slot = {e: j for j, e in enumerate(delta_experts)}
        if delta_experts:
            k = len(delta_experts)
            self.delta_gu = nn.Parameter(torch.zeros(k, two_i, hidden))
            self.delta_d = nn.Parameter(torch.zeros(k, hidden, inter))

    def _proj(self, x: torch.Tensor, e: int, weight: torch.Tensor, delta, lora_a, lora_b) -> torch.Tensor:
        y = x @ weight[e].t()
        d = self.delta_slot.get(e)
        if d is not None:
            # the delta stays in fp32: adding it to a bf16 weight first would round small updates away
            y = y + (x.to(delta.dtype) @ delta[d].t()).to(y.dtype)
        l = self.lora_slot.get(e)
        if l is not None:
            xf = x.to(lora_a.dtype)
            y = y + ((xf @ lora_a[l].t()) @ lora_b[l].t() * self.lora_scale).to(y.dtype)
        return y

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        n_tok, top_k = top_k_index.shape
        flat = top_k_index.reshape(-1)
        order = torch.argsort(flat, stable=True)
        token_of = order // top_k
        x = hidden_states[token_of]
        weights = top_k_weights.reshape(-1)[order]
        counts = torch.bincount(flat, minlength=self.num_experts).tolist()

        base = self.base
        lora = (getattr(self, "lora_gu_a", None), getattr(self, "lora_gu_b", None))
        lora_d = (getattr(self, "lora_d_a", None), getattr(self, "lora_d_b", None))
        outs = []
        for e, xe in enumerate(torch.split(x, counts)):
            if xe.shape[0] == 0:
                continue
            gate, up = self._proj(xe, e, base.gate_up_proj, getattr(self, "delta_gu", None), *lora).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            outs.append(self._proj(h, e, base.down_proj, getattr(self, "delta_d", None), *lora_d))
        y = torch.cat(outs) * weights.unsqueeze(-1).to(outs[0].dtype)

        out = torch.zeros(n_tok, hidden_states.shape[-1], device=y.device, dtype=torch.float32)
        out.index_add_(0, token_of, y.float())
        return out.to(hidden_states.dtype)


def wrap_experts(
    model: nn.Module,
    lora_rank: int = 0,
    lora_alpha: float = 16.0,
    delta_experts: dict[int, list[int]] | None = None,
    lora_experts: dict[int, list[int]] | None = None,
    layers: list[int] | None = None,
) -> list[MoELayer]:
    """Swaps every experts module for an `AdaptedExperts` wrapper.

    The trainable tensors are created in float32 on the model's device; the
    frozen expert weights keep their original dtype.
    """
    moe_layers = find_moe_layers(model)
    for layer in moe_layers:
        if layers is not None and layer.index not in layers:
            continue
        base = layer.experts
        wrapped = AdaptedExperts(
            base,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_experts=None if lora_experts is None else lora_experts.get(layer.index, []),
            delta_experts=(delta_experts or {}).get(layer.index, []),
        ).to(base.gate_up_proj.device)
        setattr(layer.block, layer.experts_name, wrapped)
        layer.experts = wrapped
    return moe_layers


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
