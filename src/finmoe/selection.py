"""Which experts to fine-tune (ESFT, Wang et al. 2024) and the controls."""

from __future__ import annotations

import torch


def select_by_threshold(scores: torch.Tensor, p: float) -> dict[int, list[int]]:
    """In every layer take the top experts until their scores add up to p."""
    selected = {}
    for layer, row in enumerate(scores):
        order = torch.argsort(row, descending=True)
        cum = torch.cumsum(row[order], 0)
        k = int((cum < p).sum().item()) + 1
        selected[layer] = sorted(order[:k].tolist())
    return selected


def select_random_like(reference: dict[int, list[int]], num_experts: int, seed: int) -> dict[int, list[int]]:
    """Control condition: the same number of experts per layer, chosen at random."""
    g = torch.Generator().manual_seed(seed)
    return {
        layer: sorted(torch.randperm(num_experts, generator=g)[: len(experts)].tolist())
        for layer, experts in reference.items()
    }


def select_specific_like(reference: dict[int, list[int]], fin: torch.Tensor, general: torch.Tensor) -> dict[int, list[int]]:
    """Same budget, but rank experts by how much more the domain uses them than general text."""
    spec = domain_specificity(fin, general)
    return {
        layer: sorted(torch.argsort(spec[layer], descending=True)[: len(experts)].tolist())
        for layer, experts in reference.items()
    }


def domain_specificity(fin: torch.Tensor, general: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """log-ratio of how often an expert is chosen on financial vs general text."""
    return torch.log((fin + eps) / (general + eps))
