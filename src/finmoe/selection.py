"""Choosing which experts to fine-tune from routing statistics (ESFT, Wang et al. 2024)."""

from __future__ import annotations

import torch


def select_by_threshold(scores: torch.Tensor, p: float, max_per_layer: int | None = None) -> dict[int, list[int]]:
    """For every layer keep the smallest set of top-scoring experts whose
    cumulative relevance reaches ``p`` (scores of a layer sum to one)."""
    selected = {}
    for layer, row in enumerate(scores):
        order = torch.argsort(row, descending=True)
        cum = torch.cumsum(row[order], 0)
        k = int((cum < p).sum().item()) + 1
        if max_per_layer is not None:
            k = min(k, max_per_layer)
        selected[layer] = sorted(order[:k].tolist())
    return selected


def select_random_like(reference: dict[int, list[int]], num_experts: int, seed: int) -> dict[int, list[int]]:
    """Control condition: the same number of experts per layer, chosen at random."""
    g = torch.Generator().manual_seed(seed)
    return {
        layer: sorted(torch.randperm(num_experts, generator=g)[: len(experts)].tolist())
        for layer, experts in reference.items()
    }


def select_least_used_like(reference: dict[int, list[int]], scores: torch.Tensor) -> dict[int, list[int]]:
    """Adversarial control: the same budget spent on the least relevant experts."""
    return {
        layer: sorted(torch.argsort(scores[layer])[: len(experts)].tolist())
        for layer, experts in reference.items()
    }


def domain_specificity(fin: torch.Tensor, general: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """log-ratio of how often an expert is chosen on financial vs general text."""
    return torch.log((fin + eps) / (general + eps))
