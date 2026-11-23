import torch

from finmoe.selection import select_by_threshold, select_random_like, select_specific_like


def test_threshold_picks_smallest_covering_set():
    scores = torch.tensor([[0.5, 0.3, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])
    chosen = select_by_threshold(scores, p=0.6)
    assert chosen == {0: [0, 1], 1: [0, 1, 2]}


def test_controls_keep_the_budget():
    scores = torch.softmax(torch.randn(3, 8), -1)
    ref = select_by_threshold(scores, 0.4)
    general = torch.softmax(torch.randn(3, 8), -1)
    for other in (select_random_like(ref, 8, seed=0), select_specific_like(ref, scores, general)):
        assert {l: len(v) for l, v in other.items()} == {l: len(v) for l, v in ref.items()}


def test_specific_prefers_domain_heavy_experts():
    fin = torch.tensor([[0.4, 0.4, 0.1, 0.1]])
    general = torch.tensor([[0.4, 0.5, 0.01, 0.09]])
    # expert 2 is used 10x more on the domain than on general text
    assert select_specific_like({0: [0]}, fin, general) == {0: [2]}
