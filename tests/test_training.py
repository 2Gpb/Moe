"""End-to-end smoke test: a tiny random Granite-MoE trained for a few steps on CPU."""

import json

import pytest
import torch
from transformers import AutoTokenizer, GraniteMoeConfig, GraniteMoeForCausalLM

from finmoe.build import load_trained
from finmoe.training import train

TOKENIZER = "ibm-granite/granite-3.1-1b-a400m-base"


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
    except OSError:
        pytest.skip("tokenizer not available offline")
    root = tmp_path_factory.mktemp("tiny")
    cfg = GraniteMoeConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2,
    )
    torch.manual_seed(0)
    GraniteMoeForCausalLM(cfg).save_pretrained(root / "model")
    tok.save_pretrained(root / "model")

    data = root / "splits"
    data.mkdir()
    for task, n_labels in (("fpb", 3), ("headlines", 2)):
        for split in ("train", "val", "test"):
            with open(data / f"{task}_{split}.jsonl", "w") as f:
                for i in range(12):
                    f.write(json.dumps({"id": f"{task}{i}", "text": f"example {i} profit rose", "label": i % n_labels}) + "\n")

    torch.save({"selection_ratio": torch.full((2, 4), 0.25), "mean_prob": torch.full((2, 4), 0.25)}, root / "stats.pt")
    return root


def make_cfg(tiny_dir, adapt):
    return {
        "model": str(tiny_dir / "model"), "seed": 0, "device": "cpu", "dtype": "float32",
        "data": {"root": str(tiny_dir / "splits"), "tasks": ["fpb", "headlines"], "max_length": 32},
        "optim": {"epochs": 1, "batch_size": 4, "eval_batch_size": 8, "lr": 1e-3, "log_every": 1, "eval_every": 3},
        "adapt": adapt,
    }


@pytest.mark.parametrize(
    "adapt",
    [
        {"attn_lora": {"rank": 2, "alpha": 4}},
        {"expert_lora": {"rank": 2}, "router": True, "aux_loss_coef": 0.01},
        {"esft": {"stats": "STATS", "score": "selection_ratio", "threshold": 0.5, "control": "random"}},
    ],
)
def test_train_and_reload(tiny_dir, tmp_path, adapt):
    if "esft" in adapt:
        adapt["esft"]["stats"] = str(tiny_dir / "stats.pt")
    summary = train(make_cfg(tiny_dir, adapt), tmp_path / "run")
    assert summary["trainable_params"] > 0
    assert set(summary["test"]) >= {"fpb", "headlines", "mean"}
    assert (tmp_path / "run" / "routing_test.pt").exists()

    model, _, _ = load_trained(tmp_path / "run", torch.device("cpu"), torch.float32)
    saved = torch.load(tmp_path / "run" / "adapter.pt")
    params = dict(model.named_parameters())
    for name, value in saved.items():
        torch.testing.assert_close(params[name].detach(), value)
