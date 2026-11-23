"""Model loading and verbalizer-based classification head."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from finmoe.data import TASKS, verbalizer_ids


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16  # e.g. T4 on Kaggle / Colab
    if device.type == "cpu":
        return torch.float32
    return torch.bfloat16


def load_model(name: str, device: torch.device, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.config.use_cache = False
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


class VerbalizerHead:
    """Maps the LM's next-token distribution onto the label words of each task.

    Only the hidden state at the last prompt position goes through the LM head,
    which avoids materialising [batch, seq, vocab] logits.
    """

    def __init__(self, model, tokenizer, max_labels: int = 3):
        self.model = model
        self.scale = getattr(model.config, "logits_scaling", 1.0)
        self.max_labels = max_labels
        self.label_ids = {name: verbalizer_ids(tokenizer, task) for name, task in TASKS.items()}

    def __call__(self, batch: dict) -> torch.Tensor:
        """Returns [batch, max_labels] class logits; missing classes are -inf."""
        body, lm_head = self.model.model, self.model.lm_head
        device = lm_head.weight.device
        hidden = body(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).last_hidden_state
        rows = torch.arange(hidden.shape[0], device=device)
        last = hidden[rows, batch["last_index"].to(device)]
        vocab_logits = lm_head(last).float() / self.scale

        out = torch.full((hidden.shape[0], self.max_labels), float("-inf"), device=device)
        for i, task in enumerate(batch["tasks"]):
            ids = self.label_ids[task]
            out[i, : len(ids)] = vocab_logits[i, ids]
        return out
