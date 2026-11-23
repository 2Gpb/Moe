import pytest
import torch
from transformers import GraniteMoeConfig, GraniteMoeForCausalLM

from finmoe.adapters import AdaptedExperts, count_parameters, wrap_experts
from finmoe.moe import RoutingRecorder, find_moe_layers


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    cfg = GraniteMoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=6,
        num_experts_per_tok=2,
    )
    model = GraniteMoeForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def test_finds_all_layers(tiny_model):
    layers = find_moe_layers(tiny_model)
    assert len(layers) == 2
    assert layers[0].num_experts == 6 and layers[0].top_k == 2


def test_wrapper_without_adapters_is_identity(tiny_model):
    x = torch.randint(0, 128, (3, 7))
    ref = tiny_model(x).logits
    wrap_experts(tiny_model)
    out = tiny_model(x).logits
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_zero_initialised_adapters_do_not_change_output(tiny_model):
    x = torch.randint(0, 128, (2, 5))
    ref = tiny_model(x).logits
    wrap_experts(tiny_model, lora_rank=4, delta_experts={0: [1, 3], 1: [0]})
    torch.testing.assert_close(tiny_model(x).logits, ref, atol=1e-5, rtol=1e-5)


def test_only_adapters_receive_gradients(tiny_model):
    wrap_experts(tiny_model, lora_rank=2, delta_experts={0: [2]})
    trainable = {n for n, p in tiny_model.named_parameters() if p.requires_grad}
    assert all("lora_" in n or "delta_" in n for n in trainable)
    tiny_model(torch.randint(0, 128, (2, 5))).logits.sum().backward()
    for name, p in tiny_model.named_parameters():
        if "lora_" in name and name.endswith("_b"):
            assert p.grad is not None and p.grad.abs().sum() > 0


def test_delta_only_affects_selected_expert(tiny_model):
    layer = find_moe_layers(tiny_model)[0]
    wrapped = AdaptedExperts(layer.experts, delta_experts=[4])
    h = torch.randn(5, 32)
    weights = torch.full((5, 2), 0.5)
    not_selected = torch.tensor([[0, 1]] * 5)
    selected = torch.tensor([[4, 1]] * 5)
    before = (wrapped(h, not_selected, weights), wrapped(h, selected, weights))
    with torch.no_grad():
        wrapped.delta_gu.normal_()
    torch.testing.assert_close(wrapped(h, not_selected, weights), before[0])
    assert not torch.allclose(wrapped(h, selected, weights), before[1])


def test_parameter_count(tiny_model):
    wrap_experts(tiny_model, lora_rank=2)
    trainable, _ = count_parameters(tiny_model)
    # per expert: gate_up A (2x32) + B (32x2) + down A (2x16) + B (32x2)
    assert trainable == 2 * 6 * (2 * 32 + 32 * 2 + 2 * 16 + 32 * 2)


def test_routing_recorder_respects_mask(tiny_model):
    layers = find_moe_layers(tiny_model)
    rec = RoutingRecorder(layers)
    x = torch.randint(0, 128, (2, 4))
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]).bool()
    rec.token_mask = mask.flatten()
    tiny_model(x)
    s = rec.summary()
    assert rec.tokens == 5
    torch.testing.assert_close(s["selection_ratio"].sum(-1), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(s["mean_prob"].sum(-1), torch.ones(2, dtype=torch.float64))



def test_router_output_order_does_not_matter():
    from finmoe.moe import unpack_router_output

    logits, weights, index = torch.randn(5, 6), torch.rand(5, 2), torch.randint(0, 6, (5, 2))
    for output in ((logits, weights, index), (index, weights, logits)):
        got_logits, got_index = unpack_router_output(output, num_experts=6)
        assert got_logits is logits and got_index is index
