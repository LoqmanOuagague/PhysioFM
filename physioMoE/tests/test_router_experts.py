import torch

from physioMoE.models.experts import ExpertBank
from physioMoE.models.router import Router, load_balancing_loss


def test_router_dense_gate_sums_to_one():
    router = Router(input_dim=16, num_experts=4)
    x = torch.randn(8, 16)
    gate_weights, logits = router(x)

    assert gate_weights.shape == (8, 4)
    assert logits.shape == (8, 4)
    torch.testing.assert_close(gate_weights.sum(dim=-1), torch.ones(8), atol=1e-5, rtol=0)


def test_router_top_k_zeroes_out_non_selected_experts():
    router = Router(input_dim=16, num_experts=6, top_k=2)
    x = torch.randn(5, 16)
    gate_weights, _ = router(x)

    nonzero_per_row = (gate_weights > 0).sum(dim=-1)
    assert torch.all(nonzero_per_row == 2)
    torch.testing.assert_close(gate_weights.sum(dim=-1), torch.ones(5), atol=1e-5, rtol=0)


def test_router_rejects_invalid_top_k():
    import pytest

    with pytest.raises(ValueError):
        Router(input_dim=16, num_experts=4, top_k=5)


def test_load_balancing_loss_penalizes_imbalance():
    balanced = torch.full((8, 4), 0.25)
    imbalanced = torch.zeros(8, 4)
    imbalanced[:, 0] = 1.0

    assert load_balancing_loss(balanced).item() < 1e-6
    assert load_balancing_loss(imbalanced).item() > load_balancing_loss(balanced).item()


def test_expert_bank_combine_matches_manual_weighted_sum():
    bank = ExpertBank(input_dim=8, num_experts=3, hidden_dim=16, output_dim=6)
    x = torch.randn(4, 8)

    expert_outputs = bank(x)
    assert expert_outputs.shape == (4, 3, 6)

    gate_weights = torch.softmax(torch.randn(4, 3), dim=-1)
    combined = bank.combine(expert_outputs, gate_weights)

    manual = sum(gate_weights[:, i : i + 1] * expert_outputs[:, i, :] for i in range(3))
    torch.testing.assert_close(combined, manual)
