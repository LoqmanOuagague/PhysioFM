import torch
import torch.nn as nn

from physioMoE.config import ModelConfig
from physioMoE.models.physio_moe import PhysioMoE


class FakeTextEncoder(nn.Module):
    """Stands in for a real HF text encoder so tests run without network access."""

    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Embedding(1000, output_dim)

    def forward(self, texts: list[str]) -> torch.Tensor:
        idx = torch.tensor([abs(hash(t)) % 1000 for t in texts])
        return self.embed(idx)


class FakeNormWearEncoder(nn.Module):
    """Stands in for the real NormWear foundation model in tests."""

    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(1, output_dim)

    def forward(self, signals: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        pooled = signals.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        return self.proj(pooled)


def build_test_model(**overrides) -> PhysioMoE:
    config = ModelConfig(hidden_dim=32, num_experts=4, expert_hidden_dim=16, **overrides)
    return PhysioMoE(
        config,
        text_encoder=FakeTextEncoder(),
        normwear_encoder=FakeNormWearEncoder(),
    )


def test_forward_output_shape_and_range():
    model = build_test_model()
    texts = ["do task A", "do task B", "do task A"]
    signals = torch.randn(3, 4, 128)  # (B, C, T)

    out = model(texts, signals)

    assert out["output"].shape == (3, 6)
    assert torch.all(out["output"] >= 0) and torch.all(out["output"] <= model.config.tlx_scale)
    assert out["gate_weights"].shape == (3, 4)
    torch.testing.assert_close(out["gate_weights"].sum(dim=-1), torch.ones(3), atol=1e-4, rtol=0)


def test_compute_loss_and_backward_updates_router_and_experts():
    model = build_test_model()
    texts = ["do task A", "do task B"]
    signals = torch.randn(2, 4, 64)
    targets = torch.rand(2, 6) * 100

    preds = model(texts, signals)
    losses = model.compute_loss(preds, targets)
    assert torch.isfinite(losses["loss"])

    losses["loss"].backward()
    router_grad_norm = sum(p.grad.abs().sum() for p in model.router.parameters() if p.grad is not None)
    expert_grad_norm = sum(p.grad.abs().sum() for p in model.experts.parameters() if p.grad is not None)

    assert router_grad_norm > 0
    assert expert_grad_norm > 0


def test_top_k_routing_end_to_end():
    model = build_test_model(top_k=2)
    texts = ["task"] * 2
    signals = torch.randn(2, 4, 32)

    out = model(texts, signals)
    assert (out["gate_weights"] > 0).sum(dim=-1).tolist() == [2, 2]
