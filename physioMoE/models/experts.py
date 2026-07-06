"""Expert heads: each maps the pooled NormWear embedding to NASA-TLX scores.

Every expert is a small independent MLP. They all read the same
physiological embedding but can specialize (e.g. one expert might learn to
weigh heart-rate-variability-like features more for "effort", another might
be more sensitive to EDA-like features for "frustration") because the
router learns to route different samples to different experts.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpertBank(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim, dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) -> (B, num_experts, output_dim)."""
        return torch.stack([expert(x) for expert in self.experts], dim=1)

    @staticmethod
    def combine(expert_outputs: torch.Tensor, gate_weights: torch.Tensor) -> torch.Tensor:
        """expert_outputs: (B, num_experts, output_dim), gate_weights: (B, num_experts) -> (B, output_dim)."""
        return torch.einsum("beo,be->bo", expert_outputs, gate_weights)
