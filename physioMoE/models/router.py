"""Gating network: decides, per sample, how much each expert should count.

The router is conditioned on both the encoded task text and the pooled
NormWear embedding, so routing can depend on *what the person is doing*
(e.g. a monitoring task vs. a manual task might rely on different
physiological experts) as well as on the physiological state itself. Experts
themselves only ever see the physiological embedding (see ``experts.py``) —
the router purely decides the mixing weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 128,
        top_k: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if top_k is not None and not (1 <= top_k <= num_experts):
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")

        self.num_experts = num_experts
        self.top_k = top_k
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, fused_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """fused_features: (B, input_dim) -> (gate_weights, logits), both (B, num_experts)."""
        logits = self.net(fused_features)

        if self.top_k is None or self.top_k == self.num_experts:
            gate_weights = F.softmax(logits, dim=-1)
            return gate_weights, logits

        top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
        sparse_logits = torch.full_like(logits, float("-inf"))
        sparse_logits.scatter_(-1, top_indices, top_values)
        gate_weights = F.softmax(sparse_logits, dim=-1)
        return gate_weights, logits


def load_balancing_loss(gate_weights: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Squared coefficient-of-variation of per-expert importance across the batch.

    Encourages the router to spread load roughly evenly across experts
    (Shazeer et al., 2017), which keeps every expert receiving gradient
    signal instead of collapsing onto one.
    """
    importance = gate_weights.sum(dim=0)  # (num_experts,)
    cv = importance.std() / (importance.mean() + eps)
    return cv**2
