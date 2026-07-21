"""FiLM (Feature-wise Linear Modulation, Perez et al. 2017) conditioning layer.

Applied per-channel: each of the 10 WESAD sensor channels' pooled NormWear
embedding is scaled/shifted by a (gamma, beta) pair predicted from that same
channel's personal-baseline embedding (the subject's own resting-state
signal, encoded with the same frozen NormWear backbone). One shared MLP
predicts (gamma, beta) for every channel, conditioned on that channel's
baseline embedding, so the module's size doesn't grow with the number of
channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    def __init__(self, feature_dim: int, cond_dim: int | None = None, hidden_dim: int = 256):
        super().__init__()
        cond_dim = cond_dim or feature_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * feature_dim),
        )
        # zero-init the last layer so gamma starts at 1 and beta at 0, i.e.
        # FiLM starts out as the identity function and has to learn its way
        # away from that rather than disrupting the frozen embedding immediately.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x, cond: (..., feature_dim) / (..., cond_dim), same leading shape."""
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        return (1.0 + gamma) * x + beta
