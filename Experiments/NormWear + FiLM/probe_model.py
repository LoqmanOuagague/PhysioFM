"""Linear/MLP probe head on top of frozen NormWear embeddings, with an
optional FiLM conditioning stage in between.

Pipeline: NormWear (frozen) -> per-channel CLS embeddings (B, C, 768)
  -> [optional] LearnableBaselineSelector(candidate baseline windows) -> baseline_embed (B, C, 768)
  -> [optional] FiLM(embed_c, baseline_embed_c) per channel c
  -> flatten channels -> MLP classifier head (the only trainable parts,
     besides FiLM and the baseline selector itself when enabled).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from baseline_selector import LearnableBaselineSelector
from film import FiLMLayer
from normwear_loader import EMBED_DIM


class NormWearFiLMProbe(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        use_film: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        max_baseline_windows: int | None = None,
        window_seconds: float = 6.0,
        selector_temperature: float = 1e-1,
    ):
        super().__init__()
        self.use_film = use_film
        if use_film:
            assert max_baseline_windows is not None, "use_film=True requires max_baseline_windows (the user-specified r_minutes upper bound)"
            self.baseline_selector = LearnableBaselineSelector(max_baseline_windows, window_seconds, selector_temperature)
            self.film = FiLMLayer(feature_dim=EMBED_DIM, cond_dim=EMBED_DIM)
        else:
            self.baseline_selector = None
            self.film = None

        in_dim = num_channels * EMBED_DIM
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        embed: torch.Tensor,
        baseline_seq: torch.Tensor | None = None,
        baseline_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """embed: (B, C, 768) per-channel window embedding.
        baseline_seq: (B, max_windows, C, 768) that window's subject's
            candidate baseline window embeddings, earliest-first, zero-padded.
        baseline_mask: (B, max_windows) bool, True where that slot is a real
            (non-padded) window. Both required iff use_film=True."""
        if self.use_film:
            assert baseline_seq is not None and baseline_mask is not None, "use_film=True requires baseline_seq and baseline_mask"
            baseline_embed = self.baseline_selector(baseline_seq, baseline_mask)  # (B, C, 768)
            embed = self.film(embed, baseline_embed)
        return self.classifier(embed)

    def effective_baseline_minutes(self) -> float | None:
        return self.baseline_selector.effective_minutes() if self.use_film else None
