"""Learns how much of a subject's baseline recording to use for FiLM
conditioning, instead of fixing `r_minutes` up front.

The user specifies only an upper bound (`max_minutes`, converted here to
`max_windows` = the number of leading baseline windows that get encoded and
made available as candidates). During training, a single scalar parameter
decides how many of those candidate windows to average into the baseline
reference embedding, learned end-to-end by gradient descent on the
classification loss -- exactly like any other model parameter.

Picking a whole number of windows isn't differentiable, so the cutoff is
relaxed into a soft gate along the window index (a sigmoid step centered at
the learned effective window count `n_eff`, `temperature` wide): windows well
before the cutoff get weight ~1, windows well past it get weight ~0, and the
few windows straddling the boundary get partial weight. Gradients flow
through `n_eff` via that gate, so training can push the effective duration up
or down (never past `max_minutes`, since `n_eff = max_windows * sigmoid(.)`
saturates there) to whatever minimizes the loss.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LearnableBaselineSelector(nn.Module):
    def __init__(self, max_windows: int, window_seconds: float, temperature: float = 1.0, init_frac: float = 0.9):
        super().__init__()
        assert max_windows > 0
        self.max_windows = max_windows
        self.window_seconds = window_seconds
        self.temperature = temperature

        init_frac = min(max(init_frac, 1e-3), 1 - 1e-3)
        init_logit = math.log(init_frac / (1 - init_frac))
        # a single global scalar: the model settles on one effective
        # baseline duration rather than a per-subject or per-channel one.
        self.raw_r = nn.Parameter(torch.tensor(float(init_logit)))

    def effective_n_windows(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_r) * self.max_windows

    def effective_minutes(self) -> float:
        return float(self.effective_n_windows().item() * self.window_seconds / 60.0)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """seq: (..., max_windows, C, E), a subject's candidate baseline
        window embeddings ordered earliest-first, zero-padded past however
        many that subject actually has.
        mask: (..., max_windows) bool, True where that slot is a real
        (non-padded) window.
        -> (..., C, E): the learned soft-averaged baseline reference.
        """
        n_eff = self.effective_n_windows()
        idx = torch.arange(self.max_windows, device=seq.device, dtype=seq.dtype)
        gate = torch.sigmoid((n_eff - idx - 0.5) / self.temperature)  # (max_windows,), earliest-first cutoff
        weight = gate * mask.to(seq.dtype)  # (..., max_windows)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("...w,...wce->...ce", weight, seq)
