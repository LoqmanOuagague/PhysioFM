"""Linear/MLP probe head on top of frozen NormWear embeddings, with an
optional FiLM conditioning stage in between.

Pipeline: NormWear (frozen) -> per-channel CLS embeddings (B, C, 768)
  -> [optional] LearnableBaselineSelector(candidate baseline windows) -> baseline_embed (B, C, 768)
  -> [optional] FiLM(embed_c, baseline_embed_c) per channel c
  -> flatten channels -> MLP classifier head (the only trainable parts,
     besides FiLM and the baseline selector itself when enabled).

The no-FiLM ("plain") arm doesn't just reuse a smaller classifier: comparing
a small plain probe against a bigger FiLM-augmented one would confound "does
personal-baseline conditioning help" with "does having more trainable
parameters at all help". So the plain classifier's hidden width is widened
(see `_matched_hidden_dim`) so its total trainable parameter count equals
that of a same-`hidden_dim` FiLM probe (its classifier + FiLMLayer +
LearnableBaselineSelector) -- an equal-capacity "small neural network" in
place of the plain linear probe, isolating what FiLM's conditioning itself
buys.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from baseline_selector import LearnableBaselineSelector
from film import FiLMLayer
from normwear_loader import EMBED_DIM


def _trainable_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _film_and_selector_param_count(feature_dim: int) -> int:
    """Trainable params of a FiLMLayer + LearnableBaselineSelector pair, for
    whatever `feature_dim` the probe uses. `max_baseline_windows` doesn't
    affect the selector's param count (it holds exactly one scalar
    regardless), so a throwaway value of 1 is used purely for construction."""
    film = FiLMLayer(feature_dim=feature_dim, cond_dim=feature_dim)
    selector = LearnableBaselineSelector(max_windows=1, window_seconds=1.0)
    return _trainable_param_count(film) + _trainable_param_count(selector)


def _matched_hidden_dim(nominal_hidden_dim: int, in_dim: int, num_classes: int, extra_params: int) -> int:
    """Solves for the hidden width of a single-hidden-layer
    Linear->BatchNorm->GELU->Dropout->Linear classifier (the plain probe's
    only trainable part) whose parameter count equals what a FiLM probe of
    the same `nominal_hidden_dim` would have in total (its identically-shaped
    classifier, plus FiLMLayer, plus LearnableBaselineSelector).

    That classifier's param count is affine in its hidden width h, with slope
    (in_dim + num_classes + 3) regardless of use_film -- the FiLM extras
    don't depend on h at all (FiLMLayer's own hidden width is a fixed
    constant, see film.py) -- so matching total params reduces to a constant
    additive offset on h, solved in closed form rather than searched:
        matched_h = nominal_h + extra_params / (in_dim + num_classes + 3)
    Rounding this offset to the nearest integer hidden unit leaves a residual
    of at most one unit's worth of slope (a few hundred params out of
    millions) -- negligible, and not worth a discrete search to shave further.
    """
    slope = in_dim + num_classes + 3
    return max(1, round(nominal_hidden_dim + extra_params / slope))


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
        in_dim = num_channels * EMBED_DIM
        if use_film:
            assert max_baseline_windows is not None, "use_film=True requires max_baseline_windows (the user-specified r_minutes upper bound)"
            self.baseline_selector = LearnableBaselineSelector(max_baseline_windows, window_seconds, selector_temperature)
            self.film = FiLMLayer(feature_dim=EMBED_DIM, cond_dim=EMBED_DIM)
            classifier_hidden_dim = hidden_dim
        else:
            self.baseline_selector = None
            self.film = None
            extra_params = _film_and_selector_param_count(EMBED_DIM)
            classifier_hidden_dim = _matched_hidden_dim(hidden_dim, in_dim, num_classes, extra_params)

        self.classifier_hidden_dim = classifier_hidden_dim
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, classifier_hidden_dim),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
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

    def trainable_param_count(self) -> int:
        return _trainable_param_count(self)
