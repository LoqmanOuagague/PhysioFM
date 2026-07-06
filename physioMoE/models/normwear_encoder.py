"""Wrapper around the NormWear foundation model for multivariate physiological signals.

NormWear (https://huggingface.co/mosaic-laboratory/normwear) is a ~0.2B
parameter transformer pretrained on PPG/ECG/EEG/GSR/IMU signals. It is loaded
via ``trust_remote_code=True`` and expects input of shape
``(batch, num_channels, sequence_length)``. Calling it with
``return_enc_out=True`` returns per-channel patch embeddings of shape
``(batch, num_channels, num_patches, 768)`` whose first patch is a CLS token.

This wrapper keeps NormWear frozen by default (it is a foundation model, not
something we want to fine-tune end-to-end on a small NASA-TLX dataset) and
exposes a single pooled embedding per sample. Participants don't all have the
same set of sensors available, so the number of channels varies per sample;
batches zero-pad the channel dimension and supply a boolean ``channel_mask``.
Pooling across the (possibly padded) per-channel CLS tokens is done with a
small learned attention head rather than a plain mean, so it can weight
channels instead of treating a 2-sensor and a 5-sensor participant identically
and so it ignores padded channels entirely.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

NORMWEAR_EMBED_DIM = 768


class ChannelAttentionPool(nn.Module):
    """Learned attention pooling over an arbitrary number of
    per-channel embeddings, collapsing (B, C, D) -> (B, D)."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * dim**-0.5)
        self.scale = dim**-0.5

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, C, D) - one D-dim embedding per channel, C can vary between calls.
        # channel_mask: optional (B, C) bool tensor, True where a channel holds
        # real (non-padded) data. Defaults to all channels being valid.
        # Score each channel by how well it aligns with the shared learned query
        # vector (a dot product per channel), same idea as attention where the
        # query is fixed instead of coming from the input itself:
        #   scores[b, c] = x[b, c, :] . query   (scaled by 1/sqrt(D) for stability,
        #   same reasoning as scaled dot-product attention).
        scores = torch.einsum("bcd,d->bc", x, self.query) * self.scale  # (B, C)
        if channel_mask is not None:
            # Padded channels get -inf score so softmax assigns them zero weight.
            scores = scores.masked_fill(~channel_mask, float("-inf"))
        # Turn the C scores per sample into a probability distribution over
        # channels, so every sample's weights sum to 1 regardless of how many
        # channels C it has - this is what makes the pooling agnostic to the
        # number of input vectors.
        weights = torch.softmax(scores, dim=-1)  # (B, C)
        # Collapse the channel dimension by taking the weighted sum of the
        # per-channel embeddings, i.e. for each batch element:
        #   out[b, :] = sum_c weights[b, c] * x[b, c, :]
        # Channels the model finds more relevant contribute more to the pooled
        # embedding than a plain unweighted mean would allow.
        return torch.einsum("bc,bcd->bd", weights, x)  # (B, D)
        

class NormWearEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "mosaic-laboratory/normwear",
        freeze: bool = True,
        local_files_only: bool = False,
    ):
        """Wrapper around the NormWear foundation model for multivariate physiological signals.

        Args:
            model_name (str): HF model id to load. Defaults to "mosaic-laboratory/normwear".
            freeze (bool): Whether to freeze the NormWear model parameters. Defaults to True.
            local_files_only (bool): Whether to load the model from local files only. Defaults to False.
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=local_files_only
        )
        self.freeze = freeze
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.pool = ChannelAttentionPool(NORMWEAR_EMBED_DIM)
        self.output_dim = NORMWEAR_EMBED_DIM

    def forward(
        self, signals: torch.Tensor, channel_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        signals: (batch, num_channels, sequence_length), zero-padded along the
            channel dim for participants with fewer available sensors.
        channel_mask: optional (batch, num_channels) bool tensor, True where a
            channel holds real (non-padded) data. Defaults to all channels
            being treated as valid.
        -> (batch, 768)
        """

        def run():
            outpack = self.model(
                signals,
                return_spec=False,
                return_enc_out=True,
                return_dec_out=False,
                zero_shot_input_pack=None,
            )
            enc_out = outpack["enc_out"]  # (B, C, num_patches, 768)
            return enc_out[:, :, 0, :]  # (B, C, 768) CLS token per channel

        if self.freeze:
            with torch.no_grad():
                cls_per_channel = run()
        else:
            cls_per_channel = run()

        return self.pool(cls_per_channel, channel_mask)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self
