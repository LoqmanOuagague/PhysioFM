"""Text encoder for the task/instruction string shown in the architecture diagram.

Uses a small pretrained sentence-embedding transformer (default:
``sentence-transformers/all-MiniLM-L6-v2``) so the whole pipeline stays
lightweight next to the NormWear backbone. Any encoder-only
model on the Hugging Face Hub that exposes last_hidden_state works as a
drop-in replacement via ``model_name``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, freeze: bool = True, max_length: int = 64):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.output_dim: int = self.model.config.hidden_size
        self.freeze = freeze
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        if self.freeze:
            with torch.no_grad():
                out = self.model(**encoded)
        else:
            out = self.model(**encoded)

        return self._mean_pool(out.last_hidden_state, encoded["attention_mask"])

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self
