"""Full architecture wiring together the diagram:

    Task --text encode--> \\
                            Router --gate weights--> ExpertBank --> weighted sum --> Output (NASA-TLX)
    Signals --NormWear--> /                      \\--(same embedding)--> Expert inputs

Text only steers the routing decision; every expert consumes the same
pooled physiological embedding, matching a standard task-conditioned MoE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from physioMoE.config import ModelConfig
from physioMoE.models.experts import ExpertBank
from physioMoE.models.normwear_encoder import NormWearEncoder
from physioMoE.models.router import Router, load_balancing_loss
from physioMoE.models.text_encoder import TextEncoder


class PhysioMoE(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        text_encoder: nn.Module | None = None,
        normwear_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config

        self.text_encoder = text_encoder or TextEncoder(
            config.text_model_name, freeze=config.freeze_text_encoder
        )
        self.normwear = normwear_encoder or NormWearEncoder(
            config.normwear_model_name, freeze=config.freeze_normwear
        )

        h = config.hidden_dim
        self.text_proj = nn.Linear(self.text_encoder.output_dim, h)
        self.physio_proj = nn.Linear(self.normwear.output_dim, h)

        self.router = Router(
            input_dim=2 * h,
            num_experts=config.num_experts,
            hidden_dim=h,
            top_k=config.top_k,
            dropout=config.dropout,
        )
        self.experts = ExpertBank(
            input_dim=h,
            num_experts=config.num_experts,
            hidden_dim=config.expert_hidden_dim,
            output_dim=config.output_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        texts: list[str],
        signals: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text_emb = self.text_encoder(texts)
        physio_emb = self.normwear(signals, channel_mask)

        text_h = F.gelu(self.text_proj(text_emb))
        physio_h = F.gelu(self.physio_proj(physio_emb))

        fused = torch.cat([text_h, physio_h], dim=-1)
        gate_weights, router_logits = self.router(fused)

        expert_outputs = self.experts(physio_h)  # (B, num_experts, output_dim)
        combined = self.experts.combine(expert_outputs, gate_weights)  # (B, output_dim)
        output = torch.sigmoid(combined) * self.config.tlx_scale

        return {
            "output": output,
            "gate_weights": gate_weights,
            "router_logits": router_logits,
            "expert_outputs": expert_outputs,
        }

    def compute_loss(
        self, predictions: dict[str, torch.Tensor], targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        task_loss = F.mse_loss(predictions["output"], targets)
        aux_loss = load_balancing_loss(predictions["gate_weights"])
        total = task_loss + self.config.aux_loss_coef * aux_loss
        return {"loss": total, "task_loss": task_loss, "aux_loss": aux_loss}

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)
