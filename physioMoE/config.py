"""Configuration dataclasses shared by training, evaluation, and model construction."""

from __future__ import annotations

from dataclasses import dataclass, field

NASA_TLX_DIMENSIONS: list[str] = [
    "mental_demand",
    "physical_demand",
    "temporal_demand",
    "performance",
    "effort",
    "frustration",
]


@dataclass
class ModelConfig:
    """Architecture hyperparameters for PhysioMoE."""

    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normwear_model_name: str = "mosaic-laboratory/normwear"

    freeze_text_encoder: bool = True
    freeze_normwear: bool = True

    hidden_dim: int = 256
    num_experts: int = 4
    top_k: int | None = None  # None -> dense gating over all experts
    expert_hidden_dim: int = 128
    dropout: float = 0.1

    output_dim: int = len(NASA_TLX_DIMENSIONS)
    tlx_scale: float = 100.0  # NASA-TLX items are conventionally scored 0-100

    aux_loss_coef: float = 0.01  # load-balancing auxiliary loss weight


@dataclass
class TrainConfig:
    manifest: str = "data/synthetic/manifest.csv"
    signals_dir: str = "data/synthetic/signals"
    output_dir: str = "checkpoints"
    val_split: float = 0.2
    seed: int = 42
    epochs: int = 20
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-2
    num_workers: int = 0
    device: str = "cuda"
    log_every: int = 10
    model: ModelConfig = field(default_factory=ModelConfig)
