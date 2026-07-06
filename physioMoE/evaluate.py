"""Evaluate a trained PhysioMoE checkpoint on a held-out manifest.

Example:
    uv run physioMoE-evaluate \\
        --checkpoint checkpoints/best.pt \\
        --manifest data/synthetic/manifest.csv \\
        --signals-dir data/synthetic
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from physioMoE.config import NASA_TLX_DIMENSIONS, ModelConfig
from physioMoE.data.dataset import PhysioTLXDataset, collate_fn
from physioMoE.metrics import compute_metrics
from physioMoE.models.physio_moe import PhysioMoE


def load_model(checkpoint_path: str, device: torch.device) -> PhysioMoE:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = ModelConfig(**checkpoint["model_config"])
    model = PhysioMoE(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: PhysioMoE, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_preds, all_targets, all_gates = [], [], []
    for texts, signals, targets in loader:
        signals = signals.to(device)
        preds = model(texts, signals)
        all_preds.append(preds["output"].cpu().numpy())
        all_gates.append(preds["gate_weights"].cpu().numpy())
        all_targets.append(targets.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_gates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--signals-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-csv", default=None, help="Optional path to dump predictions.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = PhysioTLXDataset(args.manifest, args.signals_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = load_model(args.checkpoint, device)
    predictions, targets, gate_weights = run_inference(model, loader, device)

    metrics = compute_metrics(predictions, targets)
    print("NASA-TLX evaluation metrics:")
    for dim in NASA_TLX_DIMENSIONS:
        print(
            f"  {dim:16s} MAE={metrics[f'{dim}_mae']:.3f}  "
            f"RMSE={metrics[f'{dim}_rmse']:.3f}  R2={metrics[f'{dim}_r2']:.3f}"
        )
    print(f"  {'overall':16s} MAE={metrics['overall_mae']:.3f}  RMSE={metrics['overall_rmse']:.3f}")
    print(f"  mean expert gate weights: {gate_weights.mean(axis=0).round(3)}")

    if args.output_csv:
        out = pd.DataFrame(predictions, columns=[f"pred_{d}" for d in NASA_TLX_DIMENSIONS])
        for i, dim in enumerate(NASA_TLX_DIMENSIONS):
            out[f"target_{dim}"] = targets[:, i]
        out.to_csv(args.output_csv, index=False)
        print(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
