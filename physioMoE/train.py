"""Train PhysioMoE: text-conditioned mixture-of-experts predicting NASA-TLX scores
from multichannel physiological signals encoded by NormWear.

Example:
    uv run physioMoE-train \\
        --manifest data/synthetic/manifest.csv \\
        --signals-dir data/synthetic \\
        --output-dir checkpoints --epochs 20
"""

from __future__ import annotations

import argparse
import dataclasses
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from physioMoE.config import ModelConfig, TrainConfig
from physioMoE.data.dataset import PhysioTLXDataset, collate_fn
from physioMoE.metrics import compute_metrics
from physioMoE.models.physio_moe import PhysioMoE


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    dataset = PhysioTLXDataset(cfg.manifest, cfg.signals_dir)
    val_size = max(1, int(len(dataset) * cfg.val_split))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate_loader(model: PhysioMoE, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0

    for texts, signals, targets in loader:
        signals, targets = signals.to(device), targets.to(device)
        preds = model(texts, signals)
        losses = model.compute_loss(preds, targets)
        total_loss += losses["loss"].item() * len(texts)

        all_preds.append(preds["output"].cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets))
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train(cfg: TrainConfig) -> str:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(cfg)

    model = PhysioMoE(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(cfg.output_dir, "best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}")

        for step, (texts, signals, targets) in enumerate(progress, start=1):
            signals, targets = signals.to(device), targets.to(device)

            preds = model(texts, signals)
            losses = model.compute_loss(preds, targets)

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            running_loss += losses["loss"].item()
            if step % cfg.log_every == 0:
                progress.set_postfix(loss=running_loss / step)

        val_metrics = evaluate_loader(model, val_loader, device)
        print(
            f"[epoch {epoch}] train_loss={running_loss / len(train_loader):.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_overall_mae={val_metrics['overall_mae']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": dataclasses.asdict(cfg.model),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_ckpt_path,
            )

    print(f"Best val loss: {best_val_loss:.4f} -> {best_ckpt_path}")
    return best_ckpt_path


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=TrainConfig.manifest)
    parser.add_argument("--signals-dir", default=TrainConfig.signals_dir)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--val-split", type=float, default=TrainConfig.val_split)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)

    parser.add_argument("--text-model-name", default=ModelConfig.text_model_name)
    parser.add_argument("--normwear-model-name", default=ModelConfig.normwear_model_name)
    parser.add_argument("--finetune-text-encoder", action="store_true")
    parser.add_argument("--finetune-normwear", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=ModelConfig.hidden_dim)
    parser.add_argument("--num-experts", type=int, default=ModelConfig.num_experts)
    parser.add_argument("--top-k", type=int, default=ModelConfig.top_k)
    parser.add_argument("--expert-hidden-dim", type=int, default=ModelConfig.expert_hidden_dim)
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    parser.add_argument("--aux-loss-coef", type=float, default=ModelConfig.aux_loss_coef)

    args = parser.parse_args()

    model_cfg = ModelConfig(
        text_model_name=args.text_model_name,
        normwear_model_name=args.normwear_model_name,
        freeze_text_encoder=not args.finetune_text_encoder,
        freeze_normwear=not args.finetune_normwear,
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        expert_hidden_dim=args.expert_hidden_dim,
        dropout=args.dropout,
        aux_loss_coef=args.aux_loss_coef,
    )

    return TrainConfig(
        manifest=args.manifest,
        signals_dir=args.signals_dir,
        output_dir=args.output_dir,
        val_split=args.val_split,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        log_every=args.log_every,
        model=model_cfg,
    )


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
