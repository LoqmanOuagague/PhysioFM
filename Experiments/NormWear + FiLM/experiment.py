"""Shared "train one probe on a train/test window pool, evaluate it" core.

Used both by the single-run CLI (train_linear_probe.py) and the ablation
orchestrator (run_ablation.py, which calls this once per novel class for the
class-holdout split and once per subject for leave-one-subject-out).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from normwear_loader import encode_windows
from probe_model import NormWearFiLMProbe
from wesad_dataset import (
    WINDOW_SECONDS,
    WesadManifest,
    build_baseline_sequences,
    class_holdout_split,
    load_window,
    loso_folds,
    max_baseline_windows,
)


@dataclass
class ProbeConfig:
    use_film: bool = True
    r_minutes_max: float = 5.0
    selector_temperature: float = 1.0
    hidden_dim: int = 256
    dropout: float = 0.3
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def build_embedding_cache(
    rows: pd.DataFrame, root: str, normwear: nn.Module, device: str, batch_size: int, cache_dir: str | None = None
) -> dict[str, torch.Tensor]:
    """Encodes every window in `rows` and caches {sample_id -> (C, 768) embedding}.

    Each embedding is written to its own file under `cache_dir` right after it's
    computed (rather than one big file at the end), so an interrupted run keeps
    whatever it already encoded. Samples already present under `cache_dir` are
    loaded from disk instead of being recomputed.
    """
    cache: dict[str, torch.Tensor] = {}

    def sample_path(sample_id: str) -> str:
        return os.path.join(cache_dir, f"{sample_id}.pt")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for sample_id in rows["sample_id"]:
            path = sample_path(sample_id)
            if os.path.exists(path):
                cache[sample_id] = torch.load(path)

    remaining = rows[~rows["sample_id"].isin(cache)]
    if remaining.empty:
        print(f"All {len(rows)} windows already cached in {cache_dir}")
        return cache

    print(f"Encoding {len(remaining)}/{len(rows)} windows with NormWear ({len(rows) - len(remaining)} already cached)...")
    for start in tqdm(range(0, len(remaining), batch_size)):
        chunk = remaining.iloc[start : start + batch_size]
        signals = np.stack([load_window(root, p) for p in chunk["signal_path"]])
        signals = torch.from_numpy(signals).to(device)
        embeds = encode_windows(normwear, signals).cpu()  # (b, C, 768)
        for sample_id, embed in zip(chunk["sample_id"], embeds):
            cache[sample_id] = embed
            if cache_dir:
                torch.save(embed, sample_path(sample_id))

    return cache


def build_split_arrays(rows: pd.DataFrame, cache: dict[str, torch.Tensor], uid_to_idx: dict[str, int], classes: list[str]):
    embeds = torch.stack([cache[sid] for sid in rows["sample_id"]])  # (N, C, 768)
    subject_idx = torch.tensor([uid_to_idx[uid] for uid in rows["uid"]], dtype=torch.long)
    labels = torch.tensor([classes.index(c) for c in rows["condition"]], dtype=torch.long)
    return embeds, subject_idx, labels


def evaluate(
    model: NormWearFiLMProbe,
    loader: DataLoader,
    baseline_seq_all: torch.Tensor | None,
    baseline_mask_all: torch.Tensor | None,
    device: str,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for embed, subject_idx, labels in loader:
            embed = embed.to(device)
            if model.use_film:
                b_seq = baseline_seq_all[subject_idx].to(device)
                b_mask = baseline_mask_all[subject_idx].to(device)
            else:
                b_seq = b_mask = None
            logits = model(embed, b_seq, b_mask)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=-1)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }
    try:
        metrics["roc_auc_ovo_macro"] = roc_auc_score(labels, probs, multi_class="ovo", average="macro", labels=list(range(num_classes)))
    except ValueError:
        metrics["roc_auc_ovo_macro"] = float("nan")  # a class is missing from this split (e.g. a small LOSO fold)
    return metrics


def train_probe(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    manifest: WesadManifest,
    cache: dict[str, torch.Tensor],
    config: ProbeConfig,
    classes: list[str],
    verbose: bool = True,
) -> tuple[NormWearFiLMProbe, dict[str, float]]:
    """Trains one probe (optionally +FiLM, with a learnable baseline-duration
    selector) on `train_rows`, evaluates on `test_rows`. Both dataframes are
    pools of `wesad_dataset.task_rows` (already filtered to the 3-class task)
    however the caller split them -- a class-holdout split or one LOSO fold."""
    set_seed(config.seed)

    subject_ids = sorted(pd.concat([train_rows["uid"], test_rows["uid"]]).unique())
    uid_to_idx = {uid: i for i, uid in enumerate(subject_ids)}
    if config.use_film:
        _, baseline_seq_all, baseline_mask_all = build_baseline_sequences(manifest, cache, config.r_minutes_max, subject_ids)
    else:
        baseline_seq_all = baseline_mask_all = None

    num_channels = next(iter(cache.values())).shape[0]
    train_embed, train_subj, train_labels = build_split_arrays(train_rows, cache, uid_to_idx, classes)
    test_embed, test_subj, test_labels = build_split_arrays(test_rows, cache, uid_to_idx, classes)

    train_loader = DataLoader(TensorDataset(train_embed, train_subj, train_labels), batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_embed, test_subj, test_labels), batch_size=config.batch_size, shuffle=False)

    model = NormWearFiLMProbe(
        num_channels=num_channels,
        num_classes=len(classes),
        use_film=config.use_film,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        max_baseline_windows=max_baseline_windows(config.r_minutes_max) if config.use_film else None,
        window_seconds=WINDOW_SECONDS,
        selector_temperature=config.selector_temperature,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for embed, subject_idx, labels in train_loader:
            embed, labels = embed.to(config.device), labels.to(config.device)
            if config.use_film:
                b_seq = baseline_seq_all[subject_idx].to(config.device)
                b_mask = baseline_mask_all[subject_idx].to(config.device)
            else:
                b_seq = b_mask = None
            optimizer.zero_grad()
            logits = model(embed, b_seq, b_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        train_loss = total_loss / len(train_loader.dataset)

        if verbose and ((epoch + 1) % 5 == 0 or epoch == config.epochs - 1):
            msg = f"  epoch {epoch + 1}/{config.epochs}  train_loss={train_loss:.4f}"
            if config.use_film:
                msg += f"  effective_r_minutes={model.effective_baseline_minutes():.3f}"
            print(msg)

    metrics = evaluate(model, test_loader, baseline_seq_all, baseline_mask_all, config.device, len(classes))
    if config.use_film:
        metrics["effective_r_minutes"] = model.effective_baseline_minutes()
    return model, metrics


def summarize_folds(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean/std of each metric across LOSO folds (nan-safe: a fold whose test
    subject is missing a class contributes nan to roc_auc_ovo_macro, and
    should not silently zero out the average)."""
    keys = fold_metrics[0].keys()
    mean = {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in keys}
    std = {k: float(np.nanstd([m[k] for m in fold_metrics])) for k in keys}
    return {"mean": mean, "std": std}


def run_class_holdout_experiment(
    manifest: WesadManifest, rows: pd.DataFrame, cache: dict[str, torch.Tensor], config: ProbeConfig,
    classes: list[str], novel_class: str, train_frac: float = 0.8, split_seed: int = 42,
) -> dict:
    """80/20 split of the two non-novel classes (subjects freely mixed
    between train and test); `novel_class` is withheld from training
    entirely and appears only in the test set."""
    train_rows, test_rows = class_holdout_split(rows, novel_class, train_frac=train_frac, seed=split_seed)
    _, metrics = train_probe(train_rows, test_rows, manifest, cache, config, classes)
    return {
        "mode": "class_holdout",
        "novel_class": novel_class,
        "use_film": config.use_film,
        "train_frac": train_frac,
        "metrics": metrics,
    }


def run_loso_experiment(
    manifest: WesadManifest, rows: pd.DataFrame, cache: dict[str, torch.Tensor], config: ProbeConfig, classes: list[str]
) -> dict:
    """Leave-one-subject-out cross-validation: one fold per subject, trained
    on every other subject's windows."""
    per_fold: dict[str, dict[str, float]] = {}
    fold_metrics: list[dict[str, float]] = []
    for uid, train_rows, test_rows in loso_folds(rows):
        print(f"LOSO fold: held-out subject {uid} ({len(test_rows)} windows)")
        _, metrics = train_probe(train_rows, test_rows, manifest, cache, config, classes, verbose=False)
        per_fold[uid] = metrics
        fold_metrics.append(metrics)
    return {"mode": "loso", "use_film": config.use_film, "per_fold": per_fold, **summarize_folds(fold_metrics)}
