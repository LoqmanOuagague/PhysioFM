"""Shared "train one probe on a train/test window pool, evaluate it" core.

Used both by the single-run CLI (train_linear_probe.py) and the ablation
orchestrator (run_ablation.py, which calls this once per novel class for the
class-holdout split and once per subject for leave-one-subject-out).
"""

from __future__ import annotations

import os
import random
import resource
from dataclasses import asdict, dataclass, replace

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
    carve_validation_split,
    class_holdout_split,
    load_window,
    loso_folds,
    loso_validation_split,
    max_baseline_windows,
)

# Random-search space for hyperparameter tuning (experiment.py::hyperparameter_search).
HP_SEARCH_SPACE = {
    "hidden_dim": [128, 192, 256, 384, 512],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "batch_size": [32, 64, 128],
    "min_delta": [1e-5, 1e-4, 1e-3],
}


@dataclass
class ProbeConfig:
    use_film: bool = True
    r_minutes_max: float = 5.0
    selector_temperature: float = 0.1
    hidden_dim: int = 256
    dropout: float = 0.3
    batch_size: int = 64
    epochs: int = 30  # upper bound; training early-stops once train_loss stops improving (see `patience`)
    patience: int = 10  # epochs of no train_loss improvement (beyond min_delta) before stopping
    min_delta: float = 1e-4  # smallest train_loss decrease that counts as an improvement
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

    def log_mem(stage: str) -> None:
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        print(f"[build_embedding_cache] {stage}: peak RSS so far = {rss_gb:.2f} GB", flush=True)

    log_mem("start")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for sample_id in rows["sample_id"]:
            path = sample_path(sample_id)
            if os.path.exists(path):
                cache[sample_id] = torch.load(path)
        log_mem(f"after preloading {len(cache)} cached embeddings from {cache_dir}")

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


@torch.no_grad()
def compute_loss(
    model: NormWearFiLMProbe,
    loader: DataLoader,
    baseline_seq_all: torch.Tensor | None,
    baseline_mask_all: torch.Tensor | None,
    device: str,
    criterion: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    for embed, subject_idx, labels in loader:
        embed, labels = embed.to(device), labels.to(device)
        if model.use_film:
            b_seq = baseline_seq_all[subject_idx].to(device)
            b_mask = baseline_mask_all[subject_idx].to(device)
        else:
            b_seq = b_mask = None
        logits = model(embed, b_seq, b_mask)
        total_loss += criterion(logits, labels).item() * len(labels)
    return total_loss / len(loader.dataset)


def train_probe(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    manifest: WesadManifest,
    cache: dict[str, torch.Tensor],
    config: ProbeConfig,
    classes: list[str],
    verbose: bool = True,
    track_val_loss: bool = False,
) -> tuple[NormWearFiLMProbe, dict[str, float], list[dict[str, float]] | None]:
    """Trains one probe (optionally +FiLM, with a learnable baseline-duration
    selector) on `train_rows`, evaluates on `test_rows`. Both dataframes are
    pools of `wesad_dataset.task_rows` (already filtered to the 3-class task)
    however the caller split them -- a class-holdout split or one LOSO fold.

    When `track_val_loss`, `test_rows`' loss is also computed every epoch
    (in addition to train_loss) and both are printed and returned as
    `loss_history` (a list of {epoch, train_loss, val_loss}); otherwise
    `loss_history` is None."""
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

    # drop_last: the classifier head's BatchNorm1d can't compute batch stats
    # from a single sample, which a trailing under-full batch can leave it
    # (bites some LOSO folds depending on how len(train_rows) % batch_size falls).
    train_loader = DataLoader(TensorDataset(train_embed, train_subj, train_labels), batch_size=config.batch_size, shuffle=True, drop_last=True)
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
    if verbose:
        arch_note = f"classifier_hidden_dim={model.classifier_hidden_dim} (matched to FiLM capacity)" if not config.use_film else "classifier_hidden_dim={}".format(model.classifier_hidden_dim)
        print(f"  Model: {arch_note}, trainable_params={model.trainable_param_count()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    loss_history: list[dict[str, float]] | None = [] if track_val_loss else None
    best_train_loss = float("inf")
    epochs_without_improvement = 0

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

        if track_val_loss:
            val_loss = compute_loss(model, test_loader, baseline_seq_all, baseline_mask_all, config.device, criterion)
            loss_history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            print(f"  epoch {epoch + 1}/{config.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        elif verbose and ((epoch + 1) % 5 == 0 or epoch == config.epochs - 1):
            msg = f"  epoch {epoch + 1}/{config.epochs}  train_loss={train_loss:.4f}"
            if config.use_film:
                msg += f"  effective_r_minutes={model.effective_baseline_minutes():.3f}"
            print(msg)

        if train_loss < best_train_loss - config.min_delta:
            best_train_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            if verbose or track_val_loss:
                print(f"  Early stopping at epoch {epoch + 1}/{config.epochs} "
                      f"(train_loss hasn't improved by >= {config.min_delta} for {config.patience} epochs)")
            break

    metrics = evaluate(model, test_loader, baseline_seq_all, baseline_mask_all, config.device, len(classes))
    if config.use_film:
        metrics["effective_r_minutes"] = model.effective_baseline_minutes()
    return model, metrics, loss_history


def summarize_folds(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean/std of each metric across LOSO folds (nan-safe: a fold whose test
    subject is missing a class contributes nan to roc_auc_ovo_macro, and
    should not silently zero out the average)."""
    keys = fold_metrics[0].keys()
    mean = {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in keys}
    std = {k: float(np.nanstd([m[k] for m in fold_metrics])) for k in keys}
    return {"mean": mean, "std": std}


def sample_hp_configs(base_config: ProbeConfig, use_film: bool, n_trials: int, search_epochs: int, seed: int) -> list[ProbeConfig]:
    """Draws up to `n_trials` distinct hyperparameter combinations from
    HP_SEARCH_SPACE, as variants of `base_config` with `epochs` swapped for
    the (usually smaller) `search_epochs` -- the search itself doesn't need a
    full training run to rank configurations, only the final fit does.
    selector_temperature is fixed (not searched); see ProbeConfig."""
    space = dict(HP_SEARCH_SPACE)

    rng = random.Random(seed)
    seen: set[tuple] = set()
    configs: list[ProbeConfig] = []
    attempts = 0
    while len(configs) < n_trials and attempts < n_trials * 10:
        attempts += 1
        choice = {k: rng.choice(v) for k, v in space.items()}
        key = tuple(sorted(choice.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append(replace(base_config, use_film=use_film, epochs=search_epochs, **choice))
    return configs


def hyperparameter_search(
    search_train_rows: pd.DataFrame, val_rows: pd.DataFrame, manifest: WesadManifest, cache: dict[str, torch.Tensor],
    base_config: ProbeConfig, classes: list[str], use_film: bool, n_trials: int = 12, search_epochs: int | None = None,
    seed: int | None = None,
) -> tuple[ProbeConfig, list[dict]]:
    """Random search over HP_SEARCH_SPACE, picking whichever config maximizes
    macro-F1 on `val_rows` after training on `search_train_rows` only -- never
    on the experiment's real test set/fold. Returns the winning config (with
    `epochs` restored to `base_config.epochs` for the final fit) and the full
    trial history."""
    seed = base_config.seed if seed is None else seed
    search_epochs = search_epochs or min(base_config.epochs, 15)
    candidates = sample_hp_configs(base_config, use_film, n_trials, search_epochs, seed)

    history: list[dict] = []
    best_config, best_f1 = None, -1.0
    for i, cfg in enumerate(candidates):
        _, metrics, _ = train_probe(search_train_rows, val_rows, manifest, cache, cfg, classes, verbose=False)
        f1 = metrics["f1_macro"]
        tag = ", ".join(f"{k}={getattr(cfg, k)}" for k in HP_SEARCH_SPACE if k != "hidden_dim")
        print(f"  [hp search {i + 1}/{len(candidates)}] hidden_dim={cfg.hidden_dim}, {tag} -> val_f1_macro={f1:.4f}")
        history.append({"config": asdict(cfg), "val_f1_macro": f1})
        if f1 > best_f1:
            best_f1, best_config = f1, cfg

    best_config = replace(best_config, epochs=base_config.epochs)
    print(f"  Best: val_f1_macro={best_f1:.4f}, hidden_dim={best_config.hidden_dim}, dropout={best_config.dropout}, "
          f"lr={best_config.lr}, weight_decay={best_config.weight_decay}, batch_size={best_config.batch_size}, "
          f"patience={best_config.patience}, min_delta={best_config.min_delta}")
    return best_config, history


def run_class_holdout_experiment(
    manifest: WesadManifest, rows: pd.DataFrame, cache: dict[str, torch.Tensor], config: ProbeConfig,
    classes: list[str], novel_class: str, train_frac: float = 0.8, split_seed: int = 42,
    tune: bool = True, val_frac: float = 0.2, n_trials: int = 12, search_epochs: int | None = None,
) -> dict:
    """80/20 split of the two non-novel classes (subjects freely mixed
    between train and test); `novel_class` is withheld from training
    entirely and appears only in the test set.

    When `tune`, hyperparameters are searched independently for this
    experiment: `val_frac` of `train_rows` (stratified by class) is carved
    out as a validation set (see `wesad_dataset.carve_validation_split`),
    the search trains on the remaining 1 - val_frac and picks whichever
    config scores highest validation macro-F1, and only then is the winning
    config refit on the *full* train_rows and scored on the untouched
    test_rows -- the test set never influences which hyperparameters are
    chosen."""
    train_rows, test_rows = class_holdout_split(rows, novel_class, train_frac=train_frac, seed=split_seed)

    hp_search_history = None
    if tune:
        search_train_rows, val_rows = carve_validation_split(train_rows, val_frac=val_frac, seed=split_seed)
        print(f"HP search: {len(search_train_rows)} search-train / {len(val_rows)} val windows (of {len(train_rows)} train)")
        config, hp_search_history = hyperparameter_search(
            search_train_rows, val_rows, manifest, cache, config, classes, config.use_film,
            n_trials=n_trials, search_epochs=search_epochs, seed=split_seed,
        )

    _, metrics, _ = train_probe(train_rows, test_rows, manifest, cache, config, classes)
    result = {
        "mode": "class_holdout",
        "novel_class": novel_class,
        "use_film": config.use_film,
        "train_frac": train_frac,
        "metrics": metrics,
        "config": asdict(config),
    }
    if hp_search_history is not None:
        result["hp_search"] = hp_search_history
    return result


def run_loso_experiment(
    manifest: WesadManifest, rows: pd.DataFrame, cache: dict[str, torch.Tensor], config: ProbeConfig, classes: list[str],
    tune: bool = True, n_trials: int = 12, search_epochs: int | None = None, tune_seed: int | None = None,
) -> dict:
    """Leave-one-subject-out cross-validation: one fold per subject, trained
    on every other subject's windows.

    When `tune`, hyperparameters are searched once for this experiment
    (not once per fold, which would multiply cost by the number of subjects):
    one subject (a "folder"/participant) is reserved as a validation fold
    (see `wesad_dataset.loso_validation_split`), the search trains on every
    other subject and picks whichever config scores highest macro-F1 on the
    reserved subject, and only then does the real per-subject evaluation loop
    below run with the winning config. That reserved subject still takes its
    normal turn as a test fold in the loop -- by then tuning is finished, so
    no reported test metric was used to choose hyperparameters."""
    hp_search_history = None
    if tune:
        tune_seed = config.seed if tune_seed is None else tune_seed
        val_uid, search_train_rows, val_rows = loso_validation_split(rows, seed=tune_seed)
        print(f"HP search: validation participant={val_uid} ({len(val_rows)} windows), "
              f"search-train={len(search_train_rows)} windows across {search_train_rows['uid'].nunique()} subjects")
        config, hp_search_history = hyperparameter_search(
            search_train_rows, val_rows, manifest, cache, config, classes, config.use_film,
            n_trials=n_trials, search_epochs=search_epochs, seed=tune_seed,
        )

    per_fold: dict[str, dict[str, float]] = {}
    fold_loss_history: dict[str, list[dict[str, float]]] = {}
    fold_metrics: list[dict[str, float]] = []
    for uid, train_rows, test_rows in loso_folds(rows):
        print(f"LOSO fold: held-out subject {uid} ({len(test_rows)} windows)")
        _, metrics, loss_history = train_probe(train_rows, test_rows, manifest, cache, config, classes, verbose=False, track_val_loss=True)
        per_fold[uid] = metrics
        fold_loss_history[uid] = loss_history
        fold_metrics.append(metrics)
    result = {
        "mode": "loso", "use_film": config.use_film, "config": asdict(config),
        "per_fold": per_fold, "loss_history": fold_loss_history, **summarize_folds(fold_metrics),
    }
    if hp_search_history is not None:
        result["hp_search"] = hp_search_history
    return result
