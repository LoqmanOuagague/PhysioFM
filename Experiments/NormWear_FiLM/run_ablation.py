"""Runs the full FiLM ablation study on WESAD and saves metrics for each
experiment to disk.

Every experiment only classifies baseline/stress/amusement (see
wesad_dataset.TASK_CLASSES), and varies along two independent axes:

  - split: class-holdout (train sees 80% of two classes; the third,
    "novel" class is withheld from training entirely and appears only in
    test, alongside the remaining 20% of the other two -- run once per
    choice of novel class, so 3 variants) vs. leave-one-subject-out (LOSO:
    one fold per subject, trained on every other subject, metrics averaged
    across folds).
  - use_film: plain linear probe vs. linear probe + FiLM (with a learnable
    baseline-duration selector, see baseline_selector.py).

That's 3 x 2 (class-holdout) + 1 x 2 (LOSO) = 8 runs total. Both splits are
pooled across WESAD's own train/test manifests and re-split from scratch for
this study (see wesad_dataset.class_holdout_split / loso_folds) -- see
README.md for why. Use --loso_only to skip the class-holdout splits and only
run LOSO, and/or --film_only to skip the plain (non-FiLM) arms and only run
the FiLM-conditioned ones.

The --no-use_film arm isn't a plain linear probe: probe_model.py widens its
classifier so its trainable parameter count matches a same-hidden_dim FiLM
probe's (classifier + FiLMLayer + LearnableBaselineSelector) exactly, so the
two arms are compared at equal capacity rather than confounding "does FiLM
conditioning help" with "does having more parameters help".

Unless --no-tune, hyperparameters (hidden_dim, dropout, lr, weight_decay,
batch_size, patience, and min_delta -- the latter two controlling early
stopping on validation loss, see experiment.ProbeConfig) are searched
independently for each of the 8 experiments above, by
experiment.hyperparameter_search, on a validation split carved out of that
experiment's *training* set only -- class-holdout: --val_frac of train_rows;
LOSO: one reserved subject (see wesad_dataset.carve_validation_split /
loso_validation_split) -- never the test set/fold. That same validation
split is reused (regardless of --no-tune) as the final fit's early-stopping
signal, so the final fit trains on the remaining training rows rather than
the complete training set -- see experiment.train_probe's val_rows.
selector_temperature is fixed (not searched); see --selector_temperature.

Every experiment reports Accuracy, macro-Precision, macro-Recall, macro-F1
and macro-averaged (one-vs-one) ROC AUC on its held-out data; the FiLM
experiments additionally report the learned effective_r_minutes (whole-fold
mean +- std for LOSO). Results are saved to
<this dir>/results/ablation_results.json.

Run from anywhere, e.g. from the repo root:
    python "Experiments/NormWear + FiLM/run_ablation.py" --r_minutes_max 5
"""

from __future__ import annotations

import argparse
import json
import os
import resource

import torch

from experiment import ProbeConfig, build_embedding_cache, run_class_holdout_experiment, run_loso_experiment
from normwear_loader import load_normwear
from wesad_dataset import TASK_CLASSES, load_wesad_manifest, task_rows

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "data", "WESAD")
DEFAULT_RESULTS_PATH = os.path.join(THIS_DIR, "results", "ablation_results.json")


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--loso_only", action="store_true", help="Skip the class-holdout experiments and only run the 2 LOSO runs (plain + FiLM)")
    parser.add_argument("--film_only", action="store_true", help="Skip the plain (non-FiLM) arms and only run the FiLM-conditioned experiments")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Class-holdout experiment train fraction (of the two non-novel classes)")
    parser.add_argument("--tune", action=argparse.BooleanOptionalAction, default=True, help="Search hyperparameters independently per experiment on a validation split carved from its training set (never the test set/fold)")
    parser.add_argument("--val_frac", type=float, default=0.2, help="class_holdout only: fraction of train_rows carved out as the tuning validation set")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of random hyperparameter-search trials, per experiment")
    parser.add_argument("--search_epochs", type=int, default=300, help="Epochs used only during hyperparameter search trials (default: min(epochs, 15))")
    parser.add_argument("--r_minutes_max", type=float, default=7.0, help="Upper bound (minutes) on the subject baseline signal the learnable selector may draw on")
    parser.add_argument("--selector_temperature", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_batch_size", type=int, default=16, help="Batch size used only for the one-off NormWear encoding pass")
    parser.add_argument("--epochs", type=int, default=1000, help="Upper bound on training epochs; training early-stops once validation loss stops improving (see --patience)")
    parser.add_argument("--patience", type=int, default=50, help="Epochs of no validation-loss improvement (beyond --min_delta) before early-stopping")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Smallest validation-loss decrease that counts as an improvement, for early stopping")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed_cache", default=None, help="Default: <data_root>/normwear_embed_cache")
    parser.add_argument("--results_path", default=DEFAULT_RESULTS_PATH)
    return parser


def make_config(args, use_film: bool) -> ProbeConfig:
    return ProbeConfig(
        use_film=use_film,
        r_minutes_max=args.r_minutes_max,
        selector_temperature=args.selector_temperature,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )


def get_resource_usage(device: str) -> dict:
    """Snapshot of hardware/resource usage for this run, saved alongside results."""
    cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # ru_maxrss is in KB on Linux

    gpu_name = None
    gpu_vram_gb = None
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_name = props.name
        gpu_vram_gb = round(props.total_memory / (1024 ** 3), 2)

    return {
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "cpu_count": cpu_count,
        "max_rss_mb": round(max_rss_mb, 1),
    }


def metrics_of(result: dict) -> dict:
    return result["metrics"] if result["mode"] == "class_holdout" else result["mean"]


def print_summary(results: dict):
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovo_macro"]
    print("\n" + "=" * 100)
    print(f"{'experiment':<30}" + "".join(f"{k:>14}" for k in metric_keys))
    for name, result in results.items():
        values = metrics_of(result)
        print(f"{name:<30}" + "".join(f"{values[k]:>14.4f}" for k in metric_keys))
    print("=" * 100)
    for name, result in results.items():
        if not result["use_film"]:
            continue
        print(f"{name}: learned effective_r_minutes = {metrics_of(result)['effective_r_minutes']:.3f}")
    print("=" * 100)
    for name, result in results.items():
        cfg = result["config"]
        print(f"{name}: hidden_dim={cfg['hidden_dim']}, dropout={cfg['dropout']}, lr={cfg['lr']}, "
              f"weight_decay={cfg['weight_decay']}, batch_size={cfg['batch_size']}, "
              f"patience={cfg['patience']}, min_delta={cfg['min_delta']}" +
              (f", selector_temperature={cfg['selector_temperature']}" if result["use_film"] else ""))


def main():
    args = get_args_parser().parse_args()
    embed_cache_dir = args.embed_cache or os.path.join(args.data_root, "normwear_embed_cache")

    manifest = load_wesad_manifest(args.data_root)
    rows = task_rows(manifest)
    print(f"Classes: {manifest.classes} (task subset)")
    print(f"Pooled baseline/stress/amusement windows: {len(rows)} across {rows['uid'].nunique()} subjects")

    normwear = load_normwear(device=args.device)
    cache = build_embedding_cache(rows, args.data_root, normwear, args.device, args.embed_batch_size, embed_cache_dir)
    del normwear  # shared across every experiment below; the frozen backbone isn't needed again

    film_variants = (True,) if args.film_only else (False, True)

    results = {}
    n_total = (1 if args.loso_only else 4) * len(film_variants)
    step = 0

    if not args.loso_only:
        for novel_class in TASK_CLASSES:
            for use_film in film_variants:
                step += 1
                tag = "FiLM" if use_film else "plain"
                print(f"\n[{step}/{n_total}] class-holdout (novel_class={novel_class}), {tag} linear probe")
                results[f"class_holdout_novel={novel_class}_{'film' if use_film else 'plain'}"] = run_class_holdout_experiment(
                    manifest, rows, cache, make_config(args, use_film=use_film), manifest.classes, novel_class, args.train_frac, args.seed,
                    tune=args.tune, val_frac=args.val_frac, n_trials=args.n_trials, search_epochs=args.search_epochs,
                )

    for use_film in film_variants:
        step += 1
        tag = "FiLM" if use_film else "plain"
        print(f"\n[{step}/{n_total}] leave-one-subject-out, {tag} linear probe")
        results[f"loso_{'film' if use_film else 'plain'}"] = run_loso_experiment(
            manifest, rows, cache, make_config(args, use_film=use_film), manifest.classes,
            tune=args.tune, n_trials=args.n_trials, search_epochs=args.search_epochs, tune_seed=args.seed,
        )

    print_summary(results)

    resource_usage = get_resource_usage(args.device)
    print(f"\nGPU: {resource_usage['gpu_name']} ({resource_usage['gpu_vram_gb']} GB VRAM)" if resource_usage["gpu_name"] else "\nGPU: none (ran on CPU)")
    print(f"CPUs used: {resource_usage['cpu_count']}, max RSS: {resource_usage['max_rss_mb']} MB")

    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump({"config": vars(args), "resource_usage": resource_usage, "results": results}, f, indent=2)
    print(f"\nSaved full results to {args.results_path}")


if __name__ == "__main__":
    main()
