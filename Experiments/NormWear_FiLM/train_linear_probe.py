"""Linear probing of frozen NormWear on WESAD, restricted to the
baseline/stress/amusement 3-way classification task, with an optional FiLM
module that conditions each window's embedding on that subject's own
resting-state ("baseline") signal before the classifier head.

Single-experiment CLI: runs exactly one (split mode x use_film) combination.
For the full ablation study (class-holdout for each choice of novel class,
and LOSO, each x plain/FiLM) with results saved to disk in one go, use
run_ablation.py instead -- it shares this file's underlying logic (see
experiment.py) but also re-uses one embedding cache across all runs.

Run from anywhere, e.g. from the repo root:
    python "Experiments/NormWear + FiLM/train_linear_probe.py" --eval_mode class_holdout --novel_class stress --use_film --r_minutes_max 5
    python "Experiments/NormWear + FiLM/train_linear_probe.py" --eval_mode loso --no-use_film

Pipeline:
  1. Load the WESAD manifests and pool every baseline/stress/amusement
     window across both (this ablation study re-splits from scratch; see
     wesad_dataset.class_holdout_split / loso_folds).
  2. Encode every pooled window once with frozen NormWear and cache the
     per-channel CLS embeddings (the backbone never needs a second forward
     pass across epochs since it's frozen).
  3. --eval_mode class_holdout: train sees 80% of the two non-novel classes;
     --novel_class is withheld from training entirely and appears only in
     test, alongside the remaining 20% of the other two classes -- a test of
     generalization to a class never seen during training.
     --eval_mode loso: leave-one-subject-out cross-validation, one fold per
     subject, metrics averaged across folds -- a test of generalization to a
     subject never seen during training.
  4. When --use_film is set, each subject's earliest windows of their own
     baseline condition (up to --r_minutes_max, an upper bound) are made
     available to a LearnableBaselineSelector, which learns -- jointly with
     the classifier head, by gradient descent on the training loss -- how
     many of them to actually average into the FiLM conditioning embedding.
     The value it converges to is reported as effective_r_minutes.
  5. --no-use_film runs don't use a plain linear probe: probe_model.py widens
     its classifier so its trainable parameter count matches a same-hidden_dim
     FiLM probe's (classifier + FiLMLayer + LearnableBaselineSelector) exactly,
     so the two arms are compared at equal capacity.
  6. Unless --no-tune, hyperparameters (hidden_dim, dropout, lr, weight_decay,
     batch_size, patience, and min_delta -- the latter two controlling early
     stopping on validation loss, see experiment.ProbeConfig) are searched by
     `experiment.hyperparameter_search` on a validation split carved out of
     the training set only (--eval_mode class_holdout: --val_frac of
     train_rows; --eval_mode loso: one reserved subject) -- the test set/fold
     is never used to pick hyperparameters. That same validation split is
     reused (regardless of --no-tune) as the final fit's early-stopping
     signal, so the final fit trains on the remaining training rows rather
     than the complete training set. selector_temperature is fixed
     (not searched); see --selector_temperature.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from experiment import ProbeConfig, build_embedding_cache, run_class_holdout_experiment, run_loso_experiment
from normwear_loader import load_normwear
from wesad_dataset import TASK_CLASSES, load_wesad_manifest, task_rows

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "data", "WESAD")


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="Directory with {train,test}_manifest.csv + signals/")
    parser.add_argument("--eval_mode", choices=["class_holdout", "loso"], default="class_holdout")
    parser.add_argument("--novel_class", choices=TASK_CLASSES, default=None, help="Required for --eval_mode class_holdout: the class withheld entirely from training")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Only used by --eval_mode class_holdout")
    parser.add_argument("--tune", action=argparse.BooleanOptionalAction, default=True, help="Search hyperparameters on a validation split carved from the training set before the final fit (never the test set/fold)")
    parser.add_argument("--val_frac", type=float, default=0.2, help="class_holdout only: fraction of train_rows carved out as the tuning validation set")
    parser.add_argument("--n_trials", type=int, default=12, help="Number of random hyperparameter-search trials")
    parser.add_argument("--search_epochs", type=int, default=None, help="Epochs used only during hyperparameter search trials (default: min(epochs, 15))")
    parser.add_argument("--use_film", action=argparse.BooleanOptionalAction, default=True, help="Condition the probe on a learned baseline-signal embedding via FiLM")
    parser.add_argument("--r_minutes_max", type=float, default=5.0, help="Upper bound (minutes) on the subject baseline signal the learnable selector may draw on")
    parser.add_argument("--selector_temperature", type=float, default=0.1, help="Softness of the learned baseline-window cutoff (in window units); smaller = sharper")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_batch_size", type=int, default=16, help="Batch size used only for the one-off NormWear encoding pass")
    parser.add_argument("--epochs", type=int, default=30, help="Upper bound on training epochs; training early-stops once validation loss stops improving (see --patience)")
    parser.add_argument("--patience", type=int, default=10, help="Epochs of no validation-loss improvement (beyond --min_delta) before early-stopping")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Smallest validation-loss decrease that counts as an improvement, for early stopping")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed_cache", default=None, help="Optional directory to cache per-window NormWear embeddings across runs, one file per sample (default: <data_root>/normwear_embed_cache)")
    parser.add_argument("--results_path", default=None, help="Optional path to write this run's metrics as JSON")
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    if args.eval_mode == "class_holdout" and args.novel_class is None:
        parser.error("--eval_mode class_holdout requires --novel_class {}".format(TASK_CLASSES))

    embed_cache_dir = args.embed_cache or os.path.join(args.data_root, "normwear_embed_cache")

    manifest = load_wesad_manifest(args.data_root)
    rows = task_rows(manifest)
    print(f"Classes: {manifest.classes} (task subset)")
    print(f"Pooled baseline/stress/amusement windows: {len(rows)} across {rows['uid'].nunique()} subjects")

    normwear = load_normwear(device=args.device)
    cache = build_embedding_cache(rows, args.data_root, normwear, args.device, args.embed_batch_size, embed_cache_dir)
    del normwear  # frozen backbone isn't needed again once every window is cached

    config = ProbeConfig(
        use_film=args.use_film,
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

    if args.eval_mode == "class_holdout":
        result = run_class_holdout_experiment(
            manifest, rows, cache, config, manifest.classes, args.novel_class, train_frac=args.train_frac, split_seed=args.seed,
            tune=args.tune, val_frac=args.val_frac, n_trials=args.n_trials, search_epochs=args.search_epochs,
        )
        print(f"\nTest metrics (novel_class={args.novel_class}):")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    else:
        result = run_loso_experiment(
            manifest, rows, cache, config, manifest.classes,
            tune=args.tune, n_trials=args.n_trials, search_epochs=args.search_epochs, tune_seed=args.seed,
        )
        print("\nLOSO mean +- std across {} subjects:".format(len(result["per_fold"])))
        for k in result["mean"]:
            print(f"  {k}: {result['mean'][k]:.4f} +- {result['std'][k]:.4f}")

    if args.results_path:
        os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
        with open(args.results_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved results to {args.results_path}")


if __name__ == "__main__":
    main()
