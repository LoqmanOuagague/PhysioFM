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
README.md for why.

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
    parser.add_argument("--train_frac", type=float, default=0.8, help="Class-holdout experiment train fraction (of the two non-novel classes)")
    parser.add_argument("--r_minutes_max", type=float, default=5.0, help="Upper bound (minutes) on the subject baseline signal the learnable selector may draw on")
    parser.add_argument("--selector_temperature", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_batch_size", type=int, default=16, help="Batch size used only for the one-off NormWear encoding pass")
    parser.add_argument("--epochs", type=int, default=30)
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
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )


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

    results = {}
    n_total = 3 * 2 + 2
    step = 0

    for novel_class in TASK_CLASSES:
        for use_film in (False, True):
            step += 1
            tag = "FiLM" if use_film else "plain"
            print(f"\n[{step}/{n_total}] class-holdout (novel_class={novel_class}), {tag} linear probe")
            results[f"class_holdout_novel={novel_class}_{'film' if use_film else 'plain'}"] = run_class_holdout_experiment(
                manifest, rows, cache, make_config(args, use_film=use_film), manifest.classes, novel_class, args.train_frac, args.seed
            )

    for use_film in (False, True):
        step += 1
        tag = "FiLM" if use_film else "plain"
        print(f"\n[{step}/{n_total}] leave-one-subject-out, {tag} linear probe")
        results[f"loso_{'film' if use_film else 'plain'}"] = run_loso_experiment(
            manifest, rows, cache, make_config(args, use_film=use_film), manifest.classes
        )

    print_summary(results)

    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved full results to {args.results_path}")


if __name__ == "__main__":
    main()
