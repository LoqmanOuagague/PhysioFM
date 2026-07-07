"""Evaluate a CogLoad1 LoRA-finetuned NormWear + TLX regressor (trained via
train.py) on the held-out test split, reporting per-dimension and overall
MAE/RMSE/R2.

Also owns the CWT spectrogram dataset/cache (SpecDataset): CWT only depends
on the raw signal, not on the model being trained, so it belongs on the
"data prep" side of the pipeline -- train.py imports it from here rather
than computing it itself.

Usage:
    python prepare.py
    python prepare.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from physioMoE.data.dataset import PhysioTLXDataset
from physioMoE.metrics import compute_metrics
from NormWear.main_model import spec_cwt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COGLOAD_DIR = os.path.join(REPO_ROOT, "NormWear", "data", "Cogload")
DEFAULT_TEST_MANIFEST = os.path.join(COGLOAD_DIR, "test_manifest.csv")
DEFAULT_CHECKPOINT = os.path.join(THIS_DIR, "checkpoints", "best.pt")
SPEC_CACHE_DIR = os.path.join(THIS_DIR, "spec_cache")


def _manifest_fingerprint(manifest_path: str) -> str:
    """Short hash of the manifest's contents, so caches keyed on it can never be
    silently reused across a different (e.g. truncated, regenerated) manifest."""
    with open(manifest_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:10]


class SpecDataset(Dataset):
    """Precomputed CWT spectrograms + NASA-TLX targets for one manifest split.

    CWT (via pywt, CPU-only and non-differentiable) only depends on the raw
    signal, so it's computed once here and cached to disk -- unlike the
    transformer forward pass, which must be re-run every epoch since the
    backbone is being fine-tuned.
    """

    def __init__(self, manifest_path: str, signals_dir: str, cache_name: str, rebuild_cache: bool = False):
        cache_path = os.path.join(SPEC_CACHE_DIR, f"{cache_name}_{_manifest_fingerprint(manifest_path)}.pt")
        if not rebuild_cache and os.path.isfile(cache_path):
            cached = torch.load(cache_path, weights_only=False)
            self.specs, self.targets = cached["specs"], cached["targets"]
            return

        base = PhysioTLXDataset(manifest_path, signals_dir=signals_dir)
        specs, targets = [], []
        for i in tqdm(range(len(base)), desc=f"Computing CWT spectrograms ({cache_name})"):
            _, signal, target = base[i]
            specs.append(spec_cwt(signal.numpy()).half())  # (nvar, 3, L, F); fp16 keeps the cache small
            targets.append(target)

        self.specs = torch.stack(specs)
        self.targets = torch.stack(targets)

        os.makedirs(SPEC_CACHE_DIR, exist_ok=True)
        torch.save({"specs": self.specs, "targets": self.targets}, cache_path)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.specs[idx], self.targets[idx]


@torch.no_grad()
def evaluate(
    manifest: str = DEFAULT_TEST_MANIFEST,
    checkpoint: str = DEFAULT_CHECKPOINT,
    rebuild_cache: bool = False,
    device: str = "cuda",
) -> dict[str, float]:
    """Ground-truth metric for the AutoResearch loop. Plain kwargs (not an argparse.Namespace)
    so train.py can call this directly at the end of a run without faking a Namespace."""
    # Deferred import: train.py imports SpecDataset from this module, so importing
    # train.py's model class at module level here would create a circular import.
    from train import LoRANormWearTLX

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    test_set = SpecDataset(manifest, COGLOAD_DIR, cache_name="test", rebuild_cache=rebuild_cache)

    model = LoRANormWearTLX(
        nvar=ckpt["nvar"],
        backbone_checkpoint=ckpt["backbone_checkpoint"],
        lora_r=ckpt["lora_r"],
        lora_alpha=ckpt["lora_alpha"],
        lora_dropout=ckpt["lora_dropout"],
        lora_targets=tuple(ckpt["lora_targets"]),
        hidden_dims=ckpt["hidden_dims"],
        head_dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    model.eval()

    specs = test_set.specs.float().to(device)
    targets = test_set.targets.numpy()

    preds = model(specs).cpu().numpy()

    metrics = compute_metrics(preds, targets)
    metrics["test_size"] = len(test_set)
    metrics["best_epoch"] = ckpt["epoch"]
    metrics["best_val_loss"] = ckpt["val_loss"]
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=DEFAULT_TEST_MANIFEST, help="Manifest CSV of the held-out split.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint written by train.py.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Recompute CWT spectrograms instead of using the cache.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(manifest=args.manifest, checkpoint=args.checkpoint, rebuild_cache=args.rebuild_cache, device=args.device)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
