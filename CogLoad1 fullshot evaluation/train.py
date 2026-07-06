"""Fine-tune the NormWear backbone (via LoRA adapters) together with a NASA-TLX
regression head on CogLoad1, loading raw signals from train_manifest.csv +
signals/ instead of NormWear's precomputed embedding pickles.

Only LoRA adapters + a small head are trained; the ~128M pretrained NormWear
weights stay frozen throughout. CWT spectrograms (the model's raw input) are
precomputed once per split and cached to disk, since they only depend on the
raw signal and don't change across epochs -- only the transformer forward
pass (backbone + LoRA + head) is repeated every epoch.

Usage:
    python train.py
    python train.py --epochs 30 --lora-r 8 --lora-targets qkv proj
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from physioMoE.config import NASA_TLX_DIMENSIONS
from NormWear.modules.normwear import NormWear

# prepare.py owns the CWT spectrogram dataset/cache (it's the "data prep" side of the
# pipeline, independent of the model). Imported here, not the other way around, so
# prepare.py's own evaluate() can import this module's model class without a cycle.
from prepare import SpecDataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COGLOAD_DIR = os.path.join(REPO_ROOT, "NormWear", "data", "Cogload")
DEFAULT_TRAIN_MANIFEST = os.path.join(COGLOAD_DIR, "train_manifest.csv")
DEFAULT_BACKBONE_CHECKPOINT = os.path.join(REPO_ROOT, "NormWear", "weights", "normwear_pretrain_ckpt.pth")
DEFAULT_CHECKPOINT = os.path.join(THIS_DIR, "checkpoints", "best.pt")

# NormWear was pretrained on (387, 65) CWT spectrograms with (9, 5) non-overlap
# patches (43 x 13 = 559 patches). Cogload's 6s @ 65Hz windows produce spectrograms
# of the same patch grid after Conv2d's floor division, so the pretrained
# pos_embed transfers with no interpolation needed.
BACKBONE_IMG_SIZE = (387, 65)
BACKBONE_PATCH_SIZE = (9, 5)

LORA_TARGET_NAMES = ("qkv", "proj", "fc1", "fc2")  # Attention.qkv/proj, Mlp.fc1/fc2


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update:
    y = W x + (alpha/r) * B(A(x)), with B zero-initialized so LoRA starts as a no-op.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return self.base(x) + self.scaling * update


def inject_lora(module: nn.Module, r: int, alpha: float, dropout: float, targets: tuple[str, ...]) -> int:
    """Recursively replace target nn.Linear children with LoRALinear. Returns the count replaced."""
    n_replaced = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            n_replaced += 1
        else:
            n_replaced += inject_lora(child, r, alpha, dropout, targets)
    return n_replaced


class TLXRegressor(nn.Module):
    """Feed-forward network predicting the 6 NASA-TLX dimensions from a pooled NormWear embedding."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LoRANormWearTLX(nn.Module):
    """NormWear backbone (frozen + LoRA adapters) -> mean-pooled per-channel embedding -> TLX head."""

    def __init__(
        self,
        nvar: int,
        backbone_checkpoint: str,
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_targets: tuple[str, ...],
        hidden_dims: list[int],
        head_dropout: float,
    ):
        super().__init__()
        self.backbone = NormWear(
            img_size=BACKBONE_IMG_SIZE,
            patch_size=BACKBONE_PATCH_SIZE,
            nvar=nvar,
            is_pretrain=False,
            use_cwt=True,
            comb_freq=False,
            mask_prob=0,
        )
        state_dict = torch.load(backbone_checkpoint, map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=False)

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        n_lora = inject_lora(self.backbone, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, targets=lora_targets)
        print(f"Injected LoRA (r={lora_r}, alpha={lora_alpha}) into {n_lora} Linear layers ({', '.join(lora_targets)}).")

        embed_dim = self.backbone.norm.normalized_shape[0]
        # LayerNorm (not BatchNorm) so normalization doesn't depend on batch statistics
        # and there's no running-stats buffer to persist alongside the LoRA checkpoint.
        self.embed_norm = nn.LayerNorm(nvar * embed_dim)
        self.head = TLXRegressor(nvar * embed_dim, hidden_dims, len(NASA_TLX_DIMENSIONS), head_dropout)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (bs, nvar, 3, L, F)
        z = self.backbone.feature_extractor(spec)  # (bs, nvar, num_patches+1, E)
        pooled = z.mean(dim=2).flatten(1)  # (bs, nvar*E): mean over CLS+patch tokens per channel
        pooled = self.embed_norm(pooled)
        return self.head(pooled)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


def train(args: argparse.Namespace) -> str:
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    full_train_set = SpecDataset(args.manifest, COGLOAD_DIR, cache_name="train", rebuild_cache=args.rebuild_cache)
    nvar = full_train_set.specs.shape[1]

    val_size = max(1, int(len(full_train_set) * args.val_split))
    train_size = len(full_train_set) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(full_train_set, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = LoRANormWearTLX(
        nvar=nvar,
        backbone_checkpoint=args.backbone_checkpoint,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=tuple(args.lora_targets),
        hidden_dims=args.hidden_dims,
        head_dropout=args.dropout,
    ).to(device)

    trainable = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.1f}M total ({100 * n_trainable / n_total:.2f}%)")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_decay_factor, patience=args.lr_patience
    )
    loss_fn = nn.MSELoss()

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for specs, targets in train_loader:
            specs, targets = specs.to(device).float(), targets.to(device)
            preds = model(specs)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            running_loss += loss.item() * len(specs)
        train_loss = running_loss / len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for specs, targets in val_loader:
                specs, targets = specs.to(device).float(), targets.to(device)
                preds = model(specs)
                val_loss += loss_fn(preds, targets).item() * len(specs)
        val_loss /= len(val_set)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr:.2e} ({epoch_time:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "trainable_state_dict": trainable_state_dict(model),
                    "nvar": nvar,
                    "backbone_checkpoint": args.backbone_checkpoint,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "lora_targets": list(args.lora_targets),
                    "hidden_dims": args.hidden_dims,
                    "dropout": args.dropout,
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                args.checkpoint,
            )

    print(f"Best val loss: {best_val_loss:.4f} -> {args.checkpoint}")
    return args.checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=DEFAULT_TRAIN_MANIFEST, help="train_manifest.csv to load raw signals from.")
    parser.add_argument("--backbone-checkpoint", default=DEFAULT_BACKBONE_CHECKPOINT, help="Pretrained NormWear backbone weights.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Where to save the best checkpoint.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Recompute CWT spectrograms instead of using the cache.")

    parser.add_argument("--lora-r", type=int, default=4, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=float, default=8.0, help="LoRA scaling numerator (scaling = alpha / r).")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="Dropout applied before the LoRA update.")
    parser.add_argument(
        "--lora-targets", nargs="+", default=list(LORA_TARGET_NAMES), choices=list(LORA_TARGET_NAMES),
        help="Which Linear layers (by attribute name) to attach LoRA adapters to.",
    )

    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 64], help="TLX head hidden layer sizes.")
    parser.add_argument("--dropout", type=float, default=0.2, help="TLX head dropout.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization strength (AdamW).")
    parser.add_argument("--lr-decay-factor", type=float, default=0.5, help="Factor ReduceLROnPlateau multiplies lr by on plateau.")
    parser.add_argument("--lr-patience", type=int, default=5, help="Epochs of no val-loss improvement before the scheduler decays lr.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm (gradient clipping).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of the train split held out for validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
