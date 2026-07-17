"""Loads the frozen NormWear foundation model (mosaic-laboratory/normwear).

``AutoModel.from_pretrained(..., trust_remote_code=True)`` fails on this repo's
transformers version (5.9.0): transformers now always constructs the model
under a ``torch.device("meta")`` context for fast loading, but NormWear's
``PatchEmbed_new.get_output_shape`` runs a real forward pass
(``self.proj(torch.randn(...))``) inside ``__init__`` to work out its patch
grid, which cannot execute against meta (data-less) weights and raises
``NotImplementedError: Cannot copy out of meta tensor; no data!``.

The workaround is to sidestep ``from_pretrained`` entirely: import the model
class HF already dynamically downloaded into the local hub cache, construct it
directly (a plain, non-meta initialization), and load the checkpoint weights
by hand via ``load_state_dict``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

MODEL_NAME = "mosaic-laboratory/normwear"
EMBED_DIM = 768


def _import_dynamic_module(module_name: str, snapshot_dir: str):
    hub_modules_root = str(Path.home() / ".cache" / "huggingface" / "modules")
    if hub_modules_root not in sys.path:
        sys.path.insert(0, hub_modules_root)
    return importlib.import_module(module_name)


def load_normwear(local_files_only: bool = True, device: str | torch.device = "cpu") -> nn.Module:
    """Returns the frozen, eval-mode NormWear model.

    The returned module's forward signature is the one NormWear itself
    defines: ``model(x, return_spec=False, return_enc_out=True,
    return_dec_out=False, zero_shot_input_pack=None)`` with
    ``x: (batch, num_channels, seq_len)`` raw signal (CWT spectrogram
    computation happens internally), returning a dict with
    ``enc_out: (batch, num_channels, num_patches + 1, 768)`` (index 0 along
    the patch axis is the per-channel CLS token).
    """
    snapshot_dir = snapshot_download(MODEL_NAME, local_files_only=local_files_only)

    # transformers registers the dynamically-downloaded modeling code under
    # transformers_modules.<org>.<repo>.<revision>.<file>; the revision
    # (commit hash) directory name is the snapshot dir's basename.
    revision = Path(snapshot_dir).name
    module_root = f"transformers_modules.mosaic_hyphen_laboratory.normwear.{revision}"
    modeling = _import_dynamic_module(f"{module_root}.modeling_normwear", snapshot_dir)
    configuration = _import_dynamic_module(f"{module_root}.configuration_normwear", snapshot_dir)

    config = configuration.NormWearConfig.from_pretrained(snapshot_dir)
    model = modeling.NormWearModel(config)

    weights_path = Path(snapshot_dir) / "model.safetensors"
    state_dict = load_file(str(weights_path))
    # checkpoint keys are unprefixed (the inner NormWear module's own
    # names); NormWearModel wraps it as `self.normwear`.
    state_dict = {f"normwear.{k}": v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


@torch.no_grad()
def encode_windows(model: nn.Module, signals: torch.Tensor) -> torch.Tensor:
    """signals: (batch, num_channels, seq_len) -> per-channel CLS embeddings (batch, num_channels, 768)."""
    outpack = model(signals, return_spec=False, return_enc_out=True, return_dec_out=False, zero_shot_input_pack=None)
    return outpack["enc_out"][:, :, 0, :]
