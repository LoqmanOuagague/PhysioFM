"""Dataset for (task text, multivariate physiological signal, NASA-TLX scores) triples.

Expects a manifest CSV with columns:
    sample_id, task_text, signal_path, mental_demand, physical_demand,
    temporal_demand, performance, effort, frustration

``signal_path`` is a path (relative to ``signals_dir`` if given, otherwise
absolute/relative to cwd) to a ``.npy`` file holding a float array of shape
``(num_channels, sequence_length)``. All samples must share the same number
of channels; sequence length may vary and is zero-padded within a batch.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from physioMoE.config import NASA_TLX_DIMENSIONS


class PhysioTLXDataset(Dataset):
    def __init__(self, manifest_path: str, signals_dir: str | None = None):
        self.manifest = pd.read_csv(manifest_path)
        self.signals_dir = signals_dir

        missing = [c for c in NASA_TLX_DIMENSIONS if c not in self.manifest.columns]
        if missing:
            raise ValueError(f"Manifest is missing NASA-TLX columns: {missing}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]

        signal_path = row["signal_path"]
        if self.signals_dir is not None:
            signal_path = os.path.join(self.signals_dir, signal_path)
        signal = np.load(signal_path).astype(np.float32)  # (C, T)

        target = row[NASA_TLX_DIMENSIONS].to_numpy(dtype=np.float32)

        return str(row["task_text"]), torch.from_numpy(signal), torch.from_numpy(target)


def collate_fn(
    batch: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Collate a batch of ``PhysioTLXDataset`` samples for a ``DataLoader``.

    Signals may have different sequence lengths (dim -1), so each signal is
    zero-padded on the right up to the longest one in the batch. Task texts
    and NASA-TLX targets are left as-is / stacked.

    Args:
        batch: List of ``(task_text, signal, target)`` tuples, where
            ``signal`` has shape ``(num_channels, seq_len)`` (``seq_len`` may
            vary across the batch) and ``target`` has shape
            ``(len(NASA_TLX_DIMENSIONS),)``.

    Returns:
        A tuple ``(texts, signals, targets)`` where ``texts`` is the list of
        task texts, ``signals`` is a tensor of shape
        ``(batch_size, num_channels, max_seq_len)`` zero-padded to the
        longest sequence, and ``targets`` is a tensor of shape
        ``(batch_size, len(NASA_TLX_DIMENSIONS))``.
    """
    texts, signals, targets = zip(*batch)

    max_len = max(s.shape[-1] for s in signals)
    num_channels = signals[0].shape[0]
    padded = torch.zeros(len(signals), num_channels, max_len, dtype=signals[0].dtype)
    for i, s in enumerate(signals):
        padded[i, :, : s.shape[-1]] = s

    return list(texts), padded, torch.stack(targets)
