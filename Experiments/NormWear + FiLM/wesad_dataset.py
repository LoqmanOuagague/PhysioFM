"""Loads the pre-processed WESAD dataset (data/WESAD/{train,test}_manifest.csv
+ data/WESAD/signals/*.npy, produced by utils/process_wesad.py) for the
FiLM-conditioning ablation study.

Each window is a (10, 390) float64 array: 65 Hz, 6-second windows of chest
ACC x/y/z, ECG, EMG, EDA, Temp, Resp + wrist EDA, Temp. The manifest's
``condition`` column (baseline/stress/amusement/meditation) is the
classification target; ``uid`` identifies the subject. This ablation study
only classifies baseline/stress/amusement (see `TASK_CLASSES`) -- meditation
windows are dropped everywhere except as raw material for nothing at all,
since WESAD's meditation blocks aren't one of the three conditions compared
here.

The dataset ships with its own fixed train/test manifest split (one subject
held out for test, see utils/process_wesad.py), but this ablation study
needs two different splitting strategies of its own, applied by pooling
every window across both manifests and re-splitting:
  - `class_holdout_split`: an 80/20 split of two of the three classes
    (subjects freely mixed between train and test), with the third
    ("novel") class withheld from training entirely and placed only in the
    test set -- a test of whether the model generalizes to a condition it
    has never seen a training example of.
  - `loso_folds`: leave-one-subject-out cross-validation, one fold per
    subject.

WESAD's protocol records an explicit resting "baseline" condition per
subject before the stressor tasks. `select_baseline_windows` selects that
subject's earliest baseline windows (up to a caller-supplied cap) so they can
be encoded with NormWear and used, via `build_baseline_sequences`, as
candidate input to `baseline_selector.LearnableBaselineSelector` -- which
learns how many of them to actually use for FiLM conditioning.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

WINDOW_SECONDS = 6  # must match utils/process_wesad.py
BASELINE_CONDITION = "baseline"
TASK_CLASSES = ["amusement", "baseline", "stress"]  # sorted; fixed label index for this ablation study


@dataclass
class WesadManifest:
    root: str
    train_rows: pd.DataFrame
    test_rows: pd.DataFrame
    classes: list[str]  # sorted condition names -> stable label index

    @property
    def all_rows(self) -> pd.DataFrame:
        return pd.concat([self.train_rows, self.test_rows], ignore_index=True)

    def label_index(self, condition: str) -> int:
        return self.classes.index(condition)


def load_wesad_manifest(root: str) -> WesadManifest:
    train_rows = pd.read_csv(os.path.join(root, "train_manifest.csv"))
    test_rows = pd.read_csv(os.path.join(root, "test_manifest.csv"))
    classes = sorted(set(train_rows["condition"]) | set(test_rows["condition"]))
    return WesadManifest(root=root, train_rows=train_rows, test_rows=test_rows, classes=classes)


def task_rows(manifest: WesadManifest, classes: list[str] = TASK_CLASSES) -> pd.DataFrame:
    """All windows (pooled across both manifests) whose condition is one of
    `classes`, e.g. the {baseline, stress, amusement} pool this ablation
    study trains/evaluates on. Includes the subject's baseline windows
    themselves -- "baseline" is one of the classes being classified, in
    addition to being the source of the FiLM reference signal."""
    rows = manifest.all_rows
    return rows[rows["condition"].isin(classes)].reset_index(drop=True)


def load_window(root: str, signal_path: str) -> np.ndarray:
    return np.load(os.path.join(root, signal_path)).astype(np.float32)


def select_baseline_windows(manifest: WesadManifest, uid: str, r_minutes: float) -> pd.DataFrame:
    """Returns up to `r_minutes` worth of a subject's earliest baseline
    windows (fewer if the subject's baseline segment is shorter), ordered by
    (segment, window). Draws from both manifests: the baseline reference is
    a property of the subject's recording session, not of whichever split a
    given task window happens to land in."""
    n_needed = math.ceil(r_minutes * 60 / WINDOW_SECONDS)
    rows = manifest.all_rows
    subj_baseline = rows[(rows["uid"] == uid) & (rows["condition"] == BASELINE_CONDITION)]
    subj_baseline = subj_baseline.sort_values(["segment", "window"])
    return subj_baseline.head(n_needed)


def max_baseline_windows(r_minutes_max: float) -> int:
    return math.ceil(r_minutes_max * 60 / WINDOW_SECONDS)


def class_holdout_split(
    rows: pd.DataFrame, novel_class: str, train_frac: float = 0.8, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`train_frac` of every class except `novel_class` goes to train (the
    rest of those two classes goes to test); `novel_class` is withheld from
    training entirely and placed only in the test set, alongside the held-out
    fraction of the other two. Subjects are freely mixed between train and
    test (unlike `loso_folds`) -- the axis being tested here is generalization
    to an unseen *class*, not an unseen *subject*."""
    assert novel_class in set(rows["condition"]), f"{novel_class!r} not found in rows"
    novel_rows = rows[rows["condition"] == novel_class]
    other_rows = rows[rows["condition"] != novel_class]
    train_rows, held_out_rows = train_test_split(
        other_rows, train_size=train_frac, stratify=other_rows["condition"], random_state=seed, shuffle=True
    )
    test_rows = pd.concat([held_out_rows, novel_rows], ignore_index=True)
    return train_rows.reset_index(drop=True), test_rows.reset_index(drop=True)


def loso_folds(rows: pd.DataFrame):
    """Yields (held_out_uid, train_rows, test_rows) for leave-one-subject-out
    cross-validation, one fold per unique subject in `rows`."""
    for uid in sorted(rows["uid"].unique()):
        test_rows = rows[rows["uid"] == uid].reset_index(drop=True)
        train_rows = rows[rows["uid"] != uid].reset_index(drop=True)
        yield uid, train_rows, test_rows


def build_baseline_sequences(
    manifest: WesadManifest, cache: dict[str, torch.Tensor], r_minutes_max: float, subject_ids: list[str]
) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
    """Builds the padded, per-subject candidate baseline window sequences
    consumed by `baseline_selector.LearnableBaselineSelector`.

    Returns:
      uid_to_idx: uid -> row index into the two tensors below.
      seq: (num_subjects, max_windows, C, E) float, zero-padded.
      mask: (num_subjects, max_windows) bool, True where that slot holds a
        real (non-padded) window for that subject.
    """
    max_windows = max_baseline_windows(r_minutes_max)
    uid_to_idx = {uid: i for i, uid in enumerate(subject_ids)}

    sample_embed = next(iter(cache.values()))
    num_channels, embed_dim = sample_embed.shape
    seq = torch.zeros(len(subject_ids), max_windows, num_channels, embed_dim)
    mask = torch.zeros(len(subject_ids), max_windows, dtype=torch.bool)

    for uid in subject_ids:
        rows = select_baseline_windows(manifest, uid, r_minutes_max)
        assert len(rows) > 0, f"subject {uid} has no baseline-condition windows to build a FiLM reference from"
        idx = uid_to_idx[uid]
        for w, sample_id in enumerate(rows["sample_id"]):
            seq[idx, w] = cache[sample_id]
            mask[idx, w] = True

    return uid_to_idx, seq, mask
