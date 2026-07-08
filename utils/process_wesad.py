"""
----------------------------------------------------------------------------
!!! !!! WARNING: this script opens pickle files from the WESAD dataset, which are not provided in this repository.
Please make sure you trust the source of the WESAD dataset before running this script, as pickle files can execute arbitrary code when loaded.
----------------------------------------------------------------------------
Convert the raw WESAD dataset into a manifest.csv + .npy signal dataset,
following the same method as process_cogload.py.

For every subject folder in data/WESAD_RAW/ (e.g. `S2/S2.pkl`), this script
reads the synchronised chest (RespiBAN, 700 Hz) and wrist (Empatica E4)
signals together with the per-sample study-protocol label. It keeps the
three well-defined affective conditions:
    1 = baseline, 2 = stress, 3 = amusement
(condition 4 = meditation and the transient/undefined labels 0/5/6/7 are
dropped, matching the standard WESAD 3-class stress-detection setup), locates
each condition's contiguous segment, resamples it to TARGET_SAMPLING_RATE and
splits it into non-overlapping WINDOW_SECONDS windows (a trailing partial
window is dropped).

Each window is saved as a float64 array of shape (n_signals, n_samples) to
    {out_dir}/signals/{uid}_{condition}_{segment}_{window}.npy
and gets one row in the manifest with columns:
    sample_id, signal_path, label, condition, uid, segment, window

Signals used (10 channels): chest ACC x/y/z, ECG, EMG, EDA, Temp, Resp
(all natively 700 Hz), plus wrist EDA and wrist TEMP (natively 4 Hz). Each
group is resampled independently (so their differing native rates don't need
to match) and then stacked channel-wise.

Segments (not individual windows) are randomly assigned to train/test so that
windows from the same segment never leak across the split. Two split modes
are supported via --split_mode, mirroring process_cogload.py:
    subject_dependent   (default) each segment is independently assigned to
                         train/test, so a subject's segments can appear on
                         both sides of the split.
    subject_independent one subject is randomly isolated as the test set and
                         every other subject goes to train; --train_split is
                         ignored in this mode.
Each split is written to its own manifest CSV:
    {out_dir}/train_manifest.csv
    {out_dir}/test_manifest.csv
"""

import glob
import os
import pickle
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from random import binomialvariate, choice
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NormWear.modules.signal_preprocess import preproc_all

CHEST_SAMPLING_RATE = 700  # Hz, RespiBAN chest device
WRIST_SAMPLING_RATE = 4  # Hz, Empatica E4 EDA/TEMP channels used here
TARGET_SAMPLING_RATE = 65  # Hz, expected input rate of the NormWear model
WINDOW_SECONDS = 6  # window length fed to NormWear, matching its pretraining segment length
WINDOW_SIZE = WINDOW_SECONDS * TARGET_SAMPLING_RATE  # samples per window

# study protocol label -> (condition name, class index); 0/4/5/6/7 are dropped
LABEL_TO_CONDITION = {1: ("baseline", 0), 2: ("stress", 1), 3: ("amusement", 2)}

manifest_rows = {"train": [], "test": []}


def find_segments(label_array: np.ndarray, target_label: int):
    """Return [(start, end), ...] index ranges (end exclusive) of contiguous runs where label == target_label.

    Used here to locate, within a subject's per-sample WESAD study-protocol
    label array (700 Hz, one label per chest sample), the chest-signal index
    ranges belonging to a given condition (e.g. target_label=2 for stress).
    Each returned (start, end) is then used to slice out that condition's
    chest segment (and the corresponding wrist segment) in process_subject,
    before resampling and windowing it. A condition can occur in more than
    one contiguous run per subject (e.g. if the protocol revisits it), so
    this can return multiple segments for the same label.
    """
    mask = (label_array == target_label).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask, [0]))))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def process_subject(pkl_path: str, out_dir: str, train_split: float, subject_split: str = None) -> int:
    global manifest_rows
    uid = os.path.splitext(os.path.basename(pkl_path))[0]  # e.g. "S2"

    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")

    chest = d["signal"]["chest"]
    wrist = d["signal"]["wrist"]
    label = np.asarray(d["label"]).reshape(-1)

    chest_stack = np.concatenate(
        [chest["ACC"], chest["ECG"], chest["EMG"], chest["EDA"], chest["Temp"], chest["Resp"]], axis=1
    ).T.astype(np.float64)  # shape: (8, n_chest_samples)
    wrist_eda, wrist_temp = wrist["EDA"], wrist["TEMP"]

    n_written = 0
    seg_counter = 0  # unique per subject, running across all conditions (not reset per condition)
    for raw_label, (condition, class_idx) in LABEL_TO_CONDITION.items():
        for start, end in find_segments(label, raw_label):
            seg_i = seg_counter
            seg_counter += 1
            if end - start < CHEST_SAMPLING_RATE * WINDOW_SECONDS:
                # @TODO: Pad short segments instead of dropping them? (would require a new manifest column to indicate padding)
                print(f"[skip] {uid} {condition} segment {seg_i}: shorter than {WINDOW_SECONDS}s")
                continue

            chest_segment = chest_stack[:, start:end]

            w_start = int(round(start / CHEST_SAMPLING_RATE * WRIST_SAMPLING_RATE))
            w_end = min(int(round(end / CHEST_SAMPLING_RATE * WRIST_SAMPLING_RATE)), wrist_eda.shape[0], wrist_temp.shape[0])
            if w_end <= w_start:
                print(f"[skip] {uid} {condition} segment {seg_i}: no matching wrist samples")
                continue
            wrist_segment = np.concatenate([wrist_eda[w_start:w_end], wrist_temp[w_start:w_end]], axis=1).T.astype(np.float64)

            chest_resampled = preproc_all(chest_segment, ss=CHEST_SAMPLING_RATE, ts=TARGET_SAMPLING_RATE)
            wrist_resampled = preproc_all(wrist_segment, ss=WRIST_SAMPLING_RATE, ts=TARGET_SAMPLING_RATE)

            n_samples = min(chest_resampled.shape[1], wrist_resampled.shape[1])
            data = np.concatenate([chest_resampled[:, :n_samples], wrist_resampled[:, :n_samples]], axis=0)  # (10, n_samples)

            n_windows = data.shape[1] // WINDOW_SIZE
            if n_windows == 0:
                print(f"[skip] {uid} {condition} segment {seg_i}: resampled segment shorter than {WINDOW_SECONDS}s window")
                continue

            # in subject-independent mode the whole subject was already assigned
            # a split; otherwise randomly assign this segment to train or test, so
            # that windows from the same segment never leak across the split
            split = subject_split if subject_split is not None else ("train" if binomialvariate(1, train_split) == 1 else "test")

            for w in range(n_windows):
                sample_id = f"{uid}_{condition}_{seg_i}_{w}"
                window_data = data[:, w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]

                signal_path = os.path.join("signals", f"{sample_id}.npy")
                np.save(os.path.join(out_dir, signal_path), window_data)

                manifest_rows[split].append({
                    "sample_id": sample_id,
                    "signal_path": signal_path,
                    "label": class_idx,
                    "condition": condition,
                    "uid": uid,
                    "segment": seg_i,
                    "window": w,
                })
                n_written += 1

    return n_written


def main(raw_dir: str, out_dir: str, train_split: float, split_mode: str = "subject_dependent"):
    os.makedirs(os.path.join(out_dir, "signals"), exist_ok=True)

    pkl_files = sorted(glob.glob(os.path.join(raw_dir, "S*", "S*.pkl")))
    print(f"Found {len(pkl_files)} subject files in {raw_dir}")
    print(f"Using {split_mode} split")

    test_uid = None
    if split_mode == "subject_independent":
        test_uid = choice([os.path.splitext(os.path.basename(f))[0] for f in pkl_files])
        print(f"Isolating subject {test_uid} as the test set; --train_split is ignored")

    total = 0
    for path in tqdm(pkl_files, desc="Processing subjects"):
        if split_mode == "subject_independent":
            uid = os.path.splitext(os.path.basename(path))[0]
            subject_split = "test" if uid == test_uid else "train"
        else:
            subject_split = None
        total += process_subject(path, out_dir, train_split, subject_split)

    for split, rows in manifest_rows.items():
        manifest_path = os.path.join(out_dir, f"{split}_manifest.csv")
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        print(f"Wrote {len(rows)} rows to {manifest_path}")

    print(f"Done. Wrote {total} .npy files to {os.path.join(out_dir, 'signals')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw_dir', type=str, default='data/WESAD_RAW', help='Directory containing per-subject SX/SX.pkl raw WESAD files.')
    parser.add_argument('--out_dir', type=str, default='data/WESAD', help='Directory to save the manifest CSVs and .npy signal files.')
    parser.add_argument('--train_split', type=float, default=0.8, help='Proportion (between 0 and 1) of segments to include in the training set. Ignored in subject_independent mode.')
    parser.add_argument('--split_mode', type=str, default='subject_dependent', choices=['subject_dependent', 'subject_independent'],
                         help="'subject_dependent' randomly splits individual segments into train/test (a subject can appear in both). "
                              "'subject_independent' randomly isolates one subject as the test set and puts every other subject in train "
                              "(--train_split is ignored).")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.train_split, args.split_mode)
