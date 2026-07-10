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
splits it into non-overlapping WINDOW_SECONDS windows, zero-padding a short
trailing window (or a whole segment shorter than one window) rather than
dropping it -- see utils/dataset_processing.py's write_windows. Subjects are
processed in parallel (see --n_jobs).

Each window is saved as a float64 array of shape (n_signals, n_samples) to
    {out_dir}/signals/{uid}_{condition}_{segment}_{window}.npy
and gets one row in the manifest with columns:
    sample_id, signal_path, label, condition, uid, segment, window, n_padding

Signals used (10 channels): chest ACC x/y/z, ECG, EMG, EDA, Temp, Resp
(all natively 700 Hz), plus wrist EDA and wrist TEMP (natively 4 Hz). Each
group is resampled independently (so their differing native rates don't need
to match) and then stacked channel-wise.

Segments (not individual windows) are randomly assigned to train/test so that
windows from the same segment never leak across the split; see
utils/dataset_processing.py for the shared --split_mode/--train_split
semantics (subject_dependent vs subject_independent) used by all dataset
conversion scripts in this repo.
"""

import glob
import os
import pickle
import sys
import numpy as np
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NormWear.modules.signal_preprocess import preproc_all
from utils.dataset_processing import find_segments, write_windows, assign_split, run_dataset_processing, add_common_args

CHEST_SAMPLING_RATE = 700  # Hz, RespiBAN chest device
WRIST_SAMPLING_RATE = 4  # Hz, Empatica E4 EDA/TEMP channels used here
TARGET_SAMPLING_RATE = 65  # Hz, expected input rate of the NormWear model
WINDOW_SECONDS = 6  # window length fed to NormWear, matching its pretraining segment length
WINDOW_SIZE = WINDOW_SECONDS * TARGET_SAMPLING_RATE  # samples per window

# study protocol label -> (condition name, class index); 0/4/5/6/7 are dropped
LABEL_TO_CONDITION = {1: ("baseline", 0), 2: ("stress", 1), 3: ("amusement", 2), 4: ("meditation", 3), }


def uid_of(pkl_path: str) -> str:
    return os.path.splitext(os.path.basename(pkl_path))[0]  # e.g. "S2"


def process_subject(pkl_path: str, out_dir: str, train_split: float, subject_split: str = None) -> tuple[dict, int]:
    """Process one subject's pickle file. Runs in a worker process, so it
    must read/write everything itself and return its manifest rows rather
    than mutating shared state."""
    uid = uid_of(pkl_path)

    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")

    chest = d["signal"]["chest"]
    wrist = d["signal"]["wrist"]
    label = np.asarray(d["label"]).reshape(-1)

    chest_stack = np.concatenate(
        [chest["ACC"], chest["ECG"], chest["EMG"], chest["EDA"], chest["Temp"], chest["Resp"]], axis=1
    ).T.astype(np.float64)  # shape: (8, n_chest_samples)
    wrist_eda, wrist_temp = wrist["EDA"], wrist["TEMP"]

    manifest_rows = {"train": [], "test": []}
    n_written = 0
    seg_counter = 0  # unique per subject, running across all conditions (not reset per condition)
    for raw_label, (condition, class_idx) in LABEL_TO_CONDITION.items():
        for start, end in find_segments(label, raw_label):
            seg_i = seg_counter
            seg_counter += 1
            if end - start < 2:
                # preproc_all's outlier-removal step diffs adjacent samples, so it
                # needs at least 2 raw samples to run; write_windows zero-pads
                # anything shorter than a full window once it comes back out.
                print(f"[skip] {uid} {condition} segment {seg_i}: fewer than 2 raw samples")
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

            split = assign_split(train_split, subject_split)
            extra_fields = {
                "label": class_idx,
                "condition": condition,
                "uid": uid,
                "segment": seg_i,
            }
            n_written += write_windows(data, f"{uid}_{condition}_{seg_i}", out_dir, WINDOW_SIZE, split, extra_fields, manifest_rows)

    return manifest_rows, n_written


def main(raw_dir: str, out_dir: str, train_split: float, split_mode: str = "subject_dependent", n_jobs: int = None):
    pkl_files = sorted(glob.glob(os.path.join(raw_dir, "S*", "S*.pkl")))
    run_dataset_processing(
        pkl_files, process_subject, uid_of, out_dir, train_split, split_mode,
        n_jobs=n_jobs, desc="Processing subjects",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw_dir', type=str, default='data/WESAD_RAW', help='Directory containing per-subject SX/SX.pkl raw WESAD files.')
    parser.add_argument('--out_dir', type=str, default='data/WESAD', help='Directory to save the manifest CSVs and .npy signal files.')
    add_common_args(parser)
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.train_split, args.split_mode, args.n_jobs)
