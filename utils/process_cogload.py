"""
Convert the CogLoad1 train dataset into a manifest.csv + .npy signal dataset,
in the format expected by physioMoE.data.dataset.PhysioTLXDataset.

For every participant CSV in train/raw/ (e.g. `iz2ps_sensors.csv`) and every
task/level segment reported for that participant in personality_performance.csv,
this script resamples the segment to TARGET_SAMPLING_RATE and splits it into
non-overlapping WINDOW_SECONDS windows (a trailing partial window is dropped).

Each window is saved as a float64 array of shape (n_signals, n_samples) to
    {out_dir}/signals/{uid}_{task}_{level}_{window}.npy
and gets one row in the manifest with columns:

    sample_id, task_text, signal_path,
    mental_demand, physical_demand, temporal_demand, performance, effort, frustration,
    uid, task, level, window, <personality/demographic columns>

Segments (not individual windows) are randomly assigned to train/test so that
windows from the same segment never leak across the split. Two split modes
are supported via --split_mode:
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
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from random import binomialvariate, choice
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NormWear.modules.signal_preprocess import preproc_all

RAW_SAMPLING_RATE = 1  # Hz, sensor CSVs are sampled roughly once per second
TARGET_SAMPLING_RATE = 65  # Hz, expected input rate of the NormWear model
WINDOW_SECONDS = 6  # window length fed to NormWear, matching its pretraining segment length
WINDOW_SIZE = WINDOW_SECONDS * TARGET_SAMPLING_RATE  # samples per window

SIGNAL_COLUMNS = [
    "hr",
    "gsr",
    "rr",
    "temperature",
    "band_ax",
    "band_ay",
    "band_az",
    #"opacity_median",
    #"opacity_std",
]

TLX_DIMENSIONS = {
    "TLX_mental_demand": "mental_demand",
    "TLX_physical_demand": "physical_demand",
    "TLX_temporal_demand": "temporal_demand",
    "TLX_performance": "performance",
    "TLX_effort": "effort",
    "TLX_frustration": "frustration",
}
PERSONALITY_COLUMNS = ['sincerity', 'fairness','age','sex_male',
       'greed_avoidance', 'modesty', 'fearfulness', 'anxiety', 'dependence',
       'sentimentality', 'social_self_esteem', 'social_boldness',
       'sociability', 'liveliness', 'forgiveness', 'gentleness', 'flexibility',
       'patience', 'organization', 'diligence', 'perfectionism', 'prudence',
       'aesthetic_appreciation', 'inquisitiveness', 'creativity',
       'unconventionality', 'honesty', 'emotionality', 'extraversion',
       'agreeableness', 'conscientiousness', 'openness','education']

manifest_rows = {"train": [], "test": []}


def fill_missing(segment: pd.DataFrame) -> pd.DataFrame:
    # replace a missing value with the following value, or the previous one if there is no following value
    return segment.bfill().ffill()


def process_participant(sensors_path: str, performance: pd.DataFrame, out_dir: str, train_split: float, subject_split: str = None) -> int:
    global manifest_rows
    uid = os.path.basename(sensors_path).split("_")[0]
    df = pd.read_csv(sensors_path)

    participant_labels = performance[performance["client_id"] == uid]
    if participant_labels.empty:
        print(f"[skip] no personality_performance rows for {uid}")
        return 0

    n_written = 0
    for _, row in participant_labels.iterrows():
        task = row["task"]
        level = int(row["level"])

        segment = df[(df["task"] == task) & (df["level"] == str(level))]
        if segment.empty:
            print(f"[skip] {uid} {task} level {level}: no matching sensor rows")
            continue

        signals = fill_missing(segment[SIGNAL_COLUMNS])
        data = signals.to_numpy(dtype=np.float64).T  # shape: (n_signals, n_samples)
        data = preproc_all(data, ss=RAW_SAMPLING_RATE, ts=TARGET_SAMPLING_RATE)

        n_windows = data.shape[1] // WINDOW_SIZE
        if n_windows == 0:
            print(f"[skip] {uid} {task} level {level}: segment shorter than {WINDOW_SECONDS}s window")
            continue

        labels_explicit = {short: row[col] for col, short in TLX_DIMENSIONS.items()}
        context = {col: row[col] for col in PERSONALITY_COLUMNS}
        task_text = f"Cognitive workload task '{task}' at difficulty level {level}"

        # in subject-independent mode the whole participant was already assigned
        # a split; otherwise randomly assign this segment to train or test, so
        # that windows from the same segment never leak across the split
        split = subject_split if subject_split is not None else ("train" if binomialvariate(1, train_split) == 1 else "test")

        for w in range(n_windows):
            sample_id = f"{uid}_{task}_{level}_{w}"
            window_data = data[:, w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]

            signal_path = os.path.join("signals", f"{sample_id}.npy")
            np.save(os.path.join(out_dir, signal_path), window_data)

            manifest_row = {
                "sample_id": sample_id,
                "task_text": task_text,
                "signal_path": signal_path,
                **labels_explicit,
                "uid": uid,
                "task": task,
                "level": level,
                "window": w,
                **context,
            }
            manifest_rows[split].append(manifest_row)
            n_written += 1

    return n_written


def main(raw_dir: str, out_dir: str, performance_csv: str, train_split: float, split_mode: str = "subject_dependent"):
    os.makedirs(os.path.join(out_dir, "signals"), exist_ok=True)
    performance = pd.read_csv(performance_csv)

    sensor_files = sorted(
        f for f in glob.glob(os.path.join(raw_dir, "*_sensors.csv"))
        if not os.path.basename(f).startswith("merged")
    )
    print(f"Found {len(sensor_files)} participant sensor files in {raw_dir}")
    print(f"Using {split_mode} split")

    test_uid = None
    if split_mode == "subject_independent":
        test_uid = choice([os.path.basename(f).split("_")[0] for f in sensor_files])
        print(f"Isolating subject {test_uid} as the test set; --train_split is ignored")

    total = 0
    for path in tqdm(sensor_files, desc="Processing participants"):
        # in subject-independent mode, the whole participant is assigned to a
        # single split so that no subject's segments leak across train and test
        if split_mode == "subject_independent":
            uid = os.path.basename(path).split("_")[0]
            subject_split = "test" if uid == test_uid else "train"
        else:
            subject_split = None
        total += process_participant(path, performance, out_dir, train_split, subject_split)

    for split, rows in manifest_rows.items():
        manifest_path = os.path.join(out_dir, f"{split}_manifest.csv")
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        print(f"Wrote {len(rows)} rows to {manifest_path}")

    print(f"Done. Wrote {total} .npy files to {os.path.join(out_dir, 'signals')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw_dir', type=str, help='Directory containing raw sensor CSV files.')
    parser.add_argument('--out_dir', type=str, help='Directory to save the manifest CSVs and .npy signal files.')
    parser.add_argument('--performance_csv', type=str, help='CSV file containing personality and performance data.')
    parser.add_argument('--train_split', type=float, default=0.8, help='Proportion (between 0 and 1) of segments to include in the training set. Ignored in subject_independent mode.')
    parser.add_argument('--split_mode', type=str, default='subject_dependent', choices=['subject_dependent', 'subject_independent'],
                         help="'subject_dependent' randomly splits individual segments into train/test (a subject can appear in both). "
                              "'subject_independent' randomly isolates one subject as the test set and puts every other subject in train "
                              "(--train_split is ignored).")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.performance_csv, args.train_split, args.split_mode)
