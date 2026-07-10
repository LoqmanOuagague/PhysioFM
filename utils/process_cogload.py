"""
Convert the CogLoad1 train dataset into a manifest.csv + .npy signal dataset,
in the format expected by physioMoE.data.dataset.PhysioTLXDataset.

For every participant CSV in train/raw/ (e.g. `iz2ps_sensors.csv`) and every
task/level segment reported for that participant in personality_performance.csv,
this script resamples the segment to TARGET_SAMPLING_RATE and splits it into
non-overlapping WINDOW_SECONDS windows, zero-padding a short trailing window
(or a whole segment shorter than one window) rather than dropping it -- see
utils/dataset_processing.py's write_windows. Participants are processed in
parallel (see --n_jobs).

Each window is saved as a float64 array of shape (n_signals, n_samples) to
    {out_dir}/signals/{uid}_{task}_{level}_{window}.npy
and gets one row in the manifest with columns:

    sample_id, task_text, signal_path,
    mental_demand, physical_demand, temporal_demand, performance, effort, frustration,
    uid, task, level, window, n_padding, <personality/demographic columns>

Segments (not individual windows) are randomly assigned to train/test so that
windows from the same segment never leak across the split; see
utils/dataset_processing.py for the shared --split_mode/--train_split
semantics (subject_dependent vs subject_independent) used by all dataset
conversion scripts in this repo.
"""

import glob
import os
import sys
import numpy as np
import pandas as pd
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NormWear.modules.signal_preprocess import preproc_all
from utils.dataset_processing import fill_missing, write_windows, assign_split, run_dataset_processing, add_common_args

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


def uid_of(sensors_path: str) -> str:
    return os.path.basename(sensors_path).split("_")[0]


def process_participant(sensors_path: str, out_dir: str, train_split: float, subject_split: str = None,
                         *, performance: pd.DataFrame) -> tuple[dict, int]:
    """Process one participant's sensor CSV. Runs in a worker process, so it
    must read/write everything itself and return its manifest rows rather
    than mutating shared state."""
    uid = uid_of(sensors_path)
    df = pd.read_csv(sensors_path)

    manifest_rows = {"train": [], "test": []}
    participant_labels = performance[performance["client_id"] == uid]
    if participant_labels.empty:
        print(f"[skip] no personality_performance rows for {uid}")
        return manifest_rows, 0

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

        labels_explicit = {short: row[col] for col, short in TLX_DIMENSIONS.items()}
        context = {col: row[col] for col in PERSONALITY_COLUMNS}
        task_text = f"Cognitive workload task '{task}' at difficulty level {level}"

        split = assign_split(train_split, subject_split)
        extra_fields = {
            "task_text": task_text,
            **labels_explicit,
            "uid": uid,
            "task": task,
            "level": level,
            **context,
        }
        n_written += write_windows(data, f"{uid}_{task}_{level}", out_dir, WINDOW_SIZE, split, extra_fields, manifest_rows)

    return manifest_rows, n_written


def main(raw_dir: str, out_dir: str, performance_csv: str, train_split: float,
          split_mode: str = "subject_dependent", n_jobs: int = None):
    performance = pd.read_csv(performance_csv)

    sensor_files = sorted(
        f for f in glob.glob(os.path.join(raw_dir, "*_sensors.csv"))
        if not os.path.basename(f).startswith("merged")
    )
    run_dataset_processing(
        sensor_files, process_participant, uid_of, out_dir, train_split, split_mode,
        item_kwargs={"performance": performance}, n_jobs=n_jobs, desc="Processing participants",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw_dir', type=str, help='Directory containing raw sensor CSV files.')
    parser.add_argument('--out_dir', type=str, help='Directory to save the manifest CSVs and .npy signal files.')
    parser.add_argument('--performance_csv', type=str, help='CSV file containing personality and performance data.')
    add_common_args(parser)
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.performance_csv, args.train_split, args.split_mode, args.n_jobs)
