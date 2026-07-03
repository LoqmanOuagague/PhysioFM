"""
Convert the CogLoad1 train dataset into one pickle file per (participant, task, level, window).

For every participant CSV in train/raw/ (e.g. `iz2ps_sensors.csv`) and every
task/level segment reported for that participant in personality_performance.csv,
this script resamples the segment to TARGET_SAMPLING_RATE and splits it into
non-overlapping WINDOW_SECONDS windows (a trailing partial window is dropped).
Each window builds a dictionary:

    {
        "uid": str,
        "task": str,
        "level": int,
        "window": int,        # index of the window within the (uid, task, level) segment.
        "data": np.ndarray,   # shape (n_signals, n_samples), stacked physiological signals.
        "labels": dict,       # the six NASA-TLX dimensions for this segment.
        "context": dict,      # the 32 personality traits for this participant, gender, and age of the participant.
    }

and stores it as train/processed/{uid}_{task}_{level}_{window}.pkl
"""

import glob
import os
import pickle
import sys
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from random import binomialvariate
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
train_test_split = {"train": [], "test": []}
def fill_missing(segment: pd.DataFrame) -> pd.DataFrame:
    # replace a missing value with the following value, or the previous one if there is no following value
    return segment.bfill().ffill()


def process_participant(sensors_path: str, performance: pd.DataFrame, out_dir: str, train_split: float) -> int:
    global train_test_split
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
        labels = [labels_explicit[short] for short in TLX_DIMENSIONS.values()]
        context = {col: row[col] for col in PERSONALITY_COLUMNS}
        for w in range(n_windows):
            # randomly assign this window to train or test split 
            is_in_train = binomialvariate(1, train_split) == 1
            if is_in_train:
                train_test_split["train"].append(f"{uid}_{task}_{level}_{w}.pkl")
            else:
                train_test_split["test"].append(f"{uid}_{task}_{level}_{w}.pkl")
            
            window_data = data[:, w * WINDOW_SIZE: (w + 1) * WINDOW_SIZE]
            record = {
                "uid": uid,
                "task": task,
                "level": level,
                "window": w,
                "data": window_data,
                "sampling_rate": TARGET_SAMPLING_RATE,
                "labels_explicit": labels_explicit,
                "label":[{"reg":labels}],
                "context": context,
            }

            out_path = os.path.join(out_dir, "sample_for_downstream", f"{uid}_{task}_{level}_{w}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(record, f)
            n_written += 1

    return n_written


def main(raw_dir: str, out_dir: str, performance_csv: str, train_split: float):
    os.makedirs(os.path.join(out_dir, "sample_for_downstream"), exist_ok=True)
    performance = pd.read_csv(performance_csv)

    sensor_files = sorted(
        f for f in glob.glob(os.path.join(raw_dir, "*_sensors.csv"))
        if not os.path.basename(f).startswith("merged")
    )
    print(f"Found {len(sensor_files)} participant sensor files in {raw_dir}")
    total = 0
    for path in tqdm(sensor_files, desc="Processing participants"):
        total += process_participant(path, performance,out_dir, train_split)
    with open(os.path.join(out_dir, "train_test_split.json"), "w") as f:
        json.dump(train_test_split, f, indent=2)
    print(f"Done. Wrote {total} pickle files to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the CogLoad1 train dataset into one pickle file per (participant, task, level).")
    parser.add_argument('--raw_dir', type=str, help='Directory containing raw sensor CSV files.')
    parser.add_argument('--out_dir', type=str, help='Directory to save processed pickle files.')
    parser.add_argument('--performance_csv', type=str, help='CSV file containing personality and performance data.')
    parser.add_argument('--train_split', type=float, default=0.8, help='Proportion (between 0 and 1) of data (more specifically segments) to include in training set.')
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.performance_csv, args.train_split)
