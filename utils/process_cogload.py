"""
Convert the CogLoad1 train dataset into one pickle file per (participant, task, level).

For every participant CSV in train/raw/ (e.g. `iz2ps_sensors.csv`) and every
task/level segment reported for that participant in personality_performance.csv,
this script builds a dictionary:

    {
        "uid": str,
        "task": str,
        "level": int,
        "data": np.ndarray,   # shape (n_signals, n_samples), stacked physiological signals.
        "labels": dict,       # the six NASA-TLX dimensions for this segment.
        "context": dict,      # the 32 personality traits for this participant, gender, and age of the participant.
    }

and stores it as train/processed/{uid}_{task}_{level}.pkl
"""

import glob
import os
import pickle
import sys

from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NormWear.modules.signal_preprocess import preproc_all

RAW_SAMPLING_RATE = 1  # Hz, sensor CSVs are sampled roughly once per second
TARGET_SAMPLING_RATE = 1  # Hz, expected input rate of the NormWear model

SIGNAL_COLUMNS = [
    "hr",
    "gsr",
    "rr",
    "temperature",
    "band_ax",
    "band_ay",
    "band_az",
    "opacity_median",
    "opacity_std",
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

def fill_missing(segment: pd.DataFrame) -> pd.DataFrame:
    # replace a missing value with the following value, or the previous one if there is no following value
    return segment.bfill().ffill()


def process_participant(sensors_path: str, performance: pd.DataFrame, out_dir: str) -> int:
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

        labels = {short: row[col] for col, short in TLX_DIMENSIONS.items()}
        context = {col: row[col] for col in PERSONALITY_COLUMNS}
        record = {
            "uid": uid,
            "task": task,
            "level": level,
            "data": data,
            "sampling_rate": TARGET_SAMPLING_RATE,
            "labels": labels,
            "context": context,
        }

        out_path = os.path.join(out_dir, f"{uid}_{task}_{level}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(record, f)
        n_written += 1

    return n_written


def main(raw_dir: str, out_dir: str, performance_csv: str):
    os.makedirs(out_dir, exist_ok=True)
    performance = pd.read_csv(performance_csv)

    sensor_files = sorted(
        f for f in glob.glob(os.path.join(raw_dir, "*_sensors.csv"))
        if not os.path.basename(f).startswith("merged")
    )
    print(f"Found {len(sensor_files)} participant sensor files in {raw_dir}")

    total = 0
    for path in tqdm(sensor_files, desc="Processing participants"):
        total += process_participant(path, performance,out_dir)

    print(f"Done. Wrote {total} pickle files to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the CogLoad1 train dataset into one pickle file per (participant, task, level).")
    parser.add_argument('--raw_dir', type=str, help='Directory containing raw sensor CSV files.')
    parser.add_argument('--out_dir', type=str, help='Directory to save processed pickle files.')
    parser.add_argument('--performance_csv', type=str, help='CSV file containing personality and performance data.')
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.performance_csv)
