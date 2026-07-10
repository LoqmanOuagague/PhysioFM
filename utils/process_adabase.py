"""
Convert the raw ADABase dataset into a manifest.csv + .npy signal dataset,
following the same method as process_cogload.py.

For every subject HDF5 file in data/adabase_raw/ (e.g.
`adabase-public-0000-v_0_0_2.h5py`), this script reads the SIGNALS table.
Per the dataset documentation, SIGNALS is a single dataframe merged onto one
master clock (the fastest channel's rate): a row exists for every tick, but a
slower channel only has a real value on the ticks matching its own sampling
rate and is NaN elsewhere, so dropping NaN rows on a column recovers that
signal at its native rate.

SIGNALS is read with a hand-rolled chunked h5py reader (`read_signals_columns`
below) instead of `pd.read_hdf(..., columns=[...])`. In this dataset, pytables
packs all 57 float64 columns (our 6 raw signal channels plus 51 eye-tracking/
facial-AU columns we never use) into a single compressed block, and selecting
a column subset through pandas still materializes that entire block for every
row in memory: ~7.8GB peak RSS per participant, measured, against a ~370MB
source file. Since worker processes run --n_jobs of these concurrently, that
made the default n_jobs OOM the machine. Reading the same block through h5py
in row chunks and immediately discarding the unwanted columns brings that
down to well under 1GB per participant (and is incidentally faster too, since
it also skips pytables' query/index machinery).

Each participant also has a SUBJECTIVE table with one NASA-TLX rating per
completed n-back/k-drive test level (this plays the same role as CogLoad's
personality_performance.csv). For every SUBJECTIVE row, this script locates
the matching SIGNALS rows via STUDY/PHASE/LEVEL (note SUBJECTIVE spells STUDY
differently, e.g. "nback" vs. SIGNALS' "n-back" -- see
SUBJECTIVE_TO_SIGNALS_STUDY), extracts the raw SKT/ECG(x2)/RSP/EMG/EDA
channels (the EYE-tracking and facial-AU columns are a different modality and
are left out, matching process_wesad.py's approach of picking specific
channels rather than every column), resamples each native-rate group to
TARGET_SAMPLING_RATE independently and stacks them channel-wise, then splits
the result into non-overlapping WINDOW_SECONDS windows, zero-padding a short
trailing window (or a whole segment shorter than one window) rather than
dropping it -- see utils/dataset_processing.py's write_windows. Subjects are
processed in parallel (see --n_jobs).

Each window is saved as a float64 array of shape (n_signals, n_samples) to
    {out_dir}/signals/{uid}_{study}_{level}_{window}.npy
and gets one row in the manifest with columns:
    sample_id, task_text, signal_path,
    mental_demand, physical_demand, temporal_demand, performance, effort, frustration,
    uid, study, level, window, n_padding

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
import pandas as pd
import h5py
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NormWear.modules.signal_preprocess import preproc_all
from utils.dataset_processing import write_windows, assign_split, run_dataset_processing, add_common_args

SIGNALS_READ_CHUNK_ROWS = 250_000  # bounds peak memory of read_signals_columns to well under 1GB

SKT_SAMPLING_RATE = 100  # Hz
ECG_SAMPLING_RATE = 500  # Hz
RSP_SAMPLING_RATE = 250  # Hz
EMG_SAMPLING_RATE = 1000  # Hz
EDA_SAMPLING_RATE = 500  # Hz
TARGET_SAMPLING_RATE = 65  # Hz, expected input rate of the NormWear model
WINDOW_SECONDS = 6  # window length fed to NormWear, matching its pretraining segment length
WINDOW_SIZE = WINDOW_SECONDS * TARGET_SAMPLING_RATE  # samples per window

# Raw SIGNALS columns grouped by native sampling rate. Columns sharing a rate
# share an identical NaN fill pattern (same master-clock ticks), so they can
# be dropna()'d and resampled together; each group is then resampled
# independently to TARGET_SAMPLING_RATE and stacked channel-wise, the same
# approach process_wesad.py uses for its chest/wrist groups.
RATE_GROUPS = {
    SKT_SAMPLING_RATE: ["RAW_SKT"],
    ECG_SAMPLING_RATE: ["RAW_ECG_I", "RAW_ECG_II", "RAW_EDA"],
    RSP_SAMPLING_RATE: ["RAW_RSP"],
    EMG_SAMPLING_RATE: ["RAW_EMG"],
}
SIGNAL_COLUMNS = [col for cols in RATE_GROUPS.values() for col in cols]

# SUBJECTIVE.STUDY spells the task differently than SIGNALS.STUDY.
SUBJECTIVE_TO_SIGNALS_STUDY = {"nback": "n-back", "drive": "k-drive"}

TLX_DIMENSIONS = {
    "MENTAL": "mental_demand",
    "PHYSICAL": "physical_demand",
    "TEMPORAL": "temporal_demand",
    "PERFORMANCE": "performance",
    "EFFORT": "effort",
    "FRUSTRATION": "frustration",
}


def uid_of(h5_path: str) -> str:
    return os.path.basename(h5_path).split("-")[2]  # e.g. "0000"


def _column_block_map(table: h5py.Dataset) -> dict[str, tuple[str, int]]:
    """Map each SIGNALS column name to (pytables block field name, column
    index within that block), discovered from the table's FIELD_i_NAME /
    values_block_*_kind attributes instead of assumed. pytables groups
    same-dtype columns into "blocks" (e.g. all float64 columns share one
    block), and which columns land in which block depends on which columns
    and dtypes are actually present -- e.g. a small HDF5 file (a test
    fixture, or a future dataset revision) can partition blocks differently
    than the full 62-column ADABase file."""
    mapping = {}
    i = 0
    while f"FIELD_{i}_NAME" in table.attrs:
        field_name = table.attrs[f"FIELD_{i}_NAME"]
        if isinstance(field_name, bytes):
            field_name = field_name.decode("ascii")
        kind_attr = f"{field_name}_kind"
        # Only "values_block_*" fields carry a pickled list of column names in
        # this attribute; e.g. the non-block "index" field has its own
        # "index_kind" attribute holding a plain descriptor string ("integer"),
        # not a pickled list, and would break unpickling below.
        if field_name.startswith("values_block_") and kind_attr in table.attrs:
            for pos, col in enumerate(pickle.loads(table.attrs[kind_attr])):
                mapping[col] = (field_name, pos)
        i += 1
    return mapping


def read_signals_columns(h5_path: str, float_columns: list[str], chunk_rows: int = SIGNALS_READ_CHUNK_ROWS):
    """Read STUDY/PHASE/LEVEL plus a subset of the SIGNALS table's float64
    columns, in row chunks, without ever materializing the full 57-column
    float64 block pytables packs them into (see module docstring for why that
    matters).

    Returns (float_data: (n_rows, len(float_columns)) float64 array,
             study: (n_rows,) str array, phase: (n_rows,) str array,
             level: (n_rows,) int64 array).
    """
    with h5py.File(h5_path, "r") as f:
        table = f["SIGNALS"]["table"]
        columns = _column_block_map(table)
        float_loc = [columns[c] for c in float_columns]
        study_block, study_pos = columns["STUDY"]
        phase_block, phase_pos = columns["PHASE"]
        level_block, level_pos = columns["LEVEL"]

        n_rows = table.shape[0]
        float_data = np.empty((n_rows, len(float_columns)), dtype=np.float64)
        level = np.empty(n_rows, dtype=np.int64)
        study = np.empty(n_rows, dtype="S12")
        phase = np.empty(n_rows, dtype="S12")

        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            chunk = table[start:end]  # one read decompresses every column for these rows
            for i, (block, pos) in enumerate(float_loc):
                float_data[start:end, i] = chunk[block][:, pos]
            level[start:end] = chunk[level_block][:, level_pos]
            study[start:end] = chunk[study_block][:, study_pos]
            phase[start:end] = chunk[phase_block][:, phase_pos]

    return float_data, np.char.decode(study, "ascii"), np.char.decode(phase, "ascii"), level


def process_subject(h5_path: str, out_dir: str, train_split: float, subject_split: str = None) -> tuple[dict, int]:
    """Process one subject's HDF5 file. Runs in a worker process, so it must
    read/write everything itself and return its manifest rows rather than
    mutating shared state."""
    uid = uid_of(h5_path)
    float_data, study_col, phase_col, level_col = read_signals_columns(h5_path, SIGNAL_COLUMNS)
    column_position = {col: i for i, col in enumerate(SIGNAL_COLUMNS)}
    subjective = pd.read_hdf(h5_path, "SUBJECTIVE", mode="r")

    manifest_rows = {"train": [], "test": []}
    n_written = 0
    for _, row in subjective.iterrows():
        study = SUBJECTIVE_TO_SIGNALS_STUDY.get(row["STUDY"])
        if study is None:
            print(f"[skip] {uid}: unknown SUBJECTIVE study '{row['STUDY']}'")
            continue
        level = int(row["LEVEL"])

        mask = (study_col == study) & (phase_col == "test") & (level_col == level)
        segment = float_data[mask]
        if segment.shape[0] == 0:
            print(f"[skip] {uid} {study} level {level}: no matching SIGNALS rows")
            continue

        group_arrays = []
        for rate, cols in RATE_GROUPS.items():
            group = segment[:, [column_position[c] for c in cols]]
            group = group[~np.isnan(group).any(axis=1)]  # cols in a rate group share one NaN pattern (dropna equivalent)
            group_arrays.append(preproc_all(group.T, ss=rate, ts=TARGET_SAMPLING_RATE))  # (n_cols, n_samples)

        n_samples = min(arr.shape[1] for arr in group_arrays)
        data = np.concatenate([arr[:, :n_samples] for arr in group_arrays], axis=0)

        labels_explicit = {short: row[col] for col, short in TLX_DIMENSIONS.items()}
        task_text = f"Cognitive workload task '{study}' at difficulty level {level}"

        split = assign_split(train_split, subject_split)
        extra_fields = {
            "task_text": task_text,
            **labels_explicit,
            "uid": uid,
            "study": study,
            "level": level,
        }
        n_written += write_windows(data, f"{uid}_{study}_{level}", out_dir, WINDOW_SIZE, split, extra_fields, manifest_rows)

    return manifest_rows, n_written


def main(raw_dir: str, out_dir: str, train_split: float, split_mode: str = "subject_dependent", n_jobs: int = None):
    h5_files = sorted(glob.glob(os.path.join(raw_dir, "*.h5py")))
    run_dataset_processing(
        h5_files, process_subject, uid_of, out_dir, train_split, split_mode,
        n_jobs=n_jobs, desc="Processing subjects",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw_dir', type=str, default='data/adabase_raw', help='Directory containing raw ADABase *.h5py files.')
    parser.add_argument('--out_dir', type=str, default='data/ADABase', help='Directory to save the manifest CSVs and .npy signal files.')
    add_common_args(parser)
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.train_split, args.split_mode, args.n_jobs)
