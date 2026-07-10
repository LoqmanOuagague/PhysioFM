"""
Shared machinery for converting raw physiological datasets into the
manifest.csv + .npy window format expected by physioMoE.data.dataset.*.

A dataset-specific script (process_cogload.py, process_wesad.py, ...) only
needs to provide:
    - `items`: a list of participant identifiers (e.g. raw file paths)
    - `uid_fn(item) -> str`: extracts a subject id from an item, used to keep
      a subject's windows on one side of the split in subject_independent mode
    - `process_item_fn(item, out_dir, train_split, subject_split=None, **item_kwargs)`:
      reads one participant's raw file, resamples/windows its signal
      segments (typically via `write_windows` below) and returns
      `(rows, n_written)` where `rows` is a {"train": [...], "test": [...]}
      dict of manifest row dicts.

`run_dataset_processing` then drives the whole conversion: it resolves the
train/test split, dispatches `process_item_fn` across participants in
parallel worker processes, collects the returned manifest rows, and writes
{out_dir}/{split}_manifest.csv.

Adding support for a new dataset should only require writing a new
`process_item_fn` (and, if needed, a `find_segments`-style helper); the
split logic and parallel dispatch are shared.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import binomialvariate, choice

import numpy as np
import pandas as pd
from tqdm import tqdm


def fill_missing(segment: pd.DataFrame) -> pd.DataFrame:
    # replace a missing value with the following value, or the previous one if there is no following value
    return segment.bfill().ffill()


def find_segments(label_array: np.ndarray, target_label) -> list[tuple[int, int]]:
    """Return [(start, end), ...] index ranges (end exclusive) of contiguous
    runs where label_array == target_label. A label can occur in more than
    one contiguous run (e.g. if a study protocol revisits a condition), so
    this can return multiple ranges for the same label."""
    mask = (label_array == target_label).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask, [0]))))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def assign_split(train_split: float, subject_split: str = None) -> str:
    """Pick "train" or "test" for one segment. In subject_independent mode
    the whole participant was already assigned a split upstream (pass it as
    `subject_split`); otherwise randomly assign so that, on average,
    `train_split` of segments land in train."""
    if subject_split is not None:
        return subject_split
    return "train" if binomialvariate(1, train_split) == 1 else "test"


def write_windows(data: np.ndarray, sample_id_prefix: str, out_dir: str, window_size: int,
                   split: str, extra_fields: dict, manifest_rows: dict) -> int:
    """Split a resampled (n_signals, n_samples) segment into non-overlapping
    `window_size`-sample windows and save each window as
    {out_dir}/signals/{sample_id_prefix}_{w}.npy. A window shorter than
    `window_size` (the trailing window of a longer segment, or the segment's
    only window if the whole segment is shorter than one window) is
    zero-padded at the end rather than dropped; the number of padding
    samples is recorded in each manifest row's "n_padding" field (0 for full
    windows) so downstream code can mask them out. Appends one manifest row
    per window (sample_id, signal_path, window, n_padding, plus
    `extra_fields`) to manifest_rows[split]. Returns the number of windows
    written."""
    n_samples = data.shape[1]
    n_windows = -(-n_samples // window_size)  # ceil division; 0 when n_samples == 0
    for w in range(n_windows):
        window_data = data[:, w * window_size: (w + 1) * window_size]
        n_padding = window_size - window_data.shape[1]
        if n_padding > 0:
            window_data = np.pad(window_data, ((0, 0), (0, n_padding)))

        sample_id = f"{sample_id_prefix}_{w}"
        signal_path = os.path.join("signals", f"{sample_id}.npy")
        np.save(os.path.join(out_dir, signal_path), window_data)

        manifest_rows[split].append({
            "sample_id": sample_id,
            "signal_path": signal_path,
            **extra_fields,
            "window": w,
            "n_padding": n_padding,
        })
    return n_windows


def _resolve_test_uid(items: list, uid_fn, split_mode: str):
    if split_mode != "subject_independent":
        return None
    test_uid = choice([uid_fn(item) for item in items])
    print(f"Isolating subject {test_uid} as the test set; --train_split is ignored")
    return test_uid


def run_dataset_processing(items: list, process_item_fn, uid_fn, out_dir: str, train_split: float,
                            split_mode: str = "subject_dependent", item_kwargs: dict = None,
                            n_jobs: int = None, desc: str = "Processing participants") -> int:
    """Convert a whole dataset: resolve the subject_independent test subject
    (if applicable), process participants in parallel worker processes via
    `process_item_fn(item, out_dir, train_split, subject_split, **item_kwargs)`,
    merge the returned manifest rows, and write
    {out_dir}/{train,test}_manifest.csv. `process_item_fn` must be a
    module-level function (picklable) that performs all I/O itself (reading
    the raw file and writing .npy windows, e.g. via `write_windows`) and
    returns (rows: dict[str, list[dict]], n_written: int).

    `n_jobs` defaults to os.cpu_count() (ProcessPoolExecutor's default).
    """
    os.makedirs(os.path.join(out_dir, "signals"), exist_ok=True)
    item_kwargs = item_kwargs or {}

    print(f"Found {len(items)} participant files in the raw dataset")
    print(f"Using {split_mode} split, n_jobs={n_jobs or os.cpu_count()}")

    test_uid = _resolve_test_uid(items, uid_fn, split_mode)

    manifest_rows = {"train": [], "test": []}
    total = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {}
        for item in items:
            subject_split = None
            if split_mode == "subject_independent":
                subject_split = "test" if uid_fn(item) == test_uid else "train"
            future = pool.submit(process_item_fn, item, out_dir, train_split, subject_split, **item_kwargs)
            futures[future] = item

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            item = futures[future]
            try:
                rows, n_written = future.result()
            except Exception as e:
                print(f"[error] failed to process {item}: {e}")
                continue
            for split, split_rows in rows.items():
                manifest_rows[split].extend(split_rows)
            total += n_written

    for split, rows in manifest_rows.items():
        manifest_path = os.path.join(out_dir, f"{split}_manifest.csv")
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        print(f"Wrote {len(rows)} rows to {manifest_path}")

    print(f"Done. Wrote {total} .npy files to {os.path.join(out_dir, 'signals')}")
    return total


def add_common_args(parser):
    """Add the --train_split/--split_mode/--n_jobs CLI arguments shared by
    every dataset conversion script."""
    parser.add_argument('--train_split', type=float, default=0.8,
                         help='Proportion (between 0 and 1) of segments to include in the training set. '
                              'Ignored in subject_independent mode.')
    parser.add_argument('--split_mode', type=str, default='subject_dependent',
                         choices=['subject_dependent', 'subject_independent'],
                         help="'subject_dependent' randomly splits individual segments into train/test (a subject can appear in both). "
                              "'subject_independent' randomly isolates one subject as the test set and puts every other subject in train "
                              "(--train_split is ignored).")
    parser.add_argument('--n_jobs', type=int, default=None,
                         help='Number of participants to process in parallel (default: number of CPUs).')
    return parser
