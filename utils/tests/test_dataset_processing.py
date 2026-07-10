import argparse
import os

import numpy as np
import pandas as pd
import pytest

from utils import dataset_processing as dp


# ---------------------------------------------------------------------------
# fill_missing
# ---------------------------------------------------------------------------

def test_fill_missing_backfills_leading_nans():
    df = pd.DataFrame({"a": [np.nan, np.nan, 3.0, 4.0]})
    assert dp.fill_missing(df)["a"].tolist() == [3.0, 3.0, 3.0, 4.0]


def test_fill_missing_forward_fills_trailing_nans():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, np.nan]})
    assert dp.fill_missing(df)["a"].tolist() == [1.0, 2.0, 2.0, 2.0]


def test_fill_missing_interior_nan_uses_following_value():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert dp.fill_missing(df)["a"].tolist() == [1.0, 3.0, 3.0]


# ---------------------------------------------------------------------------
# find_segments
# ---------------------------------------------------------------------------

def test_find_segments_single_run():
    labels = np.array([0, 0, 2, 2, 2, 0])
    assert dp.find_segments(labels, 2) == [(2, 5)]


def test_find_segments_multiple_runs():
    labels = np.array([1, 1, 0, 1, 0, 0, 1])
    assert dp.find_segments(labels, 1) == [(0, 2), (3, 4), (6, 7)]


def test_find_segments_no_match_returns_empty():
    labels = np.array([0, 0, 0])
    assert dp.find_segments(labels, 5) == []


def test_find_segments_run_touching_both_boundaries():
    labels = np.array([3, 3, 3])
    assert dp.find_segments(labels, 3) == [(0, 3)]


# ---------------------------------------------------------------------------
# assign_split
# ---------------------------------------------------------------------------

def test_assign_split_honors_explicit_subject_split_regardless_of_train_split():
    assert dp.assign_split(0.0, subject_split="train") == "train"
    assert dp.assign_split(1.0, subject_split="test") == "test"


def test_assign_split_random_respects_extreme_probabilities():
    assert dp.assign_split(1.0) == "train"
    assert dp.assign_split(0.0) == "test"


# ---------------------------------------------------------------------------
# write_windows
# ---------------------------------------------------------------------------

def test_write_windows_pads_partial_trailing_window(tmp_path):
    os.makedirs(tmp_path / "signals")
    data = np.arange(3 * 25).reshape(3, 25).astype(np.float64)  # 25 samples, window_size 10 -> 2 full + 1 padded
    manifest_rows = {"train": [], "test": []}

    n = dp.write_windows(data, "uid_task", str(tmp_path), window_size=10, split="train",
                          extra_fields={"uid": "uid"}, manifest_rows=manifest_rows)

    assert n == 3
    assert len(manifest_rows["train"]) == 3
    assert manifest_rows["test"] == []
    for w in range(2):
        row = manifest_rows["train"][w]
        assert row["sample_id"] == f"uid_task_{w}"
        assert row["window"] == w
        assert row["uid"] == "uid"
        assert row["n_padding"] == 0
        saved = np.load(tmp_path / row["signal_path"])
        assert saved.shape == (3, 10)
        np.testing.assert_array_equal(saved, data[:, w * 10:(w + 1) * 10])

    last_row = manifest_rows["train"][2]
    assert last_row["n_padding"] == 5
    saved = np.load(tmp_path / last_row["signal_path"])
    assert saved.shape == (3, 10)
    np.testing.assert_array_equal(saved[:, :5], data[:, 20:25])
    np.testing.assert_array_equal(saved[:, 5:], np.zeros((3, 5)))


def test_write_windows_short_segment_is_zero_padded_to_one_window(tmp_path):
    os.makedirs(tmp_path / "signals")
    data = np.arange(2 * 5).reshape(2, 5).astype(np.float64) + 1  # 5 samples, window_size 10 -> 1 padded window
    manifest_rows = {"train": [], "test": []}

    n = dp.write_windows(data, "uid", str(tmp_path), window_size=10, split="train",
                          extra_fields={}, manifest_rows=manifest_rows)

    assert n == 1
    row = manifest_rows["train"][0]
    assert row["n_padding"] == 5
    saved = np.load(tmp_path / row["signal_path"])
    assert saved.shape == (2, 10)
    np.testing.assert_array_equal(saved[:, :5], data)
    np.testing.assert_array_equal(saved[:, 5:], np.zeros((2, 5)))


def test_write_windows_empty_segment_writes_nothing(tmp_path):
    os.makedirs(tmp_path / "signals")
    data = np.zeros((2, 0))
    manifest_rows = {"train": [], "test": []}

    n = dp.write_windows(data, "uid", str(tmp_path), window_size=10, split="train",
                          extra_fields={}, manifest_rows=manifest_rows)

    assert n == 0
    assert manifest_rows["train"] == []
    assert not os.listdir(tmp_path / "signals")


# ---------------------------------------------------------------------------
# run_dataset_processing
#
# ProcessPoolExecutor is swapped for a synchronous stand-in so these tests
# exercise the orchestration logic (split resolution, aggregation, manifest
# writing, per-item error handling) without real subprocess/pickling
# constraints. utils/tests/test_process_cogload.py and
# test_process_wesad.py cover the real-multiprocessing path end to end.
# ---------------------------------------------------------------------------

class _ImmediateFuture:
    def __init__(self, fn, args, kwargs):
        try:
            self._result = fn(*args, **kwargs)
            self._exc = None
        except Exception as e:
            self._result = None
            self._exc = e

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class _ImmediateExecutor:
    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        return _ImmediateFuture(fn, args, kwargs)


@pytest.fixture
def immediate_executor(monkeypatch):
    monkeypatch.setattr(dp, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(dp, "as_completed", lambda futures: list(futures))


def _make_process_item_fn(fail_on=None):
    def process_item(item, out_dir, train_split, subject_split=None):
        if fail_on is not None and item == fail_on:
            raise ValueError(f"boom {item}")
        split = dp.assign_split(train_split, subject_split)
        rows = {"train": [], "test": []}
        rows[split].append({"sample_id": item, "signal_path": f"signals/{item}.npy", "uid": item, "window": 0})
        return rows, 1
    return process_item


def test_run_dataset_processing_subject_dependent(tmp_path, immediate_executor):
    items = ["a", "b", "c"]
    total = dp.run_dataset_processing(items, _make_process_item_fn(), uid_fn=lambda x: x, out_dir=str(tmp_path),
                                       train_split=1.0, split_mode="subject_dependent")

    assert total == 3
    train_df = pd.read_csv(tmp_path / "train_manifest.csv")
    assert sorted(train_df["uid"]) == ["a", "b", "c"]
    # zero manifest rows -> pd.DataFrame([]).to_csv writes a columnless (headerless) file
    assert (tmp_path / "test_manifest.csv").read_text().strip() == ""
    assert (tmp_path / "signals").is_dir()


def test_run_dataset_processing_subject_independent_isolates_one_uid(tmp_path, immediate_executor):
    items = ["a", "b", "c"]
    dp.run_dataset_processing(items, _make_process_item_fn(), uid_fn=lambda x: x, out_dir=str(tmp_path),
                               train_split=0.5, split_mode="subject_independent")

    train_df = pd.read_csv(tmp_path / "train_manifest.csv")
    test_df = pd.read_csv(tmp_path / "test_manifest.csv")
    assert len(test_df) == 1
    assert len(train_df) == 2
    assert set(train_df["uid"]).isdisjoint(set(test_df["uid"]))


def test_run_dataset_processing_continues_after_item_error(tmp_path, immediate_executor, capsys):
    items = ["a", "b", "c"]
    total = dp.run_dataset_processing(items, _make_process_item_fn(fail_on="b"), uid_fn=lambda x: x,
                                       out_dir=str(tmp_path), train_split=1.0, split_mode="subject_dependent")

    assert total == 2
    train_df = pd.read_csv(tmp_path / "train_manifest.csv")
    assert sorted(train_df["uid"]) == ["a", "c"]
    assert "[error] failed to process b" in capsys.readouterr().out


def test_run_dataset_processing_passes_item_kwargs_to_every_call(tmp_path, immediate_executor):
    def process_item(item, out_dir, train_split, subject_split=None, *, multiplier):
        rows = {"train": [{"sample_id": str(item), "signal_path": "x", "value": item * multiplier, "window": 0}],
                "test": []}
        return rows, 1

    dp.run_dataset_processing([1, 2], process_item, uid_fn=lambda x: str(x), out_dir=str(tmp_path),
                               train_split=1.0, split_mode="subject_dependent", item_kwargs={"multiplier": 10})

    train_df = pd.read_csv(tmp_path / "train_manifest.csv")
    assert sorted(train_df["value"]) == [10, 20]


# ---------------------------------------------------------------------------
# add_common_args
# ---------------------------------------------------------------------------

def test_add_common_args_defaults():
    parser = argparse.ArgumentParser()
    dp.add_common_args(parser)
    args = parser.parse_args([])

    assert args.train_split == 0.8
    assert args.split_mode == "subject_dependent"
    assert args.n_jobs is None


def test_add_common_args_rejects_invalid_split_mode():
    parser = argparse.ArgumentParser()
    dp.add_common_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--split_mode", "bogus"])
