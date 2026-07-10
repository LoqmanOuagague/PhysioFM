import pickle

import numpy as np
import pandas as pd

from utils import process_wesad as pw

CHEST_HZ = pw.CHEST_SAMPLING_RATE
WRIST_HZ = pw.WRIST_SAMPLING_RATE


def _write_subject_pickle(path, seconds_per_condition=10, labels=(1, 2, 3)):
    n_chest = seconds_per_condition * CHEST_HZ * len(labels)
    label = np.concatenate([np.full(seconds_per_condition * CHEST_HZ, l, dtype=np.int64) for l in labels])
    chest = {
        "ACC": np.random.randn(n_chest, 3),
        "ECG": np.random.randn(n_chest, 1),
        "EMG": np.random.randn(n_chest, 1),
        "EDA": np.random.randn(n_chest, 1),
        "Temp": np.random.randn(n_chest, 1),
        "Resp": np.random.randn(n_chest, 1),
    }
    n_wrist = seconds_per_condition * WRIST_HZ * len(labels)
    wrist = {"EDA": np.random.randn(n_wrist, 1), "TEMP": np.random.randn(n_wrist, 1)}
    d = {"signal": {"chest": chest, "wrist": wrist}, "label": label}
    with open(path, "wb") as f:
        pickle.dump(d, f)


def test_uid_of_strips_pkl_extension():
    assert pw.uid_of("/data/WESAD_RAW/S7/S7.pkl") == "S7"


def test_process_subject_writes_all_three_conditions(tmp_path):
    (tmp_path / "out" / "signals").mkdir(parents=True)
    pkl_path = tmp_path / "S2.pkl"
    _write_subject_pickle(pkl_path)

    rows, n_written = pw.process_subject(str(pkl_path), str(tmp_path / "out"), train_split=1.0, subject_split=None)

    assert n_written > 0
    assert {row["condition"] for row in rows["train"]} == {"baseline", "stress", "amusement"}
    for row in rows["train"]:
        assert row["uid"] == "S2"
        saved = np.load(tmp_path / "out" / row["signal_path"])
        assert saved.shape == (10, pw.WINDOW_SIZE)


def test_process_subject_maps_condition_to_label_index(tmp_path):
    (tmp_path / "out" / "signals").mkdir(parents=True)
    pkl_path = tmp_path / "S2.pkl"
    _write_subject_pickle(pkl_path)

    rows, _ = pw.process_subject(str(pkl_path), str(tmp_path / "out"), train_split=1.0, subject_split=None)

    label_by_condition = {row["condition"]: row["label"] for row in rows["train"]}
    assert label_by_condition == {"baseline": 0, "stress": 1, "amusement": 2}


def test_process_subject_pads_too_short_segments_instead_of_dropping(tmp_path):
    (tmp_path / "out" / "signals").mkdir(parents=True)
    pkl_path = tmp_path / "S3.pkl"
    _write_subject_pickle(pkl_path, seconds_per_condition=1)  # well under WINDOW_SECONDS

    rows, n_written = pw.process_subject(str(pkl_path), str(tmp_path / "out"), train_split=1.0, subject_split=None)

    assert n_written == 3  # one padded window per condition, instead of being dropped
    for row in rows["train"]:
        assert row["n_padding"] > 0
        saved = np.load(tmp_path / "out" / row["signal_path"])
        assert saved.shape == (10, pw.WINDOW_SIZE)
        np.testing.assert_array_equal(saved[:, -row["n_padding"]:], 0)


def test_process_subject_skips_segments_with_fewer_than_two_raw_samples(tmp_path, capsys):
    (tmp_path / "out" / "signals").mkdir(parents=True)
    pkl_path = tmp_path / "S6.pkl"
    label = np.array([0, 1, 0, 0], dtype=np.int64)  # a single-sample run of label 1 (baseline)
    n_chest = len(label)
    chest = {
        "ACC": np.random.randn(n_chest, 3),
        "ECG": np.random.randn(n_chest, 1),
        "EMG": np.random.randn(n_chest, 1),
        "EDA": np.random.randn(n_chest, 1),
        "Temp": np.random.randn(n_chest, 1),
        "Resp": np.random.randn(n_chest, 1),
    }
    wrist = {"EDA": np.random.randn(4, 1), "TEMP": np.random.randn(4, 1)}
    d = {"signal": {"chest": chest, "wrist": wrist}, "label": label}
    with open(pkl_path, "wb") as f:
        pickle.dump(d, f)

    rows, n_written = pw.process_subject(str(pkl_path), str(tmp_path / "out"), train_split=1.0, subject_split=None)

    assert n_written == 0
    assert rows == {"train": [], "test": []}
    assert "fewer than 2 raw samples" in capsys.readouterr().out


def test_process_subject_assigns_unique_segment_ids_across_conditions(tmp_path):
    (tmp_path / "out" / "signals").mkdir(parents=True)
    pkl_path = tmp_path / "S4.pkl"
    _write_subject_pickle(pkl_path)

    rows, _ = pw.process_subject(str(pkl_path), str(tmp_path / "out"), train_split=1.0, subject_split=None)

    # 3 distinct conditions in this fixture -> 3 distinct running segment ids, not reset per condition
    assert len(set(row["segment"] for row in rows["train"])) == 3


def test_process_subject_honors_explicit_subject_split(tmp_path):
    (tmp_path / "out" / "signals").mkdir(parents=True)
    pkl_path = tmp_path / "S5.pkl"
    _write_subject_pickle(pkl_path)

    # train_split=0.0 would force "test" if it were consulted; subject_split="test" must win either way
    rows, n_written = pw.process_subject(str(pkl_path), str(tmp_path / "out"), train_split=0.0, subject_split="test")

    assert n_written > 0
    assert len(rows["test"]) == n_written
    assert rows["train"] == []


def test_main_processes_subjects_in_parallel_with_no_subject_leakage(tmp_path):
    """End-to-end test of the real ProcessPoolExecutor dispatch (n_jobs=2),
    not a mocked executor: exercises actual pickling of process_subject
    across worker processes."""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    for uid in ["S2", "S3"]:
        subject_dir = raw_dir / uid
        subject_dir.mkdir(parents=True)
        _write_subject_pickle(subject_dir / f"{uid}.pkl")

    pw.main(str(raw_dir), str(out_dir), train_split=1.0, split_mode="subject_independent", n_jobs=2)

    train_df = pd.read_csv(out_dir / "train_manifest.csv")
    test_df = pd.read_csv(out_dir / "test_manifest.csv")
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert set(train_df["uid"]).isdisjoint(set(test_df["uid"]))
