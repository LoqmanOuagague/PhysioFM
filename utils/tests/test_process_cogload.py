import os

import numpy as np
import pandas as pd

from utils import process_cogload as pc

SIGNAL_COLUMNS = pc.SIGNAL_COLUMNS


def _performance_row(uid, task, level):
    row = {
        "client_id": uid, "task": task, "level": level,
        "TLX_mental_demand": 1, "TLX_physical_demand": 2, "TLX_temporal_demand": 3,
        "TLX_performance": 4, "TLX_effort": 5, "TLX_frustration": 6,
    }
    row.update({col: 1 for col in pc.PERSONALITY_COLUMNS})
    return row


def _write_sensor_csv(path, segments, n_seconds=20):
    """segments: list of (task, level) tuples; writes n_seconds rows of
    random signal data per segment. Also writes one row with a non-numeric
    level, which mirrors real raw CogLoad CSVs and keeps pandas from
    inferring the level column as int64 on read-back (process_participant
    matches segments via `df["level"] == str(level)`)."""
    rows = []
    for task, level in segments:
        for _ in range(n_seconds):
            row = {"task": task, "level": str(level)}
            row.update({c: np.random.randn() for c in SIGNAL_COLUMNS})
            rows.append(row)
    extra = {"task": "REST", "level": "rest"}
    extra.update({c: np.random.randn() for c in SIGNAL_COLUMNS})
    rows.append(extra)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_uid_of_parses_prefix_before_first_underscore():
    assert pc.uid_of("/some/dir/iz2ps_sensors.csv") == "iz2ps"


def test_process_participant_writes_windows_and_manifest_rows(tmp_path):
    os.makedirs(tmp_path / "out" / "signals")
    sensors_path = tmp_path / "p1_sensors.csv"
    _write_sensor_csv(sensors_path, [("NC", 0), ("HP", 1)])
    performance = pd.DataFrame([_performance_row("p1", "NC", 0), _performance_row("p1", "HP", 1)])

    rows, n_written = pc.process_participant(str(sensors_path), str(tmp_path / "out"), train_split=1.0,
                                              subject_split=None, performance=performance)

    assert n_written > 0
    assert len(rows["train"]) == n_written
    assert rows["test"] == []
    for row in rows["train"]:
        assert row["uid"] == "p1"
        assert row["mental_demand"] == 1
        assert row["task_text"].startswith("Cognitive workload task")
        saved = np.load(tmp_path / "out" / row["signal_path"])
        assert saved.shape == (len(SIGNAL_COLUMNS), pc.WINDOW_SIZE)


def test_process_participant_skips_when_no_performance_rows(tmp_path, capsys):
    os.makedirs(tmp_path / "out" / "signals")
    sensors_path = tmp_path / "p1_sensors.csv"
    _write_sensor_csv(sensors_path, [("NC", 0)])
    performance = pd.DataFrame([_performance_row("someone_else", "NC", 0)])

    rows, n_written = pc.process_participant(str(sensors_path), str(tmp_path / "out"), train_split=1.0,
                                              subject_split=None, performance=performance)

    assert n_written == 0
    assert rows == {"train": [], "test": []}
    assert "no personality_performance rows for p1" in capsys.readouterr().out


def test_process_participant_skips_segment_with_no_sensor_rows(tmp_path, capsys):
    os.makedirs(tmp_path / "out" / "signals")
    sensors_path = tmp_path / "p1_sensors.csv"
    _write_sensor_csv(sensors_path, [("NC", 0)])
    performance = pd.DataFrame([_performance_row("p1", "HP", 1)])  # task/level absent from the sensors CSV

    rows, n_written = pc.process_participant(str(sensors_path), str(tmp_path / "out"), train_split=1.0,
                                              subject_split=None, performance=performance)

    assert n_written == 0
    assert "no matching sensor rows" in capsys.readouterr().out


def test_process_participant_honors_explicit_subject_split(tmp_path):
    os.makedirs(tmp_path / "out" / "signals")
    sensors_path = tmp_path / "p1_sensors.csv"
    _write_sensor_csv(sensors_path, [("NC", 0)])
    performance = pd.DataFrame([_performance_row("p1", "NC", 0)])

    # train_split=0.0 would force "test" if it were consulted; subject_split="test" must win either way
    rows, n_written = pc.process_participant(str(sensors_path), str(tmp_path / "out"), train_split=0.0,
                                              subject_split="test", performance=performance)

    assert n_written > 0
    assert len(rows["test"]) == n_written
    assert rows["train"] == []


def test_main_processes_participants_in_parallel_with_no_subject_leakage(tmp_path):
    """End-to-end test of the real ProcessPoolExecutor dispatch (n_jobs=2),
    not a mocked executor: exercises actual pickling of process_participant
    and the performance DataFrame across worker processes."""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()

    perf_rows = []
    for uid in ["p1", "p2", "p3"]:
        _write_sensor_csv(raw_dir / f"{uid}_sensors.csv", [("NC", 0), ("HP", 1)])
        perf_rows.append(_performance_row(uid, "NC", 0))
        perf_rows.append(_performance_row(uid, "HP", 1))
    performance_csv = raw_dir / "personality_performance.csv"
    pd.DataFrame(perf_rows).to_csv(performance_csv, index=False)

    pc.main(str(raw_dir), str(out_dir), str(performance_csv), train_split=1.0,
            split_mode="subject_independent", n_jobs=2)

    train_df = pd.read_csv(out_dir / "train_manifest.csv")
    test_df = pd.read_csv(out_dir / "test_manifest.csv")
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert set(train_df["uid"]).isdisjoint(set(test_df["uid"]))
