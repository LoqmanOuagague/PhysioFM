import numpy as np
import pandas as pd

from utils import process_adabase as pa

RATE_GROUPS = pa.RATE_GROUPS
MASTER_RATE = pa.EMG_SAMPLING_RATE  # fastest channel -> the SIGNALS master clock


def _segment_frame(study, phase, level, seconds):
    n = seconds * MASTER_RATE
    df = pd.DataFrame({"STUDY": [study] * n, "PHASE": [phase] * n, "LEVEL": [level] * n})
    for rate, cols in RATE_GROUPS.items():
        step = MASTER_RATE // rate
        for col in cols:
            values = np.full(n, np.nan)
            values[::step] = np.random.randn(len(values[::step]))
            df[col] = values
    return df


def _write_signals_h5(path, levels, seconds_per_level, study, mode):
    signals = pd.concat([_segment_frame(study, "test", level, seconds_per_level) for level in levels], ignore_index=True)
    signals["TS"] = np.arange(len(signals))
    signals.to_hdf(path, key="SIGNALS", mode=mode, format="table")


def _write_subjective_h5(path, levels, subj_study, mode):
    subjective = pd.DataFrame([
        {
            "STUDY": subj_study, "PHASE": "test", "LEVEL": f"{level:02d}",
            "EFFORT": 1.0, "FRUSTRATION": 2.0, "MENTAL": 3.0,
            "PERFORMANCE": 4.0, "PHYSICAL": 5.0, "TEMPORAL": 6.0,
        }
        for level in levels
    ])
    subjective.to_hdf(path, key="SUBJECTIVE", mode=mode, format="table")


def _write_subject_h5(path, levels=(1, 2), seconds_per_level=10, study="n-back", subj_study="nback"):
    _write_signals_h5(path, levels, seconds_per_level, study, mode="w")
    _write_subjective_h5(path, levels, subj_study, mode="a")


def test_uid_of_parses_participant_id_from_filename():
    assert pa.uid_of("/data/adabase_raw/adabase-public-0011-v_0_0_2.h5py") == "0011"


def test_process_subject_writes_windows_and_manifest_rows(tmp_path):
    (tmp_path / "out" / "signals").mkdir(parents=True)
    h5_path = tmp_path / "adabase-public-0000-v_0_0_2.h5py"
    _write_subject_h5(h5_path, levels=(1, 2))

    rows, n_written = pa.process_subject(str(h5_path), str(tmp_path / "out"), train_split=1.0, subject_split=None)

    assert n_written > 0
    assert len(rows["train"]) == n_written
    assert rows["test"] == []
    for row in rows["train"]:
        assert row["study"] == "n-back"
        assert row["mental_demand"] == 3.0
        assert row["frustration"] == 2.0
        assert row["task_text"].startswith("Cognitive workload task")
        saved = np.load(tmp_path / "out" / row["signal_path"])
        assert saved.shape == (len(pa.SIGNAL_COLUMNS), pa.WINDOW_SIZE)


def test_process_subject_translates_subjective_study_names():
    # SUBJECTIVE spells the driving study "drive" while SIGNALS spells it "k-drive"
    assert pa.SUBJECTIVE_TO_SIGNALS_STUDY["drive"] == "k-drive"
    assert pa.SUBJECTIVE_TO_SIGNALS_STUDY["nback"] == "n-back"


def test_process_subject_skips_when_no_matching_signals_rows(tmp_path, capsys):
    (tmp_path / "out" / "signals").mkdir(parents=True)
    h5_path = tmp_path / "adabase-public-0000-v_0_0_2.h5py"
    _write_signals_h5(h5_path, levels=(1,), seconds_per_level=10, study="n-back", mode="w")  # SIGNALS only has level 1
    # ask SUBJECTIVE about a level that was never recorded in SIGNALS
    _write_subjective_h5(h5_path, levels=(9,), subj_study="nback", mode="a")

    rows, n_written = pa.process_subject(str(h5_path), str(tmp_path / "out"), train_split=1.0, subject_split=None)

    assert n_written == 0
    assert rows == {"train": [], "test": []}
    assert "no matching SIGNALS rows" in capsys.readouterr().out


def test_process_subject_honors_explicit_subject_split(tmp_path):
    (tmp_path / "out" / "signals").mkdir(parents=True)
    h5_path = tmp_path / "adabase-public-0000-v_0_0_2.h5py"
    _write_subject_h5(h5_path, levels=(1,))

    # train_split=0.0 would force "test" if it were consulted; subject_split="test" must win either way
    rows, n_written = pa.process_subject(str(h5_path), str(tmp_path / "out"), train_split=0.0, subject_split="test")

    assert n_written > 0
    assert len(rows["test"]) == n_written
    assert rows["train"] == []


def test_main_processes_subjects_in_parallel_with_no_subject_leakage(tmp_path):
    """End-to-end test of the real ProcessPoolExecutor dispatch (n_jobs=2),
    not a mocked executor: exercises actual pickling of process_subject
    across worker processes."""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    for uid in ["0000", "0001"]:
        _write_subject_h5(raw_dir / f"adabase-public-{uid}-v_0_0_2.h5py", levels=(1, 2))

    pa.main(str(raw_dir), str(out_dir), train_split=1.0, split_mode="subject_independent", n_jobs=2)

    train_df = pd.read_csv(out_dir / "train_manifest.csv")
    test_df = pd.read_csv(out_dir / "test_manifest.csv")
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert set(train_df["uid"]).isdisjoint(set(test_df["uid"]))
