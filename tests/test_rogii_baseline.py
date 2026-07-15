import importlib.util
import json
from pathlib import Path
import pickle
import signal
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "workspaces"
    / "rogii-wellbore-geology-prediction"
    / "scripts"
    / "rogii_baseline.py"
)
SPEC = importlib.util.spec_from_file_location("rogii_baseline", SCRIPT_PATH)
assert SPEC and SPEC.loader
rogii = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rogii
SPEC.loader.exec_module(rogii)

STATE_SPACE_PATH = (
    Path(__file__).parents[1]
    / "workspaces"
    / "rogii-wellbore-geology-prediction"
    / "scripts"
    / "rogii_state_space.py"
)
STATE_SPACE_SPEC = importlib.util.spec_from_file_location("rogii_state_space", STATE_SPACE_PATH)
assert STATE_SPACE_SPEC and STATE_SPACE_SPEC.loader
state_space = importlib.util.module_from_spec(STATE_SPACE_SPEC)
sys.modules[STATE_SPACE_SPEC.name] = state_space
STATE_SPACE_SPEC.loader.exec_module(state_space)


class ZeroResidualModel:
    best_iteration_ = 1

    def predict(self, features, num_iteration=None):
        del num_iteration
        return np.zeros(len(features), dtype=float)


def write_well(root: Path, well_id: str, offset: float = 0.0, include_target: bool = True):
    root.mkdir(parents=True, exist_ok=True)
    depth = np.arange(10, dtype=float)
    target = offset + 100.0 + 2.0 * depth
    input_target = target.copy()
    input_target[3:] = np.nan
    frame = pd.DataFrame(
        {
            "MD": depth,
            "X": depth * 3.0,
            "Y": depth * 4.0,
            "Z": -depth,
            "GR": 50.0 + depth,
            "TVT_input": input_target,
        }
    )
    if include_target:
        frame["TVT"] = target
    horizontal = root / f"{well_id}__horizontal_well.csv"
    frame.to_csv(horizontal, index=False)
    pd.DataFrame(
        {
            "TVT": np.linspace(offset + 95.0, offset + 125.0, 31),
            "GR": np.linspace(45.0, 75.0, 31),
            "Geology": ["A"] * 31,
        }
    ).to_csv(root / f"{well_id}__typewell.csv", index=False)
    return horizontal


def test_features_do_not_depend_on_target_values(tmp_path):
    horizontal = write_well(tmp_path, "well0001")
    before = rogii.build_well_features(horizontal, require_target=True)

    changed = pd.read_csv(horizontal)
    changed["TVT"] += 9999.0
    changed.to_csv(horizontal, index=False)
    after = rogii.build_well_features(horizontal, require_target=True)

    pd.testing.assert_frame_equal(before[rogii.FEATURE_COLUMNS], after[rogii.FEATURE_COLUMNS])
    assert not before["_target"].equals(after["_target"])


def test_interpolation_training_writes_complete_artifacts(tmp_path):
    train_root = tmp_path / "data" / "raw" / "train"
    for index in range(3):
        write_well(train_root, f"well{index:04d}", offset=index * 10.0)

    model_dir = rogii.train_last_known(tmp_path, n_splits=3)

    expected = {
        "run.json",
        "cv_scores.json",
        "oof_preds.npy",
        "oof_rows.parquet",
        "feature_list.txt",
        "importance.csv",
        "model.pkl",
    }
    assert expected.issubset({path.name for path in model_dir.iterdir()})
    run = pd.read_json(model_dir / "run.json", typ="series")
    scores = pd.read_json(model_dir / "cv_scores.json", typ="series")
    assert run["status"] == "completed"
    assert scores["metric"] == "rmse"
    assert scores["mean"] > 0
    assert len(np.load(model_dir / "oof_preds.npy")) == 21


def test_prediction_follows_sample_submission_order(tmp_path):
    train_root = tmp_path / "data" / "raw" / "train"
    for index in range(2):
        write_well(train_root, f"train{index:03d}", offset=index * 10.0)
    model_dir = rogii.train_last_known(tmp_path, n_splits=2)

    test_root = tmp_path / "data" / "raw" / "test"
    write_well(test_root, "test0001", include_target=False)
    sample = pd.DataFrame(
        {
            "id": [
                "test0001_6",
                "test0001_3",
                "test0001_9",
                "test0001_5",
                "test0001_4",
                "test0001_8",
                "test0001_7",
            ],
            "tvt": 0.0,
        }
    )
    sample_path = tmp_path / "data" / "raw" / "sample_submission.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False)

    submission_path = rogii.predict_last_known(tmp_path, model_dir)
    submission = pd.read_csv(submission_path)

    assert submission["id"].tolist() == sample["id"].tolist()
    assert np.allclose(submission["tvt"], 104.0)
    assert (model_dir / "test_preds.npy").exists()
    metadata = pd.read_json(submission_path.with_suffix(".json"), typ="series")
    scores = pd.read_json(model_dir / "cv_scores.json", typ="series")
    assert metadata["local_cv_score"] == scores["overall"]


def test_sample_aligned_writer_creates_submission_directory(tmp_path):
    raw_root = tmp_path / "data" / "raw"
    raw_root.mkdir(parents=True)
    pd.DataFrame({"id": ["well_2", "well_1"], "tvt": [0.0, 0.0]}).to_csv(
        raw_root / "sample_submission.csv", index=False
    )
    model_dir = tmp_path / "models" / "v001"
    model_dir.mkdir(parents=True)
    predictions = pd.DataFrame({"_id": ["well_1", "well_2"], "prediction": [10.0, 20.0]})

    path = rogii._write_sample_aligned_submission(
        tmp_path, model_dir, predictions, "fresh", {"submitted": False}
    )

    assert path.exists()
    assert pd.read_csv(path)["tvt"].tolist() == [20.0, 10.0]


def test_residual_prediction_falls_back_for_one_malformed_well(tmp_path):
    test_root = tmp_path / "data" / "raw" / "test"
    write_well(test_root, "valid", include_target=False)
    pd.DataFrame({"TVT_input": [100.0, 101.0, np.nan], "GR": [1.0, 2.0, 3.0]}).to_csv(
        test_root / "broken__horizontal_well.csv", index=False
    )
    pd.DataFrame({"TVT": [90.0, 110.0], "GR": [1.0, 2.0]}).to_csv(
        test_root / "broken__typewell.csv", index=False
    )
    sample = pd.DataFrame({"id": ["broken_2", "valid_5", "valid_3"], "tvt": [0.0, 0.0, 0.0]})
    sample_path = tmp_path / "data" / "raw" / "sample_submission.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False)

    model_dir = tmp_path / "models" / "v001"
    model_dir.mkdir(parents=True)
    with (model_dir / "models.pkl").open("wb") as handle:
        pickle.dump([ZeroResidualModel()], handle)
    (model_dir / "cv_scores.json").write_text(
        json.dumps({"overall": 1.0, "mean": 1.0}), encoding="utf-8"
    )

    submission_path = rogii.predict_residual(tmp_path, model_dir)
    submission = pd.read_csv(submission_path)

    assert submission["id"].tolist() == sample["id"].tolist()
    assert submission.loc[0, "tvt"] == 101.0
    assert np.isfinite(submission["tvt"]).all()
    metadata = json.loads(submission_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["dry_run"]["feature_fallback_wells"] == 1


def test_state_space_fold_manifest_is_deterministic_and_row_balanced():
    counts = pd.DataFrame(
        {
            "well": ["a", "b", "c", "d", "e", "f"],
            "scored_rows": [100, 80, 60, 40, 20, 10],
        }
    )

    first = state_space.assign_balanced_folds(counts, n_splits=3)
    second = state_space.assign_balanced_folds(counts.sample(frac=1, random_state=7), n_splits=3)

    pd.testing.assert_frame_equal(first, second)
    fold_weights = first.groupby("fold")["scored_rows"].sum()
    assert fold_weights.max() - fold_weights.min() <= counts["scored_rows"].max()
    assert set(first["fold"]) == {1, 2, 3}


def test_state_space_candidates_do_not_depend_on_suffix_target(tmp_path):
    horizontal_path = write_well(tmp_path, "state001")
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(tmp_path / "state001__typewell.csv")

    before, _ = state_space.particle_candidates(
        horizontal,
        typewell,
        particles=12,
        seeds=2,
        scales=(5.0,),
        holds=(0.0, 0.5),
        reference_step=1.0,
    )
    horizontal.loc[horizontal["TVT_input"].isna(), "TVT"] += 50_000.0
    after, _ = state_space.particle_candidates(
        horizontal,
        typewell,
        particles=12,
        seeds=2,
        scales=(5.0,),
        holds=(0.0, 0.5),
        reference_step=1.0,
    )

    pd.testing.assert_frame_equal(before, after)
    assert np.isfinite(before.filter(like="pf_scale").to_numpy()).all()


def test_state_space_skips_missing_suffix_gr(tmp_path):
    horizontal_path = write_well(tmp_path, "missinggr")
    horizontal = pd.read_csv(horizontal_path)
    missing = horizontal["TVT_input"].isna()
    horizontal.loc[missing, "GR"] = np.nan

    candidates, diagnostics = state_space.particle_candidates(
        horizontal,
        pd.read_csv(tmp_path / "missinggr__typewell.csv"),
        particles=12,
        seeds=2,
        scales=(5.0,),
        holds=(0.0,),
        reference_step=1.0,
    )

    assert diagnostics["observed_gr_rows"] == 0
    assert diagnostics["missing_gr_rows"] == int(missing.sum())
    assert diagnostics["log_likelihood_min"] == 0.0
    assert diagnostics["log_likelihood_max"] == 0.0
    assert np.isfinite(candidates["pf_scale_5_hold_0"].to_numpy()).all()


@pytest.mark.parametrize(
    ("cv_decision", "test_failures", "expected"),
    [
        ("candidate_ready", [], "candidate_ready"),
        ("candidate_ready", [{"well": "bad"}], "candidate_blocked_test_failures"),
        ("promising_continue", [], "promising_continue"),
        ("exhausted", [], "exhausted"),
    ],
)
def test_state_space_submission_gate(cv_decision, test_failures, expected):
    assert state_space.finalize_candidate_decision(cv_decision, test_failures) == expected


def test_state_space_writer_preserves_sample_order(tmp_path):
    raw_root = tmp_path / "data" / "raw"
    raw_root.mkdir(parents=True)
    sample = pd.DataFrame({"id": ["well_2", "well_1"], "tvt": [0.0, 0.0]})
    sample.to_csv(raw_root / "sample_submission.csv", index=False)
    predictions = pd.DataFrame({"_id": ["well_1", "well_2"], "prediction": [10.0, 20.0]})

    path, values = state_space.write_submission(tmp_path, "v999", predictions, {"submitted": False})

    assert pd.read_csv(path)["id"].tolist() == sample["id"].tolist()
    assert values.tolist() == [20.0, 10.0]
    assert json.loads(path.with_suffix(".json").read_text())["submitted"] is False


def test_state_space_training_records_interrupt(tmp_path, monkeypatch):
    train_root = tmp_path / "data" / "raw" / "train"
    for index in range(2):
        write_well(train_root, f"interrupt{index}", offset=index * 10.0)

    def interrupt(*args, **kwargs):
        del args, kwargs
        raise KeyboardInterrupt("test stop")

    monkeypatch.setattr(state_space, "NUMBA_AVAILABLE", True)
    monkeypatch.setattr(state_space, "build_training_candidates", interrupt)

    with pytest.raises(KeyboardInterrupt, match="test stop"):
        state_space.train(
            tmp_path,
            version="v001",
            folds=2,
            particles=2,
            seeds=1,
            scales=(5.0,),
            holds=(0.0,),
            candidate_gate=12.5,
            exhausted_gate=14.0,
        )

    run = json.loads((tmp_path / "models" / "v001" / "run.json").read_text())
    assert run["status"] == "failed"
    assert run["error"] == "KeyboardInterrupt: test stop"


def patch_fast_state_space_training(monkeypatch, prediction: float):
    def build_candidates(workspace, manifest, feature_path, **kwargs):
        del workspace, kwargs
        candidate = "pf_scale_5_hold_0"
        rows = []
        summaries = []
        for index, item in enumerate(manifest.itertuples(index=False)):
            rows.append(
                {
                    "_id": f"{item.well}_3",
                    "_well_id": item.well,
                    "_row_index": 3,
                    "_target": 0.0,
                    "fold": item.fold,
                    candidate: prediction,
                }
            )
            summaries.append(
                {
                    "well": item.well,
                    "fold": item.fold,
                    "candidate": candidate,
                    "squared_error": prediction**2,
                    "rows": 1,
                    "rmse": prediction,
                }
            )
        pd.DataFrame(rows).to_parquet(feature_path, index=False)
        return pd.DataFrame(summaries), [{"well": row["_well_id"]} for row in rows], []

    monkeypatch.setattr(state_space, "NUMBA_AVAILABLE", True)
    monkeypatch.setattr(state_space, "build_training_candidates", build_candidates)
    monkeypatch.setattr(
        state_space,
        "baseline_scores_on_manifest",
        lambda workspace, manifest, version: {
            "overall": 15.0,
            "fold_scores": [15.0] * manifest["fold"].nunique(),
        },
    )


def run_fast_state_space_training(tmp_path, *, prediction: float, monkeypatch):
    train_root = tmp_path / "data" / "raw" / "train"
    for index in range(5):
        write_well(train_root, f"gate{index}", offset=index * 10.0)
    patch_fast_state_space_training(monkeypatch, prediction)
    return state_space.train(
        tmp_path,
        version="v001",
        folds=5,
        particles=2,
        seeds=1,
        scales=(5.0,),
        holds=(0.0,),
        candidate_gate=12.5,
        exhausted_gate=14.0,
    )


def test_state_space_promising_result_does_not_predict_or_write_submission(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        del args, kwargs
        raise AssertionError("submission path must remain gated")

    monkeypatch.setattr(state_space, "predict_visible_test", unexpected)
    monkeypatch.setattr(state_space, "write_submission", unexpected)

    model_dir = run_fast_state_space_training(tmp_path, prediction=13.0, monkeypatch=monkeypatch)

    run = json.loads((model_dir / "run.json").read_text())
    assert run["decision"] == "promising_continue"
    assert run["submission_path"] is None
    assert not (tmp_path / "submissions").exists()


def test_state_space_test_fallback_blocks_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(
        state_space,
        "predict_visible_test",
        lambda *args, **kwargs: (
            pd.DataFrame({"_id": ["bad_3"], "prediction": [0.0]}),
            [{"well": "bad", "error": "test failure"}],
        ),
    )

    def unexpected(*args, **kwargs):
        del args, kwargs
        raise AssertionError("fallback predictions must not become a submission")

    monkeypatch.setattr(state_space, "write_submission", unexpected)

    model_dir = run_fast_state_space_training(tmp_path, prediction=10.0, monkeypatch=monkeypatch)

    run = json.loads((model_dir / "run.json").read_text())
    scores = json.loads((model_dir / "cv_scores.json").read_text())
    assert run["decision"] == "candidate_blocked_test_failures"
    assert scores["decision"] == "candidate_blocked_test_failures"
    assert run["test_fallback_wells"] == 1
    assert run["submission_path"] is None
    assert not (tmp_path / "submissions").exists()


def test_state_space_training_records_sigterm(tmp_path, monkeypatch):
    train_root = tmp_path / "data" / "raw" / "train"
    for index in range(2):
        write_well(train_root, f"sigterm{index}", offset=index * 10.0)

    def terminate(*args, **kwargs):
        del args, kwargs
        signal.raise_signal(signal.SIGTERM)

    previous_handler = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(state_space, "NUMBA_AVAILABLE", True)
    monkeypatch.setattr(state_space, "build_training_candidates", terminate)

    with pytest.raises(KeyboardInterrupt, match="received SIGTERM"):
        state_space.train(
            tmp_path,
            version="v001",
            folds=2,
            particles=2,
            seeds=1,
            scales=(5.0,),
            holds=(0.0,),
            candidate_gate=12.5,
            exhausted_gate=14.0,
        )

    run = json.loads((tmp_path / "models" / "v001" / "run.json").read_text())
    assert run["status"] == "failed"
    assert run["error"] == "KeyboardInterrupt: received SIGTERM"
    assert signal.getsignal(signal.SIGTERM) is previous_handler
