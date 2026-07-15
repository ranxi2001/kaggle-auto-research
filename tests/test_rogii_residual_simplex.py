import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "workspaces"
    / "rogii-wellbore-geology-prediction"
    / "scripts"
    / "rogii_residual_simplex.py"
)
SPEC = importlib.util.spec_from_file_location("rogii_residual_simplex", SCRIPT_PATH)
assert SPEC and SPEC.loader
simplex = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = simplex
SPEC.loader.exec_module(simplex)


def synthetic_streams(seed=17, rows=600):
    rng = np.random.default_rng(seed)
    parent = rng.normal(10_000.0, 15.0, rows)
    deltas = rng.normal(size=(rows, 3)) * np.array([20.0, 35.0, 45.0])
    return np.column_stack([parent, parent[:, None] + deltas])


@pytest.mark.parametrize(
    "weights",
    [
        np.array([0.40, 0.20, 0.25, 0.15]),
        np.array([0.00, 0.20, 0.30, 0.50]),
        np.array([0.65, 0.35, 0.00, 0.00]),
        np.array([1.00, 0.00, 0.00, 0.00]),
    ],
)
def test_four_stream_simplex_recovers_exact_feasible_weights(weights):
    streams = synthetic_streams()
    target = streams @ weights

    fit = simplex.fit_simplex_weights(streams, target)

    assert fit.attempted_systems == 15
    assert len(fit.support_audit) == 15
    assert len({item["support"] for item in fit.support_audit}) == 15
    assert np.all(fit.weights >= 0.0)
    assert fit.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert fit.weights == pytest.approx(weights, abs=1e-10)
    assert fit.squared_error == pytest.approx(0.0, abs=1e-16)
    assert fit.direct_sse_verified


def test_degenerate_identical_streams_use_conservative_parent_tie_break():
    parent = np.linspace(10.0, 20.0, 200)
    streams = np.column_stack([parent, parent, parent, parent])

    fit = simplex.fit_simplex_weights(streams, parent.copy())

    assert np.array_equal(fit.weights, np.array([1.0, 0.0, 0.0, 0.0]))
    assert fit.selected_face == "all_parent"
    assert fit.attempted_systems == 15


def test_rank_deficient_duplicate_streams_remain_feasible_and_exact():
    rng = np.random.default_rng(9)
    parent = rng.normal(size=300)
    correction = rng.normal(size=300)
    streams = np.column_stack(
        [parent, parent + correction, parent + correction, parent - correction]
    )
    target = parent + 0.25 * correction

    fit = simplex.fit_simplex_weights(streams, target)

    assert np.all(fit.weights >= 0.0)
    assert fit.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.sqrt(fit.squared_error / len(target)) < 1e-12
    assert any(item["solve_status"] == "lstsq" for item in fit.support_audit)


def test_duplicate_stream_tie_uses_tolerance_aware_hierarchical_preference():
    rng = np.random.default_rng(42)
    parent = rng.normal(size=400)
    correction = rng.normal(size=400)
    hmm = parent + correction
    ancc = hmm.copy()
    best6 = parent + rng.normal(size=400)
    streams = np.column_stack([parent, hmm, ancc, best6])
    target = 0.2 * parent + 0.8 * ancc

    fit = simplex.fit_simplex_weights(streams, target)

    assert fit.weights == pytest.approx([0.2, 0.0, 0.8, 0.0], abs=1e-10)
    assert np.sqrt(fit.squared_error / len(target)) < 1e-12


def test_simplex_rejects_nonfinite_and_invalid_gram():
    streams = synthetic_streams(rows=20)
    target = streams[:, 0].copy()
    streams[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        simplex.fit_simplex_weights(streams, target)

    gram = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="positive semidefinite"):
        simplex.enumerate_simplex_supports(gram, np.zeros(3), 1.0)


def feature_frames():
    ids = ["well_b_0", "well_a_0", "well_a_1", "well_b_1", "well_c_0"]
    wells = ["well_b", "well_a", "well_a", "well_b", "well_c"]
    row_index = [0, 0, 1, 1, 0]
    folds = [2, 1, 1, 2, 3]
    base = pd.DataFrame(
        {
            "_id": ids,
            "_well_id": wells,
            "_row_index": row_index,
            "_target": np.arange(5, dtype=float) + 100.0,
            "fold": folds,
            "parent_v006": np.arange(5, dtype=float) + 99.0,
        }
    )
    hmm = base.assign(hmm_wr21_slow=np.arange(5, dtype=float) + 98.0)
    formation = base.assign(
        sp_plane_ancc_k10=np.arange(5, dtype=float) + 97.0,
        sp_plane_best6_k10=np.arange(5, dtype=float) + 96.0,
    )
    for outer in range(1, 6):
        formation[simplex.nested_stream_column(outer, "sp_plane_ancc_k10")] = (
            np.arange(5, dtype=float) + 90.0 + outer
        )
        formation[simplex.nested_stream_column(outer, "sp_plane_best6_k10")] = (
            np.arange(5, dtype=float) + 80.0 + outer
        )
    return hmm, formation


def test_feature_join_uses_global_id_not_repeated_row_index_and_restores_v005_order(tmp_path):
    hmm, formation = feature_frames()
    hmm_path = tmp_path / "v004.parquet"
    formation_path = tmp_path / "v005.parquet"
    hmm.sample(frac=1.0, random_state=3).to_parquet(hmm_path, index=False)
    formation.to_parquet(formation_path, index=False)

    joined, audit = simplex.join_locked_features(hmm_path, formation_path)

    assert joined["_id"].tolist() == formation["_id"].tolist()
    assert joined["_well_id"].tolist() == formation["_well_id"].tolist()
    assert joined["_row_index"].tolist().count(0) == 3
    assert np.array_equal(
        joined["hmm_wr21_slow"], hmm.set_index("_id").loc[joined["_id"], "hmm_wr21_slow"]
    )
    assert audit["join_key"] == "_id"
    assert audit["global_unique_ids"]


@pytest.mark.parametrize("failure", ["duplicate_id", "target_ulp", "missing_id"])
def test_feature_join_fails_closed_on_key_or_exact_field_mismatch(tmp_path, failure):
    hmm, formation = feature_frames()
    if failure == "duplicate_id":
        hmm.loc[1, "_id"] = hmm.loc[0, "_id"]
    elif failure == "target_ulp":
        hmm.loc[1, "_target"] = np.nextafter(hmm.loc[1, "_target"], np.inf)
    else:
        hmm = hmm.iloc[:-1].copy()
    hmm_path = tmp_path / "v004.parquet"
    formation_path = tmp_path / "v005.parquet"
    hmm.to_parquet(hmm_path, index=False)
    formation.to_parquet(formation_path, index=False)

    with pytest.raises(ValueError):
        simplex.join_locked_features(hmm_path, formation_path)


def nested_frame(seed=22):
    rng = np.random.default_rng(seed)
    rows_per_fold = 40
    folds = np.repeat(np.arange(1, 6), rows_per_fold)
    parent = rng.normal(1000.0, 3.0, len(folds))
    hmm = parent + rng.normal(0.0, 4.0, len(folds))
    ancc = parent + rng.normal(0.0, 6.0, len(folds))
    best6 = parent + rng.normal(0.0, 7.0, len(folds))
    frame = pd.DataFrame(
        {
            "_target": 0.65 * parent + 0.25 * hmm + 0.10 * ancc,
            "fold": folds,
            "parent_v006": parent,
            "hmm_wr21_slow": hmm,
            "sp_plane_ancc_k10": ancc,
            "sp_plane_best6_k10": best6,
        }
    )
    for outer in range(1, 6):
        frame[simplex.nested_stream_column(outer, "sp_plane_ancc_k10")] = ancc + rng.normal(
            0.0, 0.2, len(folds)
        )
        frame[simplex.nested_stream_column(outer, "sp_plane_best6_k10")] = best6 + rng.normal(
            0.0, 0.2, len(folds)
        )
    return frame


def test_nested_stack_uses_only_outer_specific_meta_columns_and_canonical_evaluation(monkeypatch):
    frame = nested_frame()
    original = simplex.fit_simplex_weights
    calls = []

    def spy(streams, target):
        calls.append((np.asarray(streams).copy(), np.asarray(target).copy()))
        return original(streams, target)

    monkeypatch.setattr(simplex, "fit_simplex_weights", spy)
    selections, _, oof = simplex.nested_four_stream_stack(frame)

    outer = 1
    train_mask = frame["fold"].to_numpy() != outer
    expected = frame.loc[train_mask, simplex._outer_meta_columns(outer)].to_numpy(dtype=float)
    assert np.array_equal(calls[0][0], expected)
    assert np.array_equal(calls[0][1], frame.loc[train_mask, "_target"].to_numpy(dtype=float))
    validation_mask = ~train_mask
    weights = np.array([selections[0]["weights"][name] for name in simplex.STREAM_COLUMNS])
    expected_oof = (
        frame.loc[validation_mask, simplex.STREAM_COLUMNS].to_numpy(dtype=float) @ weights
    )
    assert np.allclose(oof[validation_mask], expected_oof, rtol=0.0, atol=1e-12)
    assert len(calls) == 6


def test_outer_one_weights_ignore_fold_one_target_and_other_outer_nested_columns():
    frame = nested_frame()
    first, _, _ = simplex.nested_four_stream_stack(frame)

    changed = frame.copy()
    changed.loc[changed["fold"] == 1, "_target"] += 10_000.0
    changed[simplex.nested_stream_column(2, "sp_plane_ancc_k10")] -= 20_000.0
    changed[simplex.nested_stream_column(5, "sp_plane_best6_k10")] += 30_000.0
    second, _, _ = simplex.nested_four_stream_stack(changed)

    assert first[0]["weights"] == pytest.approx(second[0]["weights"], abs=1e-12)


@pytest.mark.parametrize(
    ("score", "folds", "expected"),
    [
        (10.95, 4, "candidate_ready"),
        (11.70, 3, "promising_continue"),
        (11.75, 5, "exhausted"),
        (12.00, 5, "exhausted"),
    ],
)
def test_candidate_gate_is_fixed(score, folds, expected):
    assert simplex.candidate_decision(score, folds) == expected


def test_test_inference_is_never_called_below_candidate_gate(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("test inference must not run")

    monkeypatch.setattr(simplex, "predict_visible_test", fail)
    result = simplex.maybe_predict_visible_test(
        Path("unused"), "promising_continue", np.ones(4) / 4
    )
    assert result == ("promising_continue", None, [], [], "not_eligible")
