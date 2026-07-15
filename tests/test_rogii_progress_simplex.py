from __future__ import annotations

import importlib.util
import json
import signal
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "workspaces"
    / "rogii-wellbore-geology-prediction"
    / "scripts"
    / "rogii_progress_simplex.py"
)
SPEC = importlib.util.spec_from_file_location("rogii_progress_simplex_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
progress = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = progress
SPEC.loader.exec_module(progress)


def progress_frame():
    return pd.DataFrame(
        {
            "_well_id": ["a"] * 6 + ["b"] * 3,
            "_row_index": [10, 11, 12, 13, 14, 15, 100, 102, 104],
            "parent_v006": [100.0] * 9,
            "hmm_wr21_slow": [110.0] * 9,
            "_target": np.arange(9, dtype=float),
        }
    )


def nested_frame(seed=12, rows_per_fold=24):
    rng = np.random.default_rng(seed)
    folds = np.repeat(np.arange(1, 6), rows_per_fold)
    rows = len(folds)
    parent = 10_000 + rng.normal(0, 20, rows)
    slow = parent + rng.normal(0, 8, rows)
    ancc = parent + rng.normal(0, 10, rows)
    best = parent + rng.normal(0, 12, rows)
    frame = pd.DataFrame(
        {
            "_id": [f"w{fold}_{row}" for row, fold in enumerate(folds)],
            "_well_id": [f"w{fold}" for fold in folds],
            "_row_index": np.tile(np.arange(rows_per_fold), 5),
            "fold": folds,
            "parent_v006": parent,
            "hmm_wr21_slow": slow,
            "sp_plane_ancc_k10": ancc,
            "sp_plane_best6_k10": best,
        }
    )
    shaped, _ = progress.add_progress_stream(frame)
    frame["_target"] = (
        0.55 * parent
        + 0.15 * slow
        + 0.12 * shaped["hmm_front20"].to_numpy()
        + 0.10 * ancc
        + 0.08 * best
    )
    for outer in range(1, 6):
        frame[progress.nested_stream_column(outer, "sp_plane_ancc_k10")] = ancc + rng.normal(
            0, 0.3, rows
        )
        frame[progress.nested_stream_column(outer, "sp_plane_best6_k10")] = best + rng.normal(
            0, 0.3, rows
        )
    return frame


def test_progress_stream_has_locked_endpoints_and_is_per_well():
    shaped, diagnostics = progress.add_progress_stream(progress_frame())
    a = shaped[shaped["_well_id"] == "a"]
    b = shaped[shaped["_well_id"] == "b"]
    np.testing.assert_allclose(a["_suffix_progress"], np.linspace(0, 1, 6))
    np.testing.assert_allclose(b["_suffix_progress"], [0.0, 0.5, 1.0])
    assert a["hmm_front20"].iloc[0] == 110.0
    assert a["hmm_front20"].iloc[1] == 100.0
    assert (a["hmm_front20"].iloc[1:] == 100.0).all()
    assert diagnostics["target_columns_read"] == []


def test_progress_stream_does_not_read_target_values():
    frame = progress_frame()
    first, _ = progress.add_progress_stream(frame)
    frame["_target"] = 1e12
    second, _ = progress.add_progress_stream(frame)
    np.testing.assert_array_equal(first["hmm_front20"], second["hmm_front20"])


def test_progress_fails_on_single_row_or_duplicate_row_index():
    with pytest.raises(ValueError, match="multiple suffix rows"):
        progress.add_progress_stream(progress_frame().iloc[[0]])
    duplicate = progress_frame()
    duplicate.loc[1, "_row_index"] = duplicate.loc[0, "_row_index"]
    with pytest.raises(ValueError, match="unique within"):
        progress.add_progress_stream(duplicate)


def test_five_stream_solver_recovers_known_weights_and_31_supports():
    rng = np.random.default_rng(3)
    parent = 10_000 + rng.normal(0, 20, 1500)
    streams = np.column_stack(
        [parent] + [parent + rng.normal(0, s, len(parent)) for s in (6, 8, 10, 12)]
    )
    expected = np.asarray([0.5, 0.1, 0.15, 0.12, 0.13])
    fit = progress.fit_simplex_weights(streams, streams @ expected)
    np.testing.assert_allclose(fit.weights, expected, atol=1e-10, rtol=0.0)
    assert fit.attempted_systems == 31
    assert len(fit.support_audit) == 31
    assert len({item["support"] for item in fit.support_audit}) == 31


def test_duplicate_stream_tie_uses_tolerance_aware_hierarchy():
    x = np.linspace(-2, 2, 600)
    parent = 100 + x
    correction = np.sin(x)
    slow = parent + correction
    front = parent + correction
    ancc = parent + correction
    best = parent - correction
    streams = np.column_stack([parent, slow, front, ancc, best])
    target = 0.2 * parent + 0.8 * ancc
    fit = progress.fit_simplex_weights(streams, target)
    np.testing.assert_allclose(fit.weights, [0.2, 0.0, 0.0, 0.8, 0.0], atol=1e-10)


def test_solver_degenerate_parent_and_parent_zero_vertex():
    parent = np.linspace(0, 1, 300)
    same = np.column_stack([parent] * 5)
    degenerate = progress.fit_simplex_weights(same, parent + 1)
    np.testing.assert_array_equal(degenerate.weights, [1, 0, 0, 0, 0])
    streams = np.column_stack([parent, parent + 1, parent + 2, parent + 3, parent + 4])
    vertex = progress.fit_simplex_weights(streams, streams[:, 4])
    np.testing.assert_allclose(vertex.weights, [0, 0, 0, 0, 1], atol=1e-10)


def test_nested_stack_uses_nested_meta_and_canonical_eval():
    frame = nested_frame()
    shaped, selections, final_fit, before, diagnostics = progress.nested_five_stream_stack(frame)
    assert diagnostics["violations"] == 0 and final_fit.attempted_systems == 31
    fold1 = next(item for item in selections if item["fold"] == 1)
    changed = frame.copy()
    changed.loc[changed["fold"] != 1, "sp_plane_ancc_k10"] += 1000
    _, after_selections, _, _, _ = progress.nested_five_stream_stack(changed)
    assert (
        next(item for item in after_selections if item["fold"] == 1)["weights"] == fold1["weights"]
    )
    eval_changed = frame.copy()
    eval_changed.loc[eval_changed["fold"] == 1, "sp_plane_ancc_k10"] += 100
    _, eval_selections, _, after, _ = progress.nested_five_stream_stack(eval_changed)
    assert (
        next(item for item in eval_selections if item["fold"] == 1)["weights"] == fold1["weights"]
    )
    assert not np.array_equal(after[shaped["fold"] == 1], before[shaped["fold"] == 1])


@pytest.mark.parametrize(
    ("score", "v009", "expected"),
    [
        (11.49, 3, "diagnostic_improvement_not_candidate"),
        (10.00, 5, "diagnostic_improvement_not_candidate"),
        (11.50, 5, "exhausted"),
        (11.0, 2, "exhausted"),
    ],
)
def test_diagnostic_gates_never_promote_candidate(score, v009, expected):
    decision = progress.candidate_decision(score, v009)
    assert decision == expected
    assert decision != "candidate_ready"


def test_fit_integrity_requires_31_unique_supports_and_direct_sse():
    frame = nested_frame()
    _, selections, final_fit, _, _ = progress.nested_five_stream_stack(frame)
    assert progress.fit_integrity_violations(selections, final_fit) == (0, 0)

    selections[0]["support_audit"][1]["support"] = selections[0]["support_audit"][0]["support"]
    selections[1]["direct_sse_verified"] = False
    assert progress.fit_integrity_violations(selections, final_fit) == (1, 1)


def test_adaptive_diagnostic_has_no_test_inference_path():
    source = SCRIPT.read_text().lower()
    assert "def predict_visible_test" not in source
    assert "test_preds.npy" not in source
    assert "candidate_ready" not in source


def test_v009_audit_hash_tamper_fails_closed(tmp_path):
    source = tmp_path / "scripts" / "rogii_residual_simplex.py"
    run = tmp_path / "models" / "v009" / "run.json"
    source.parent.mkdir(parents=True)
    run.parent.mkdir(parents=True)
    source.write_text("source")
    run.write_text("run")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "version": "v009",
                "status": "verified_no_blocker",
                "sha256": {
                    "scripts/rogii_residual_simplex.py": progress.sha256_file(source),
                    "models/v009/run.json": progress.sha256_file(run),
                },
            }
        )
    )
    progress._validate_v009_audit(tmp_path, audit)
    run.write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        progress._validate_v009_audit(tmp_path, audit)


@pytest.mark.parametrize("failure", [RuntimeError("join failed"), KeyboardInterrupt("stop")])
def test_train_failure_publishes_failed_run(tmp_path, monkeypatch, failure):
    workspace = tmp_path / type(failure).__name__
    reports = workspace / "reports"
    reports.mkdir(parents=True)
    fold_path = reports / "folds.csv"
    fold_path.write_text("well,fold\n")
    manifest = pd.DataFrame({"well": [f"w{i}" for i in range(1, 6)], "fold": range(1, 6)})
    hashes = {"source_v010": "s", "plan_v010": "p", "upstream_fold_manifest": "f"}
    monkeypatch.setattr(progress.V008, "ensure_fold_manifest", lambda *args: (fold_path, manifest))
    monkeypatch.setattr(progress, "critical_input_hashes", lambda *args: hashes.copy())
    monkeypatch.setattr(progress.V008, "data_fingerprint", lambda root: "raw")
    monkeypatch.setattr(
        progress.V009, "join_locked_features", lambda *args: (_ for _ in ()).throw(failure)
    )
    old = signal.getsignal(signal.SIGTERM)
    with pytest.raises(type(failure)):
        progress.train(workspace)
    run = json.loads((workspace / "models" / "v010" / "run.json").read_text())
    assert run["status"] == "failed" and run["test_predictions_saved"] is False
    assert signal.getsignal(signal.SIGTERM) == old


def test_cli_has_no_submit_command():
    args = progress.parse_args(["--workspace", "/tmp/x", "train"])
    assert args.command == "train"
    source = SCRIPT.read_text().lower()
    assert "kaggle competitions submit" not in source
    assert "def write_submission" not in source
