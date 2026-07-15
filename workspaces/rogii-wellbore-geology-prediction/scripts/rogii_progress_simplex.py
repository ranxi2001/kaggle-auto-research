#!/usr/bin/env python3
"""ROGII v010 progress-shaped HMM stream and exact five-stream simplex."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import signal
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


VERSION = "v010"
STREAM_COLUMNS = (
    "parent_v006",
    "hmm_wr21_slow",
    "hmm_front20",
    "sp_plane_ancc_k10",
    "sp_plane_best6_k10",
)
PROMISING_GATE = 11.50
EXPECTED_SUPPORTS = 31
TAPER_FACTOR = 5.0


def _load_v009_module():
    name = "rogii_residual_simplex_v009_dependency"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("rogii_residual_simplex.py").resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load v009 helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V009 = _load_v009_module()
V008 = V009.V008


@dataclass(frozen=True)
class SimplexCandidate:
    face: str
    weights: np.ndarray
    squared_error: float


@dataclass(frozen=True)
class SimplexFit:
    weights: np.ndarray
    squared_error: float
    rows: int
    delta_gram: np.ndarray
    delta_cross: np.ndarray
    residual_squared: float
    selected_face: str
    feasible_candidates: int
    attempted_systems: int
    direct_sse_verified: bool
    support_audit: tuple[dict, ...]


class TerminationRequested(RuntimeError):
    pass


def _raise_on_sigterm(signum, frame) -> None:
    del frame
    raise TerminationRequested(f"received signal {signum}")


def sha256_file(path: Path) -> str:
    return V009.sha256_file(path)


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return V009.rmse(target, prediction)


def nested_stream_column(outer_fold: int, stream: str) -> str:
    return V009.nested_stream_column(outer_fold, stream)


def _fixed_settings() -> dict:
    return {
        "streams": list(STREAM_COLUMNS),
        "progress": "(row_index-min_suffix_row)/(max_suffix_row-min_suffix_row) per well",
        "taper": "max(0,1-5*q)",
        "taper_factor": TAPER_FACTOR,
        "supports": EXPECTED_SUPPORTS,
        "lstsq_rcond": 1e-12,
        "linear_residual_tolerance": V009.LINEAR_RESIDUAL_TOLERANCE,
        "raw_feasibility_tolerance": V009.FEASIBILITY_TOLERANCE,
        "weight_sum_tolerance": V009.WEIGHT_SUM_TOLERANCE,
        "direct_sse_tolerance": V009.DIRECT_SSE_TOLERANCE,
        "tie_tolerance": "64*eps(float64)*max(1,SSE_a,SSE_b)",
        "tie_break": "max parent, min slow, min front20, min ANCC, min best6",
        "row_sampling": False,
        "regularization": False,
        "weight_clipping": False,
        "weight_renormalization": False,
        "outer_meta_formation": "nested_outer_k columns",
        "outer_evaluation_formation": "canonical formation columns",
    }


def add_progress_stream(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"_well_id", "_row_index", "parent_v006", "hmm_wr21_slow"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Progress stream input is missing columns: {missing}")
    result = frame.copy()
    row_index = pd.to_numeric(result["_row_index"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(row_index).all() or not np.array_equal(row_index, np.rint(row_index)):
        raise ValueError("Suffix row indices must be finite exact integers")
    limits = np.iinfo(np.int64)
    if np.any(row_index < limits.min) or np.any(row_index > limits.max):
        raise ValueError("Suffix row indices are outside exact int64 range")
    if result.duplicated(["_well_id", "_row_index"]).any():
        raise ValueError("Suffix row indices must be unique within each well")
    result["_progress_row"] = row_index
    grouped = result.groupby("_well_id", sort=False)["_progress_row"]
    minimum = grouped.transform("min").to_numpy(dtype=np.float64)
    maximum = grouped.transform("max").to_numpy(dtype=np.float64)
    denominator = maximum - minimum
    if np.any(denominator <= 0.0):
        raise ValueError("Every well must have multiple suffix rows with positive progress span")
    progress = (row_index - minimum) / denominator
    if not np.isfinite(progress).all() or np.any(progress < 0.0) or np.any(progress > 1.0):
        raise ValueError("Suffix progress must be finite and remain in [0,1]")
    progress_frame = pd.DataFrame({"well": result["_well_id"].to_numpy(), "q": progress})
    endpoints = progress_frame.groupby("well", sort=False)["q"].agg(["min", "max"])
    if (
        not (endpoints["min"].to_numpy() == 0.0).all()
        or not (endpoints["max"].to_numpy() == 1.0).all()
    ):
        raise ValueError("Every well must attain exact progress endpoints 0 and 1")
    taper = np.maximum(0.0, 1.0 - TAPER_FACTOR * progress)
    parent = result["parent_v006"].to_numpy(dtype=np.float64)
    slow = result["hmm_wr21_slow"].to_numpy(dtype=np.float64)
    front = parent + taper * (slow - parent)
    if not np.isfinite(progress).all() or not np.isfinite(front).all():
        raise ValueError("Progress-shaped HMM stream contains non-finite values")
    result["_suffix_progress"] = progress
    result["hmm_front20"] = front
    verified = parent + np.maximum(0.0, 1.0 - 5.0 * progress) * (slow - parent)
    if not np.array_equal(front, verified):
        raise ValueError("Progress-shaped HMM formula verification failed")
    result = result.drop(columns=["_progress_row"])
    return result, {
        "rows": len(result),
        "wells": int(result["_well_id"].nunique()),
        "progress_min": float(np.min(progress)),
        "progress_max": float(np.max(progress)),
        "front_active_rows": int(np.count_nonzero(taper > 0.0)),
        "all_wells_endpoint_zero": True,
        "all_wells_endpoint_one": True,
        "formula_exact": True,
        "target_columns_read": [],
        "violations": 0,
    }


def _quadratic_sse(gram, cross, residual_squared, delta_weights) -> float:
    return float(
        residual_squared - 2.0 * np.dot(cross, delta_weights) + delta_weights @ gram @ delta_weights
    )


def enumerate_simplex_supports(gram, cross, residual_squared) -> list[dict]:
    gram = np.asarray(gram, dtype=np.float64)
    cross = np.asarray(cross, dtype=np.float64)
    if gram.shape != (4, 4) or cross.shape != (4,):
        raise ValueError("Five-stream simplex requires a 4x4 Gram and four-vector")
    if not np.isfinite(gram).all() or not np.isfinite(cross).all():
        raise ValueError("Simplex sufficient statistics must be finite")
    if not np.allclose(gram, gram.T, rtol=0.0, atol=1e-10):
        raise ValueError("Simplex Gram matrix must be symmetric")
    eigen_tolerance = V009.LINEAR_RESIDUAL_TOLERANCE * max(1.0, float(np.max(np.abs(gram))))
    if float(np.min(np.linalg.eigvalsh(gram))) < -eigen_tolerance:
        raise ValueError("Simplex Gram matrix is not positive semidefinite")
    if not np.isfinite(residual_squared) or residual_squared < 0.0:
        raise ValueError("Residual squared norm must be finite and non-negative")
    audits = [
        {
            "support": "all_parent",
            "face_class": "vertex",
            "delta_subset": [],
            "solve_status": "solved",
            "status": "feasible",
            "linear_residual_inf": 0.0,
            "raw_weights": [1.0, 0.0, 0.0, 0.0, 0.0],
            "weights": [1.0, 0.0, 0.0, 0.0, 0.0],
            "canonicalized_to": None,
        }
    ]
    for size in range(1, 5):
        for subset_tuple in combinations(range(4), size):
            subset = np.asarray(subset_tuple, dtype=np.int64)
            sub_gram = gram[np.ix_(subset, subset)]
            sub_cross = cross[subset]
            solution, solve_status, solve_residual = V009._solve_checked(sub_gram, sub_cross)
            active = {
                "support": "parent_active:" + ",".join(map(str, subset_tuple)),
                "face_class": "parent_active",
                "delta_subset": list(subset_tuple),
                "solve_status": solve_status,
                "status": "singular-invalid" if solution is None else "infeasible",
                "linear_residual_inf": solve_residual,
                "raw_weights": None,
                "weights": None,
                "canonicalized_to": None,
            }
            if solution is not None:
                delta = np.zeros(4, dtype=np.float64)
                delta[subset] = solution
                weights = np.concatenate([[1.0 - float(delta.sum())], delta])
                active["raw_weights"] = weights.tolist()
                tolerant = np.all(weights >= -V009.FEASIBILITY_TOLERANCE)
                if tolerant and np.all(weights >= 0.0):
                    active["status"] = "feasible"
                    active["weights"] = weights.tolist()
                elif tolerant:
                    active["status"] = "needs-canonicalization"
            audits.append(active)

            kkt = np.block(
                [
                    [sub_gram, np.ones((size, 1))],
                    [np.ones((1, size)), np.zeros((1, 1))],
                ]
            )
            kkt_rhs = np.concatenate([sub_cross, [1.0]])
            solution, solve_status, solve_residual = V009._solve_checked(kkt, kkt_rhs)
            zero = {
                "support": "parent_zero:" + ",".join(map(str, subset_tuple)),
                "face_class": "parent_zero",
                "delta_subset": list(subset_tuple),
                "solve_status": solve_status,
                "status": "singular-invalid" if solution is None else "infeasible",
                "linear_residual_inf": solve_residual,
                "raw_weights": None,
                "weights": None,
                "canonicalized_to": None,
            }
            if solution is not None:
                delta = np.zeros(4, dtype=np.float64)
                delta[subset] = solution[:-1]
                weights = np.concatenate([[0.0], delta])
                zero["raw_weights"] = weights.tolist()
                tolerant = (
                    np.all(weights >= -V009.FEASIBILITY_TOLERANCE)
                    and abs(float(delta.sum()) - 1.0) <= V009.FEASIBILITY_TOLERANCE
                )
                if tolerant and np.all(weights >= 0.0):
                    zero["status"] = "feasible"
                    zero["weights"] = weights.tolist()
                elif tolerant:
                    zero["status"] = "needs-canonicalization"
            audits.append(zero)
    if len(audits) != EXPECTED_SUPPORTS:
        raise RuntimeError(f"Audited {len(audits)} supports, expected 31")
    exact = [item for item in audits if item["status"] == "feasible"]
    for item in audits:
        if item["status"] != "needs-canonicalization":
            continue
        raw = np.asarray(item["raw_weights"], dtype=np.float64)
        matches = [
            candidate
            for candidate in exact
            if np.allclose(
                raw,
                candidate["weights"],
                rtol=0.0,
                atol=V009.FEASIBILITY_TOLERANCE,
            )
        ]
        if matches:
            canonical = min(matches, key=lambda candidate: candidate["support"])
            item["status"] = "feasible"
            item["weights"] = list(canonical["weights"])
            item["canonicalized_to"] = canonical["support"]
        else:
            item["status"] = "infeasible"
    return audits


def _select_weight_tie(candidates: Sequence[SimplexCandidate]) -> SimplexCandidate:
    remaining = list(candidates)
    largest_parent = max(float(item.weights[0]) for item in remaining)
    remaining = [
        item
        for item in remaining
        if float(item.weights[0]) >= largest_parent - V009.FEASIBILITY_TOLERANCE
    ]
    for index in range(1, 5):
        smallest = min(float(item.weights[index]) for item in remaining)
        remaining = [
            item
            for item in remaining
            if float(item.weights[index]) <= smallest + V009.FEASIBILITY_TOLERANCE
        ]
    return min(remaining, key=lambda item: item.face)


def fit_simplex_weights(streams, target) -> SimplexFit:
    predictions = np.asarray(streams, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] != 5:
        raise ValueError("Five-stream input must have shape (n,5)")
    if truth.shape != (len(predictions),) or not len(truth):
        raise ValueError("Target must align with non-empty streams")
    if not np.isfinite(predictions).all() or not np.isfinite(truth).all():
        raise ValueError("Simplex inputs must be finite")
    delta = predictions[:, 1:] - predictions[:, [0]]
    residual = truth - predictions[:, 0]
    gram = delta.T @ delta
    cross = delta.T @ residual
    residual_squared = float(np.dot(residual, residual))
    audits = enumerate_simplex_supports(gram, cross, residual_squared)
    candidates = []
    for item in audits:
        if item["status"] != "feasible":
            continue
        weights = np.asarray(item["weights"], dtype=np.float64)
        if np.any(weights < 0.0) or not np.isclose(
            weights.sum(), 1.0, rtol=0.0, atol=V009.WEIGHT_SUM_TOLERANCE
        ):
            item["status"] = "infeasible-final-contract"
            continue
        direct_residual = truth - predictions @ weights
        direct_sse = float(np.dot(direct_residual, direct_residual))
        quadratic = _quadratic_sse(gram, cross, residual_squared, weights[1:])
        discrepancy = abs(quadratic - direct_sse)
        item.update(
            {
                "quadratic_squared_error": quadratic,
                "direct_squared_error": direct_sse,
                "quadratic_direct_discrepancy": discrepancy,
                "direct_sse_verified": discrepancy
                <= V009.DIRECT_SSE_TOLERANCE * max(1.0, direct_sse),
            }
        )
        if not item["direct_sse_verified"]:
            item["status"] = "singular-invalid"
            continue
        candidates.append(SimplexCandidate(item["support"], weights, direct_sse))
    if not candidates:
        raise RuntimeError("No direct-SSE-verified simplex support")
    minimum = min(item.squared_error for item in candidates)
    tied = [
        item
        for item in candidates
        if abs(item.squared_error - minimum)
        <= V009.TIE_EPS_MULTIPLIER
        * np.finfo(np.float64).eps
        * max(1.0, item.squared_error, minimum)
    ]
    selected = _select_weight_tie(tied)
    direct = truth - predictions @ selected.weights
    direct_sse = float(np.dot(direct, direct))
    return SimplexFit(
        selected.weights,
        direct_sse,
        len(truth),
        gram,
        cross,
        residual_squared,
        selected.face,
        len(candidates),
        EXPECTED_SUPPORTS,
        True,
        tuple(audits),
    )


def nested_five_stream_stack(frame, folds=(1, 2, 3, 4, 5)):
    fold_values = tuple(sorted(V009._integer_fold_values(np.asarray(folds)).tolist()))
    if fold_values != (1, 2, 3, 4, 5):
        raise ValueError("v010 requires canonical folds 1..5")
    shaped, progress_diagnostics = add_progress_stream(frame)
    required = {
        "_target",
        "fold",
        *STREAM_COLUMNS,
        *(
            nested_stream_column(outer, stream)
            for outer in fold_values
            for stream in ("sp_plane_ancc_k10", "sp_plane_best6_k10")
        ),
    }
    missing = sorted(required - set(shaped.columns))
    if missing:
        raise ValueError(f"Joined v010 frame is missing columns: {missing}")
    row_folds = V009._integer_fold_values(shaped["fold"])
    if sorted(np.unique(row_folds).tolist()) != list(fold_values):
        raise ValueError("Joined v010 rows do not contain canonical folds")
    target = shaped["_target"].to_numpy(dtype=np.float64)
    canonical = shaped.loc[:, STREAM_COLUMNS].to_numpy(dtype=np.float64)
    oof = np.full(len(shaped), np.nan, dtype=np.float64)
    selections = []
    for outer in fold_values:
        train_mask = row_folds != outer
        valid_mask = row_folds == outer
        meta_columns = (
            "parent_v006",
            "hmm_wr21_slow",
            "hmm_front20",
            nested_stream_column(outer, "sp_plane_ancc_k10"),
            nested_stream_column(outer, "sp_plane_best6_k10"),
        )
        fit = fit_simplex_weights(
            shaped.loc[train_mask, meta_columns].to_numpy(dtype=np.float64),
            target[train_mask],
        )
        prediction = canonical[valid_mask] @ fit.weights
        oof[valid_mask] = prediction
        selections.append(
            {
                "fold": outer,
                "meta_training_folds": sorted(set(fold_values) - {outer}),
                "meta_stream_columns": list(meta_columns),
                "evaluation_stream_columns": list(STREAM_COLUMNS),
                "weights": dict(zip(STREAM_COLUMNS, fit.weights.tolist())),
                "training_rows": fit.rows,
                "training_squared_error": fit.squared_error,
                "validation_rows": int(valid_mask.sum()),
                "validation_rmse": rmse(target[valid_mask], prediction),
                "selected_face": fit.selected_face,
                "attempted_systems": fit.attempted_systems,
                "feasible_candidates": fit.feasible_candidates,
                "direct_sse_verified": fit.direct_sse_verified,
                "support_audit": list(fit.support_audit),
            }
        )
    if not np.isfinite(oof).all():
        raise ValueError("Nested v010 OOF is incomplete or non-finite")
    final_fit = fit_simplex_weights(canonical, target)
    return shaped, selections, final_fit, oof, progress_diagnostics


def candidate_decision(
    overall,
    folds_better_v009,
    *,
    integrity_violations=0,
):
    values = (folds_better_v009, integrity_violations)
    if not np.isfinite(overall) or any(int(value) != value or value < 0 for value in values):
        raise ValueError("Invalid v010 candidate gate inputs")
    integrity_ok = int(integrity_violations) == 0
    if overall < PROMISING_GATE and folds_better_v009 >= 3:
        return "diagnostic_improvement_not_candidate" if integrity_ok else "exhausted"
    return "exhausted"


def fit_integrity_violations(selections, final_fit) -> tuple[int, int]:
    support_violations = 0
    direct_sse_violations = 0
    payloads = [*selections, _fit_payload(final_fit)]
    for payload in payloads:
        support_audit = payload.get("support_audit", [])
        support_names = [item.get("support") for item in support_audit]
        support_violations += int(
            payload.get("attempted_systems") != EXPECTED_SUPPORTS
            or len(support_audit) != EXPECTED_SUPPORTS
            or len(set(support_names)) != EXPECTED_SUPPORTS
        )
        direct_sse_violations += int(payload.get("direct_sse_verified") is not True)
    return support_violations, direct_sse_violations


def _plan_path(workspace):
    return Path(workspace) / "reports" / "v010_progress_shaped_hmm_plan.md"


def _validate_v009_audit(workspace, path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "verified_no_blocker" or payload.get("version") != "v009":
        raise ValueError("v009 artifact audit is not verified_no_blocker")
    hashes = payload.get("sha256", {})
    required = {"scripts/rogii_residual_simplex.py", "models/v009/run.json"}
    if not required.issubset(hashes):
        raise ValueError("v009 audit lacks source/run provenance hashes")
    mismatches = [
        relative
        for relative, expected in hashes.items()
        if not (Path(workspace) / relative).is_file()
        or sha256_file(Path(workspace) / relative) != expected
    ]
    if mismatches:
        raise ValueError(f"v009 artifact audit hash mismatch: {mismatches}")
    return payload


def critical_input_hashes(workspace, fold_manifest):
    workspace = Path(workspace)
    hashes = {
        f"upstream_{key}": value
        for key, value in V009.critical_input_hashes(workspace, fold_manifest).items()
    }
    extra = {
        "source_v010": Path(__file__).resolve(),
        "source_v009": Path(__file__).with_name("rogii_residual_simplex.py").resolve(),
        "plan_v010": _plan_path(workspace),
        "v009_run": workspace / "models" / "v009" / "run.json",
        "v009_artifact_audit": workspace / "reports" / "v009_artifact_audit.json",
        "v009_postmortem": workspace / "reports" / "v009_postmortem.md",
    }
    missing = [str(path) for path in extra.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Critical v010 inputs are missing: {missing}")
    run = json.loads(extra["v009_run"].read_text(encoding="utf-8"))
    if run.get("status") != "completed" or run.get("source_sha256") != sha256_file(
        extra["source_v009"]
    ):
        raise ValueError("v009 run/source contract changed")
    if run.get("raw_test_fingerprint") != run.get("raw_test_fingerprint_end"):
        raise ValueError("v009 raw-test fingerprint was not stable")
    _validate_v009_audit(workspace, extra["v009_artifact_audit"])
    hashes.update({key: sha256_file(path) for key, path in extra.items()})
    return hashes


def _fit_payload(fit):
    return {
        "weights": dict(zip(STREAM_COLUMNS, fit.weights.tolist())),
        "rows": fit.rows,
        "squared_error": fit.squared_error,
        "selected_face": fit.selected_face,
        "feasible_candidates": fit.feasible_candidates,
        "attempted_systems": fit.attempted_systems,
        "direct_sse_verified": fit.direct_sse_verified,
        "support_audit": list(fit.support_audit),
    }


def _write_summary(path, overall, fold_scores, decision, final_weights, test_status):
    text = "\n".join(
        [
            "# ROGII v010 Progress-Shaped HMM Simplex",
            "",
            f"Status: `{decision}`.",
            "",
            f"- Nested pooled OOF RMSE: `{overall:.9f}`",
            f"- Fold RMSE: `{', '.join(f'{value:.6f}' for value in fold_scores)}`",
            "- Final weights: "
            + ", ".join(
                f"`{name}={weight:.9f}`" for name, weight in zip(STREAM_COLUMNS, final_weights)
            ),
            f"- Visible-test status: `{test_status}`",
            "",
            "No submission file, notebook, kernel, or API request is created.",
            "",
        ]
    )
    V008._atomic_write_text(path, text)


def train(workspace, *, version=VERSION, folds=5):
    if version != VERSION or folds != 5:
        raise ValueError("The fixed experiment is exactly v010 with five folds")
    workspace = Path(workspace).resolve()
    fold_path, manifest = V008.ensure_fold_manifest(workspace, folds)
    input_start = critical_input_hashes(workspace, fold_path)
    raw_test_start = V008.data_fingerprint(workspace / "data" / "raw" / "test")
    model_dir = workspace / "models" / version
    staging = workspace / "models" / f".{version}.partial"
    for path in (model_dir, staging):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    reports = {
        "summary": workspace / "reports" / "v010_progress_simplex_summary.md",
        "folds": workspace / "reports" / "v010_progress_simplex_folds.csv",
        "by_well": workspace / "reports" / "v010_progress_simplex_by_well.csv",
        "diagnostics": workspace / "reports" / "v010_progress_simplex_diagnostics.json",
        "audit": workspace / "reports" / "v010_artifact_audit.json",
    }
    for path in reports.values():
        if path.exists() or Path(f"{path}.partial").exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    staging.mkdir(parents=True)
    started = time.perf_counter()
    common = {
        "version": version,
        "status": "running",
        "parent_runs": ["v006", "v007", "v008", "v009"],
        "template": "rogii-progress-shaped-hmm-five-stream-simplex",
        "command": " ".join(sys.argv),
        "source_sha256": input_start["source_v010"],
        "plan_sha256": input_start["plan_v010"],
        "fixed_settings": _fixed_settings(),
        "fold_manifest_sha256": input_start["upstream_fold_manifest"],
        "metric": "rmse",
        "metric_direction": "minimize",
        "target_column": "TVT",
        "input_sha256": input_start,
        "raw_test_fingerprint": raw_test_start,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "decision": "running",
        "test_predictions_saved": False,
        "submitted": False,
    }
    run_path = staging / "run.json"
    V008._atomic_write_json(run_path, {**common, "runtime_seconds": None, "error": None})
    old_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        joined, join_diag = V009.join_locked_features(
            workspace / "data" / "features" / "v004.parquet",
            workspace / "data" / "features" / "v005.parquet",
        )
        if len(joined) != 3_783_989 or joined["_well_id"].nunique() != 773:
            raise ValueError("v010 row/well contract changed")
        parent_diag = V009.validate_parent_alignment(workspace, joined, manifest)
        hmm_diag, boundary_wells = V009.validate_hmm_alignment(workspace, joined)
        shaped, selections, final_fit, oof, progress_diag = nested_five_stream_stack(joined)
        target = shaped["_target"].to_numpy(dtype=np.float64)
        row_folds = V009._integer_fold_values(shaped["fold"])
        fold_scores = [
            rmse(target[row_folds == fold], oof[row_folds == fold]) for fold in range(1, 6)
        ]
        overall = rmse(target, oof)
        v006 = V008.baseline_scores_on_manifest(workspace, manifest)
        v009 = json.loads((workspace / "models" / "v009" / "cv_scores.json").read_text())
        folds_better_v006 = sum(a < b for a, b in zip(fold_scores, v006["fold_scores"]))
        folds_better_v009 = sum(a < b for a, b in zip(fold_scores, v009["fold_scores"]))
        support_audit_violations, direct_sse_violations = fit_integrity_violations(
            selections, final_fit
        )
        alignment_violations = (
            int(join_diag["alignment_violations"])
            + int(parent_diag["violations"])
            + int(hmm_diag["violations"])
        )
        progress_violations = int(progress_diag["violations"])
        integrity_violations = (
            support_audit_violations
            + direct_sse_violations
            + alignment_violations
            + progress_violations
        )
        decision = candidate_decision(
            overall, folds_better_v009, integrity_violations=integrity_violations
        )
        test_status = "forbidden_adaptive_diagnostic"
        oof_rows = shaped.loc[:, ["_id", "_well_id", "_row_index", "_target", "fold"]]
        scored = oof_rows.assign(squared_error=np.square(target - oof))
        by_well = (
            scored.groupby(["_well_id", "fold"], sort=True)
            .agg(rows=("squared_error", "size"), squared_error=("squared_error", "sum"))
            .reset_index()
            .rename(columns={"_well_id": "well"})
        )
        by_well["rmse"] = np.sqrt(by_well["squared_error"] / by_well["rows"])
        fold_frame = pd.DataFrame(
            [
                {
                    "fold": item["fold"],
                    "validation_rmse": score,
                    **{f"weight_{name}": item["weights"][name] for name in STREAM_COLUMNS},
                    "selected_face": item["selected_face"],
                    "attempted_systems": item["attempted_systems"],
                }
                for item, score in zip(selections, fold_scores)
            ]
        )
        diagnostic_payload = {
            "join": join_diag,
            "parent": parent_diag,
            "hmm": hmm_diag,
            "progress": progress_diag,
            "boundary_wells": boundary_wells,
            "fold_fits": selections,
            "final_fit": _fit_payload(final_fit),
            "test_inference_forbidden": True,
        }
        V008._atomic_write_csv(reports["folds"], fold_frame)
        V008._atomic_write_csv(reports["by_well"], by_well)
        V008._atomic_write_json(reports["diagnostics"], diagnostic_payload)
        _write_summary(
            reports["summary"], overall, fold_scores, decision, final_fit.weights, test_status
        )
        cv = {
            "metric": "rmse",
            "direction": "minimize",
            "fold_scores": fold_scores,
            "overall": overall,
            "valid_rows": len(shaped),
            "valid_wells": int(shaped["_well_id"].nunique()),
            "fold_selections": selections,
            "final_fit": _fit_payload(final_fit),
            "canonical_v006": v006,
            "canonical_v009": {"overall": v009["overall"], "fold_scores": v009["fold_scores"]},
            "folds_better_than_v006": folds_better_v006,
            "folds_better_than_v009": folds_better_v009,
            "alignment_violations": alignment_violations,
            "progress_violations": progress_violations,
            "support_audit_violations": support_audit_violations,
            "direct_sse_violations": direct_sse_violations,
            "integrity_violations": integrity_violations,
            "decision": decision,
        }
        model = {
            "model": "progress_shaped_hmm_five_stream_simplex",
            "fold_selections": selections,
            "final_fit": _fit_payload(final_fit),
            "fixed_settings": _fixed_settings(),
        }
        importance = pd.DataFrame(
            [
                {
                    "scope": "outer_fold",
                    "fold": item["fold"],
                    "feature": name,
                    "weight": item["weights"][name],
                }
                for item in selections
                for name in STREAM_COLUMNS
            ]
            + [
                {"scope": "final_all_oof", "fold": 0, "feature": name, "weight": weight}
                for name, weight in zip(STREAM_COLUMNS, final_fit.weights)
            ]
        )
        V008._atomic_write_json(staging / "cv_scores.json", cv)
        V008._atomic_write_npy(staging / "oof_preds.npy", oof)
        V008._atomic_write_parquet(staging / "oof_rows.parquet", oof_rows)
        V008._atomic_write_pickle(staging / "model.pkl", model)
        V008._atomic_write_text(staging / "feature_list.txt", "\n".join(STREAM_COLUMNS) + "\n")
        V008._atomic_write_csv(staging / "importance.csv", importance)
        input_end = critical_input_hashes(workspace, fold_path)
        raw_test_end = V008.data_fingerprint(workspace / "data" / "raw" / "test")
        if input_end != input_start or raw_test_end != raw_test_start:
            raise ValueError("v010 critical inputs changed during run")
        outputs = {
            **{f"report_{key}": path for key, path in reports.items() if key != "audit"},
            **{
                name: staging / filename
                for name, filename in {
                    "cv_scores": "cv_scores.json",
                    "oof_predictions": "oof_preds.npy",
                    "oof_rows": "oof_rows.parquet",
                    "model": "model.pkl",
                    "feature_list": "feature_list.txt",
                    "importance": "importance.csv",
                }.items()
            },
        }
        output_hashes = {key: sha256_file(path) for key, path in outputs.items()}
        run = {
            **common,
            "status": "completed",
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "error": None,
            "input_sha256_end": input_end,
            "raw_test_fingerprint_end": raw_test_end,
            "output_sha256": output_hashes,
            "decision": decision,
            "cv_score": overall,
            "folds_better_than_v006": folds_better_v006,
            "folds_better_than_v009": folds_better_v009,
            "final_weights": dict(zip(STREAM_COLUMNS, final_fit.weights.tolist())),
            "boundary_well_count": len(boundary_wells),
            "alignment_violations": alignment_violations,
            "progress_violations": progress_violations,
            "support_audit_violations": support_audit_violations,
            "direct_sse_violations": direct_sse_violations,
            "integrity_violations": integrity_violations,
            "test_inference_forbidden": True,
            "test_predictions_saved": False,
        }
        V008._atomic_replace_json(run_path, run)
        audit_paths = {
            "scripts/rogii_progress_simplex.py": Path(__file__).resolve(),
            "reports/v010_progress_shaped_hmm_plan.md": _plan_path(workspace),
            "models/v010/run.json": run_path,
            **{
                f"models/v010/{path.name}": path
                for path in outputs.values()
                if str(path).startswith(str(staging))
            },
            **{
                str(path.relative_to(workspace)): path
                for key, path in reports.items()
                if key != "audit"
            },
        }
        V008._atomic_write_json(
            reports["audit"],
            {
                "schema_version": 1,
                "version": version,
                "status": "verified_no_blocker",
                "decision": decision,
                "sha256": {name: sha256_file(path) for name, path in audit_paths.items()},
            },
        )
        partials = [*staging.rglob("*.partial")]
        partials.extend(
            Path(f"{path}.partial") for path in reports.values() if Path(f"{path}.partial").exists()
        )
        if partials:
            raise ValueError(f"v010 partial artifacts remain before publish: {partials}")
        os.replace(staging, model_dir)
        return model_dir
    except BaseException as exc:
        if run_path.is_file():
            try:
                V008._atomic_replace_json(
                    run_path,
                    {
                        **common,
                        "status": "failed",
                        "runtime_seconds": round(time.perf_counter() - started, 3),
                        "error": f"{type(exc).__name__}: {exc}",
                        "decision": "failed",
                    },
                )
            except Exception:
                pass
        if staging.exists() and not model_dir.exists():
            os.replace(staging, model_dir)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--version", default=VERSION)
    train_parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "train":
        print(train(args.workspace, version=args.version, folds=args.folds))
        return 0
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
