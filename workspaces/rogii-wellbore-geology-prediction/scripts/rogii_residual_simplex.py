#!/usr/bin/env python3
"""Cross-fitted four-stream residual simplex for the fixed ROGII v009 protocol.

This module reads only immutable v006/v007/v008 artifacts.  It never reruns an
upstream predictor and has no submission, notebook, Kaggle kernel, or API path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import platform
import signal
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - runtime preflight handles this
    pq = None


VERSION = "v009"
PARENT_VERSION = "v006"
HMM_VERSION = "v007"
FORMATION_VERSION = "v008"
PARENT_CANDIDATE = "pf_scale_12_hold_0p2"
STREAM_COLUMNS = (
    "parent_v006",
    "hmm_wr21_slow",
    "sp_plane_ancc_k10",
    "sp_plane_best6_k10",
)
BASE_COLUMNS = ("_id", "_well_id", "_row_index", "_target", "fold", "parent_v006")
CANDIDATE_GATE = 10.95
PROMISING_GATE = 11.75
LINEAR_RESIDUAL_TOLERANCE = 1e-10
DIRECT_SSE_TOLERANCE = 1e-8
FEASIBILITY_TOLERANCE = 1e-10
WEIGHT_SUM_TOLERANCE = 1e-12
TIE_EPS_MULTIPLIER = 64.0
EXPECTED_ACTIVE_SYSTEMS = 15


def _load_v008_module():
    name = "rogii_formation_spatial_v008_dependency"
    if name in sys.modules:
        return sys.modules[name]
    source = Path(__file__).with_name("rogii_formation_spatial.py").resolve()
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load v008 helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V008 = _load_v008_module()


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
    """Raised in the main thread so SIGTERM can publish a failed run record."""


def _raise_on_sigterm(signum, frame) -> None:
    del frame
    raise TerminationRequested(f"received signal {signum}")


def sha256_file(path: Path) -> str:
    return V008.sha256_file(Path(path))


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return V008.rmse(target, prediction)


def nested_stream_column(outer_fold: int, stream: str) -> str:
    return V008.nested_stream_column(outer_fold, stream)


def _integer_fold_values(values: pd.Series | np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.array_equal(numeric, np.rint(numeric)):
        raise ValueError("Fold labels must be finite exact integers")
    return numeric.astype(np.int64)


def _fixed_settings() -> dict:
    return {
        "streams": list(STREAM_COLUMNS),
        "join_key": "globally unique _id",
        "join_order": "v005 canonical row order",
        "exact_join_fields": ["_well_id", "_row_index", "_target", "fold", "parent_v006"],
        "simplex_constraints": "weights >= 0, sum(weights) = 1, no intercept",
        "simplex_base": "parent_v006",
        "delta_streams": list(STREAM_COLUMNS[1:]),
        "active_systems": EXPECTED_ACTIVE_SYSTEMS,
        "parent_active_faces": 7,
        "parent_zero_faces": 7,
        "all_parent_vertex": True,
        "linear_solver": "solve, then deterministic lstsq only with original-equation check",
        "linear_residual_tolerance": LINEAR_RESIDUAL_TOLERANCE,
        "lstsq_rcond": 1e-12,
        "raw_feasibility_tolerance": FEASIBILITY_TOLERANCE,
        "final_weight_sum_tolerance": WEIGHT_SUM_TOLERANCE,
        "direct_sse_tolerance": DIRECT_SSE_TOLERANCE,
        "row_sampling": False,
        "regularization": False,
        "weight_clipping": False,
        "weight_renormalization": False,
        "tie_tolerance": "64*eps(float64)*max(1,SSE_a,SSE_b)",
        "tie_break": "max parent, min HMM, min ANCC, min best6",
        "outer_meta_formation": "nested_outer_k columns on folds j != k",
        "outer_evaluation_formation": "canonical formation columns on fold k",
    }


def _solve_checked(matrix: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray | None, str, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Linear system matrix must be square")
    if rhs.shape != (matrix.shape[0],):
        raise ValueError("Linear system right-hand side has the wrong shape")
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise ValueError("Linear system contains non-finite values")
    try:
        solution = np.linalg.solve(matrix, rhs)
        solve_status = "solved"
    except np.linalg.LinAlgError:
        solution, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=1e-12)
        solve_status = "lstsq"
    if not np.isfinite(solution).all():
        return None, "singular-invalid", np.inf
    residual = matrix @ solution - rhs
    scale = max(
        1.0,
        float(np.linalg.norm(rhs, ord=np.inf)),
        float(
            np.linalg.norm(matrix, ord=np.inf)
            * max(1.0, float(np.linalg.norm(solution, ord=np.inf)))
        ),
    )
    residual_norm = float(np.linalg.norm(residual, ord=np.inf))
    if residual_norm > LINEAR_RESIDUAL_TOLERANCE * scale:
        return None, "singular-invalid", residual_norm
    return solution, solve_status, residual_norm


def _quadratic_sse(
    gram: np.ndarray,
    cross: np.ndarray,
    residual_squared: float,
    delta_weights: np.ndarray,
) -> float:
    return float(
        residual_squared - 2.0 * np.dot(cross, delta_weights) + delta_weights @ gram @ delta_weights
    )


def enumerate_simplex_supports(
    gram: np.ndarray, cross: np.ndarray, residual_squared: float
) -> list[dict]:
    """Audit all 15 supports before direct row-level SSE evaluation."""
    gram = np.asarray(gram, dtype=np.float64)
    cross = np.asarray(cross, dtype=np.float64)
    if gram.shape != (3, 3) or cross.shape != (3,):
        raise ValueError("Four-stream simplex requires a 3x3 Gram and length-three cross vector")
    if not np.isfinite(gram).all() or not np.isfinite(cross).all():
        raise ValueError("Simplex sufficient statistics must be finite")
    if not np.allclose(gram, gram.T, rtol=0.0, atol=1e-10):
        raise ValueError("Simplex Gram matrix must be symmetric")
    eigen_tolerance = LINEAR_RESIDUAL_TOLERANCE * max(1.0, float(np.max(np.abs(gram))))
    if float(np.min(np.linalg.eigvalsh(gram))) < -eigen_tolerance:
        raise ValueError("Simplex Gram matrix is not positive semidefinite")
    if not np.isfinite(residual_squared) or residual_squared < 0.0:
        raise ValueError("Residual squared norm must be finite and non-negative")

    audits: list[dict] = [
        {
            "support": "all_parent",
            "face_class": "vertex",
            "delta_subset": [],
            "solve_status": "solved",
            "status": "feasible",
            "linear_residual_inf": 0.0,
            "raw_weights": [1.0, 0.0, 0.0, 0.0],
            "weights": [1.0, 0.0, 0.0, 0.0],
            "canonicalized_to": None,
        }
    ]
    indices = range(3)
    for size in range(1, 4):
        for subset_tuple in combinations(indices, size):
            subset = np.asarray(subset_tuple, dtype=np.int64)
            sub_gram = gram[np.ix_(subset, subset)]
            sub_cross = cross[subset]

            parent_active, solve_status, solve_residual = _solve_checked(sub_gram, sub_cross)
            active_audit = {
                "support": "parent_active:" + ",".join(map(str, subset_tuple)),
                "face_class": "parent_active",
                "delta_subset": list(subset_tuple),
                "solve_status": solve_status,
                "status": "singular-invalid" if parent_active is None else "infeasible",
                "linear_residual_inf": solve_residual,
                "raw_weights": None,
                "weights": None,
                "canonicalized_to": None,
            }
            if parent_active is not None:
                delta_weights = np.zeros(3, dtype=np.float64)
                delta_weights[subset] = parent_active
                parent_weight = 1.0 - float(delta_weights.sum())
                weights = np.concatenate([[parent_weight], delta_weights])
                active_audit["raw_weights"] = weights.tolist()
                raw_feasible = bool(
                    np.all(weights >= -FEASIBILITY_TOLERANCE)
                    and float(delta_weights.sum()) <= 1.0 + FEASIBILITY_TOLERANCE
                )
                if raw_feasible and np.all(weights >= 0.0):
                    active_audit["status"] = "feasible"
                    active_audit["weights"] = weights.tolist()
                elif raw_feasible:
                    active_audit["status"] = "needs-canonicalization"
            audits.append(active_audit)

            kkt = np.block(
                [
                    [sub_gram, np.ones((size, 1), dtype=np.float64)],
                    [np.ones((1, size), dtype=np.float64), np.zeros((1, 1))],
                ]
            )
            kkt_rhs = np.concatenate([sub_cross, [1.0]])
            parent_zero, solve_status, solve_residual = _solve_checked(kkt, kkt_rhs)
            zero_audit = {
                "support": "parent_zero:" + ",".join(map(str, subset_tuple)),
                "face_class": "parent_zero",
                "delta_subset": list(subset_tuple),
                "solve_status": solve_status,
                "status": "singular-invalid" if parent_zero is None else "infeasible",
                "linear_residual_inf": solve_residual,
                "raw_weights": None,
                "weights": None,
                "canonicalized_to": None,
            }
            if parent_zero is not None:
                active_weights = parent_zero[:-1]
                delta_weights = np.zeros(3, dtype=np.float64)
                delta_weights[subset] = active_weights
                weights = np.concatenate([[0.0], delta_weights])
                zero_audit["raw_weights"] = weights.tolist()
                raw_feasible = bool(
                    np.all(weights >= -FEASIBILITY_TOLERANCE)
                    and abs(float(delta_weights.sum()) - 1.0) <= FEASIBILITY_TOLERANCE
                )
                if raw_feasible and np.all(weights >= 0.0):
                    zero_audit["status"] = "feasible"
                    zero_audit["weights"] = weights.tolist()
                elif raw_feasible:
                    zero_audit["status"] = "needs-canonicalization"
            audits.append(zero_audit)

    if len(audits) != EXPECTED_ACTIVE_SYSTEMS:
        raise RuntimeError(f"Audited {len(audits)} supports, expected 15")
    exact_feasible = [item for item in audits if item["status"] == "feasible"]
    for item in audits:
        if item["status"] != "needs-canonicalization":
            continue
        raw = np.asarray(item["raw_weights"], dtype=np.float64)
        matches = [
            candidate
            for candidate in exact_feasible
            if np.allclose(
                raw,
                np.asarray(candidate["weights"], dtype=np.float64),
                rtol=0.0,
                atol=FEASIBILITY_TOLERANCE,
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
    """Apply the declared preference without letting roundoff reorder equal weights."""
    remaining = list(candidates)
    if not remaining:
        raise ValueError("Weight tie selection requires at least one candidate")
    largest_parent = max(float(item.weights[0]) for item in remaining)
    remaining = [
        item
        for item in remaining
        if float(item.weights[0]) >= largest_parent - FEASIBILITY_TOLERANCE
    ]
    for index in range(1, 4):
        smallest = min(float(item.weights[index]) for item in remaining)
        remaining = [
            item
            for item in remaining
            if float(item.weights[index]) <= smallest + FEASIBILITY_TOLERANCE
        ]
    return min(remaining, key=lambda item: item.face)


def fit_simplex_weights(streams: np.ndarray, target: np.ndarray) -> SimplexFit:
    predictions = np.asarray(streams, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] != 4:
        raise ValueError("Four-stream simplex input must have shape (n, 4)")
    if truth.shape != (len(predictions),) or len(truth) == 0:
        raise ValueError("Simplex target must align with non-empty predictions")
    if not np.isfinite(predictions).all() or not np.isfinite(truth).all():
        raise ValueError("Simplex rows must be finite")
    delta = predictions[:, 1:] - predictions[:, [0]]
    residual = truth - predictions[:, 0]
    gram = delta.T @ delta
    cross = delta.T @ residual
    residual_squared = float(np.dot(residual, residual))
    support_audit = enumerate_simplex_supports(gram, cross, residual_squared)
    candidates: list[SimplexCandidate] = []
    for item in support_audit:
        if item["status"] != "feasible":
            continue
        weights = np.asarray(item["weights"], dtype=np.float64)
        if (
            not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=WEIGHT_SUM_TOLERANCE)
        ):
            item["status"] = "infeasible-final-contract"
            item["invalid_reason"] = "weights violate strict nonnegative/unit-sum contract"
            continue
        direct_residual = truth - predictions @ weights
        direct_sse = float(np.dot(direct_residual, direct_residual))
        formula_sse = _quadratic_sse(gram, cross, residual_squared, weights[1:])
        discrepancy = abs(formula_sse - direct_sse)
        item["quadratic_squared_error"] = formula_sse
        item["direct_squared_error"] = direct_sse
        item["quadratic_direct_discrepancy"] = discrepancy
        item["direct_sse_verified"] = bool(
            np.isfinite(direct_sse) and discrepancy <= DIRECT_SSE_TOLERANCE * max(1.0, direct_sse)
        )
        if not item["direct_sse_verified"]:
            item["status"] = "singular-invalid"
            item["invalid_reason"] = "quadratic/direct SSE discrepancy"
            continue
        candidates.append(
            SimplexCandidate(face=item["support"], weights=weights, squared_error=direct_sse)
        )
    if not candidates:
        raise RuntimeError("Simplex enumeration produced no direct-SSE-verified candidate")
    minimum_sse = min(candidate.squared_error for candidate in candidates)
    epsilon = np.finfo(np.float64).eps
    tied = [
        candidate
        for candidate in candidates
        if abs(candidate.squared_error - minimum_sse)
        <= TIE_EPS_MULTIPLIER * epsilon * max(1.0, candidate.squared_error, minimum_sse)
    ]
    selected = _select_weight_tie(tied)
    weights = selected.weights
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("Selected simplex weights are invalid")
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=WEIGHT_SUM_TOLERANCE):
        raise ValueError("Selected simplex weights do not sum to one")
    direct_residual = truth - predictions @ weights
    direct_sse = float(np.dot(direct_residual, direct_residual))
    formula_sse = _quadratic_sse(gram, cross, residual_squared, weights[1:])
    tolerance = DIRECT_SSE_TOLERANCE * max(1.0, direct_sse)
    if not np.isfinite(direct_sse) or abs(formula_sse - direct_sse) > tolerance:
        raise ValueError("Selected simplex failed direct full-row SSE verification")
    return SimplexFit(
        weights=weights,
        squared_error=direct_sse,
        rows=len(truth),
        delta_gram=gram,
        delta_cross=cross,
        residual_squared=residual_squared,
        selected_face=selected.face,
        feasible_candidates=len(candidates),
        attempted_systems=EXPECTED_ACTIVE_SYSTEMS,
        direct_sse_verified=True,
        support_audit=tuple(support_audit),
    )


def _required_hmm_columns() -> list[str]:
    return [*BASE_COLUMNS, "hmm_wr21_slow"]


def _required_formation_columns() -> list[str]:
    return [
        *BASE_COLUMNS,
        "sp_plane_ancc_k10",
        "sp_plane_best6_k10",
        *(
            nested_stream_column(outer, stream)
            for outer in range(1, 6)
            for stream in ("sp_plane_ancc_k10", "sp_plane_best6_k10")
        ),
    ]


def join_locked_features(hmm_path: Path, formation_path: Path) -> tuple[pd.DataFrame, dict]:
    """Join globally by unique _id and restore immutable v005 canonical order."""
    hmm_path = Path(hmm_path)
    formation_path = Path(formation_path)
    for path in (hmm_path, formation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    hmm = pd.read_parquet(hmm_path, columns=_required_hmm_columns())
    formation = pd.read_parquet(formation_path, columns=_required_formation_columns())
    if hmm["_id"].isna().any() or formation["_id"].isna().any():
        raise ValueError("Locked feature IDs may not be missing")
    if hmm["_id"].duplicated().any() or formation["_id"].duplicated().any():
        raise ValueError("Locked feature IDs must be globally unique on both sides")
    if len(hmm) != len(formation) or set(hmm["_id"]) != set(formation["_id"]):
        raise ValueError("v004 and v005 feature ID sets do not match exactly")

    formation = formation.assign(_canonical_order=np.arange(len(formation), dtype=np.int64))
    joined = formation.merge(
        hmm,
        on="_id",
        how="outer",
        validate="one_to_one",
        suffixes=("__formation", "__hmm"),
        indicator=True,
        sort=False,
    )
    if not (joined["_merge"] == "both").all():
        raise ValueError("v004/v005 one-to-one ID join was incomplete")
    joined = joined.sort_values("_canonical_order", kind="stable").reset_index(drop=True)
    exact_fields = ("_well_id", "_row_index", "_target", "fold", "parent_v006")
    for field in exact_fields:
        left = joined[f"{field}__formation"].to_numpy()
        right = joined[f"{field}__hmm"].to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"v004/v005 exact field mismatch: {field}")

    result = pd.DataFrame(
        {
            "_id": joined["_id"].astype(str),
            **{field: joined[f"{field}__formation"].to_numpy() for field in exact_fields},
            "hmm_wr21_slow": joined["hmm_wr21_slow"].to_numpy(dtype=np.float64),
            "sp_plane_ancc_k10": joined["sp_plane_ancc_k10"].to_numpy(dtype=np.float64),
            "sp_plane_best6_k10": joined["sp_plane_best6_k10"].to_numpy(dtype=np.float64),
        }
    )
    for outer in range(1, 6):
        for stream in ("sp_plane_ancc_k10", "sp_plane_best6_k10"):
            column = nested_stream_column(outer, stream)
            result[column] = joined[column].to_numpy(dtype=np.float64)
    numeric = result.drop(columns=["_id", "_well_id"]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Joined locked feature rows contain non-finite values")
    if result["_id"].duplicated().any():
        raise ValueError("Joined feature IDs are not globally unique")
    _integer_fold_values(result["fold"])
    return result, {
        "rows": len(result),
        "left_v005_rows": len(formation),
        "right_v004_rows": len(hmm),
        "matched_rows": len(result),
        "wells": int(result["_well_id"].nunique()),
        "join_key": "_id",
        "global_unique_ids": True,
        "left_duplicate_ids": 0,
        "right_duplicate_ids": 0,
        "missing_left_ids": 0,
        "missing_right_ids": 0,
        "id_sets_equal": True,
        "canonical_order_source": str(formation_path),
        "canonical_order_restored": True,
        "exact_fields_verified": list(exact_fields),
        "exact_field_mismatches": 0,
        "alignment_violations": 0,
    }


def _outer_meta_columns(outer_fold: int) -> tuple[str, str, str, str]:
    return (
        "parent_v006",
        "hmm_wr21_slow",
        nested_stream_column(outer_fold, "sp_plane_ancc_k10"),
        nested_stream_column(outer_fold, "sp_plane_best6_k10"),
    )


def nested_four_stream_stack(
    frame: pd.DataFrame, folds: Sequence[int] = (1, 2, 3, 4, 5)
) -> tuple[list[dict], SimplexFit, np.ndarray]:
    """Fit outer weights on nested formation streams and evaluate canonical streams."""
    fold_values = tuple(sorted(_integer_fold_values(np.asarray(folds)).tolist()))
    if fold_values != (1, 2, 3, 4, 5):
        raise ValueError("v009 requires canonical folds labelled exactly 1..5")
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
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Joined v009 frame is missing columns: {missing}")
    row_folds = _integer_fold_values(frame["fold"])
    observed_folds = sorted(np.unique(row_folds).tolist())
    if observed_folds != list(fold_values):
        raise ValueError("Joined v009 rows do not contain exactly the canonical folds")
    target = frame["_target"].to_numpy(dtype=np.float64)
    canonical_streams = frame.loc[:, STREAM_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or not np.isfinite(canonical_streams).all():
        raise ValueError("Canonical v009 target/streams contain non-finite values")

    oof = np.full(len(frame), np.nan, dtype=np.float64)
    selections: list[dict] = []
    for outer_fold in fold_values:
        train_mask = row_folds != outer_fold
        validation_mask = row_folds == outer_fold
        meta_columns = _outer_meta_columns(outer_fold)
        meta_streams = frame.loc[train_mask, meta_columns].to_numpy(dtype=np.float64)
        meta_target = target[train_mask]
        expected_meta_folds = sorted(set(fold_values) - {outer_fold})
        if sorted(np.unique(row_folds[train_mask]).tolist()) != expected_meta_folds:
            raise ValueError(f"Outer fold {outer_fold} meta rows violate canonical isolation")
        fit = fit_simplex_weights(meta_streams, meta_target)
        validation_streams = canonical_streams[validation_mask]
        prediction = validation_streams @ fit.weights
        if not np.isfinite(prediction).all():
            raise ValueError(f"Outer fold {outer_fold} produced non-finite held-out predictions")
        oof[validation_mask] = prediction
        validation_target = target[validation_mask]
        selections.append(
            {
                "fold": outer_fold,
                "meta_training_folds": expected_meta_folds,
                "meta_stream_columns": list(meta_columns),
                "evaluation_stream_columns": list(STREAM_COLUMNS),
                "weights": dict(zip(STREAM_COLUMNS, fit.weights.tolist())),
                "training_rows": fit.rows,
                "training_squared_error": fit.squared_error,
                "training_rmse": float(np.sqrt(fit.squared_error / fit.rows)),
                "validation_rows": int(validation_mask.sum()),
                "validation_rmse": rmse(validation_target, prediction),
                "selected_face": fit.selected_face,
                "feasible_candidates": fit.feasible_candidates,
                "attempted_systems": fit.attempted_systems,
                "centered_delta_gram": fit.delta_gram.tolist(),
                "centered_delta_cross": fit.delta_cross.tolist(),
                "centered_residual_squared": fit.residual_squared,
                "direct_sse_verified": fit.direct_sse_verified,
                "support_audit": list(fit.support_audit),
            }
        )
    if not np.isfinite(oof).all():
        raise ValueError("Nested v009 OOF contains uncovered or non-finite rows")
    final_fit = fit_simplex_weights(canonical_streams, target)
    return selections, final_fit, oof


def candidate_decision(
    overall: float,
    folds_better: int,
    *,
    alignment_violations: int = 0,
    fold_violations: int = 0,
    donor_provenance_violations: int = 0,
    nonfinite_violations: int = 0,
    direct_sse_violations: int = 0,
) -> str:
    if not np.isfinite(overall) or int(folds_better) != folds_better or folds_better < 0:
        raise ValueError("Candidate gate score/fold count is invalid")
    counts = (
        alignment_violations,
        fold_violations,
        donor_provenance_violations,
        nonfinite_violations,
        direct_sse_violations,
    )
    if any(int(value) != value or value < 0 for value in counts):
        raise ValueError("Candidate integrity counts must be non-negative integers")
    integrity_ok = sum(int(value) for value in counts) == 0
    if overall <= CANDIDATE_GATE and folds_better >= 4:
        return "candidate_ready" if integrity_ok else "candidate_blocked_integrity"
    if overall < PROMISING_GATE and folds_better >= 3:
        return "promising_continue" if integrity_ok else "promising_blocked_integrity"
    return "exhausted"


def _plan_path(workspace: Path) -> Path:
    return Path(workspace) / "reports" / "v009_cross_fitted_residual_simplex_plan.md"


def _critical_paths(workspace: Path, fold_manifest: Path) -> dict[str, Path]:
    workspace = Path(workspace)
    return {
        "config": workspace / "config.yaml",
        "source_v009": Path(__file__).resolve(),
        "source_v007": Path(__file__).with_name("rogii_hmm_smoother.py").resolve(),
        "source_v008": Path(__file__).with_name("rogii_formation_spatial.py").resolve(),
        "plan_v009": _plan_path(workspace),
        "feature_v004": workspace / "data" / "features" / "v004.parquet",
        "feature_v005": workspace / "data" / "features" / "v005.parquet",
        "v007_artifact_audit": workspace / "reports" / "v007_artifact_audit.json",
        "v007_cv_scores": workspace / "models" / "v007" / "cv_scores.json",
        "v007_oof_rows": workspace / "models" / "v007" / "oof_rows.parquet",
        "v007_oof_predictions": workspace / "models" / "v007" / "oof_preds.npy",
        "v007_boundary_diagnostics": workspace / "reports" / "v007_hmm_diagnostics.json",
        "v006_run": workspace / "models" / "v006" / "run.json",
        "v007_run": workspace / "models" / "v007" / "run.json",
        "v008_run": workspace / "models" / "v008" / "run.json",
        "v006_oof_rows": workspace / "models" / "v006" / "oof_rows.parquet",
        "v006_oof_predictions": workspace / "models" / "v006" / "oof_preds.npy",
        "v006_test_predictions": workspace / "models" / "v006" / "test_preds.npy",
        "v008_model": workspace / "models" / "v008" / "model.pkl",
        "sample_submission": workspace / "data" / "raw" / "sample_submission.csv",
        "fold_manifest": Path(fold_manifest),
    }


def _validate_v007_artifact_audit(workspace: Path, audit_path: Path) -> dict:
    payload = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    if payload.get("status") != "verified_no_blocker" or payload.get("version") != "v007":
        raise ValueError("v007 artifact audit is not verified_no_blocker")
    hashes = payload.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("v007 artifact audit has no SHA256 mapping")
    required = {"data/features/v004.parquet", "models/v007/run.json"}
    if not required.issubset(hashes):
        raise ValueError("v007 artifact audit lacks required provenance bridge hashes")
    mismatches = []
    for relative, expected_hash in hashes.items():
        path = Path(workspace) / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(f"v007 artifact audit hash mismatch: {mismatches}")
    return payload


def _validate_upstream_runs(
    workspace: Path, fold_manifest: Path, paths: Mapping[str, Path]
) -> dict:
    workspace = Path(workspace)
    V008._validate_parent_run_contract(workspace, fold_manifest, paths["v006_run"])
    fold_hash = sha256_file(fold_manifest)
    v007 = json.loads(paths["v007_run"].read_text(encoding="utf-8"))
    v008 = json.loads(paths["v008_run"].read_text(encoding="utf-8"))
    v007_expected = {
        "status": "completed",
        "source_sha256": sha256_file(paths["source_v007"]),
        "fold_manifest_sha256": fold_hash,
        "feature_path": "data/features/v004.parquet",
        "final_candidate": "hmm_wr21_slow",
    }
    v008_expected = {
        "status": "completed",
        "source_sha256": sha256_file(paths["source_v008"]),
        "fold_manifest_sha256": fold_hash,
        "feature_path": "data/features/v005.parquet",
    }
    v007_mismatch = [key for key, value in v007_expected.items() if v007.get(key) != value]
    v008_mismatch = [key for key, value in v008_expected.items() if v008.get(key) != value]
    if v007_mismatch:
        raise ValueError(f"v007 run contract changed: {v007_mismatch}")
    if v008_mismatch:
        raise ValueError(f"v008 run contract changed: {v008_mismatch}")
    output_hashes = v008.get("output_sha256", {})
    if output_hashes.get("feature") != sha256_file(paths["feature_v005"]):
        raise ValueError("v008 run no longer binds the current v005 feature")
    if output_hashes.get("model") != sha256_file(paths["v008_model"]):
        raise ValueError("v008 run no longer binds the current donor model")
    audit = _validate_v007_artifact_audit(workspace, paths["v007_artifact_audit"])
    return {"v007": v007, "v008": v008, "v007_artifact_audit": audit}


def critical_input_hashes(workspace: Path, fold_manifest: Path) -> dict[str, str]:
    paths = _critical_paths(workspace, fold_manifest)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Critical v009 inputs are missing: {missing}")
    _validate_upstream_runs(workspace, fold_manifest, paths)
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_parent_alignment(workspace: Path, frame: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    workspace = Path(workspace)
    rows = pd.read_parquet(
        workspace / "models" / "v006" / "oof_rows.parquet",
        columns=["_id", "_well_id", "_row_index", "_target", "fold"],
    )
    prediction = np.load(workspace / "models" / "v006" / "oof_preds.npy")
    if rows["_id"].isna().any() or rows["_id"].duplicated().any():
        raise ValueError("v006 parent OOF IDs are not globally unique and non-null")
    if len(rows) != len(prediction) or len(rows) != len(frame):
        raise ValueError("v006 parent OOF row count differs from joined streams")
    parent = rows.assign(_parent_prediction=np.asarray(prediction, dtype=np.float64))
    aligned = frame.loc[:, BASE_COLUMNS].merge(
        parent,
        on="_id",
        how="left",
        validate="one_to_one",
        suffixes=("__joined", "__parent"),
        sort=False,
    )
    if aligned["_parent_prediction"].isna().any():
        raise ValueError("v006 parent OOF does not cover every joined ID")
    for field in ("_well_id", "_row_index", "_target", "fold"):
        if not np.array_equal(
            aligned[f"{field}__joined"].to_numpy(),
            aligned[f"{field}__parent"].to_numpy(),
        ):
            raise ValueError(f"v006 parent alignment mismatch: {field}")
    if not np.array_equal(
        aligned["parent_v006"].to_numpy(dtype=np.float64),
        aligned["_parent_prediction"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("Joined parent_v006 values differ from locked v006 OOF")
    fold_map = manifest.set_index("well")["fold"].astype(int)
    mapped = frame["_well_id"].map(fold_map)
    if mapped.isna().any() or not np.array_equal(
        mapped.to_numpy(dtype=int), frame["fold"].to_numpy(dtype=int)
    ):
        raise ValueError("Joined folds differ from the canonical manifest")
    return {"rows": len(frame), "ids_aligned": True, "target_aligned": True, "violations": 0}


def validate_hmm_alignment(
    workspace: Path, frame: pd.DataFrame, *, expected_boundary_count: int = 186
) -> tuple[dict, list[str]]:
    workspace = Path(workspace)
    cv = json.loads((workspace / "models" / "v007" / "cv_scores.json").read_text())
    selections = cv.get("fold_selections", [])
    if len(selections) != 5 or {str(item.get("selected_candidate")) for item in selections} != {
        "hmm_wr21_slow"
    }:
        raise ValueError("v007 outer selections are not unanimously hmm_wr21_slow")
    if cv.get("final_candidate") != "hmm_wr21_slow":
        raise ValueError("v007 final candidate is not hmm_wr21_slow")
    rows = pd.read_parquet(
        workspace / "models" / "v007" / "oof_rows.parquet",
        columns=["_id", "_well_id", "_row_index", "_target", "fold"],
    )
    prediction = np.load(workspace / "models" / "v007" / "oof_preds.npy")
    if rows["_id"].isna().any() or rows["_id"].duplicated().any():
        raise ValueError("v007 OOF IDs are not globally unique and non-null")
    if len(rows) != len(prediction) or len(rows) != len(frame):
        raise ValueError("v007 OOF row count differs from joined streams")
    locked = rows.assign(_hmm_prediction=np.asarray(prediction, dtype=np.float64))
    aligned = frame.loc[:, [*BASE_COLUMNS, "hmm_wr21_slow"]].merge(
        locked,
        on="_id",
        how="left",
        validate="one_to_one",
        suffixes=("__joined", "__v007"),
        sort=False,
    )
    if aligned["_hmm_prediction"].isna().any():
        raise ValueError("v007 OOF does not cover every joined ID")
    for field in ("_well_id", "_row_index", "_target", "fold"):
        if not np.array_equal(
            aligned[f"{field}__joined"].to_numpy(),
            aligned[f"{field}__v007"].to_numpy(),
        ):
            raise ValueError(f"v007 OOF alignment mismatch: {field}")
    if not np.array_equal(
        aligned["hmm_wr21_slow"].to_numpy(dtype=np.float64),
        aligned["_hmm_prediction"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("v004 slow stream differs from locked v007 OOF")

    boundary = sorted(set(map(str, cv.get("selected_boundary_wells", []))))
    final_boundary = sorted(set(map(str, cv.get("final_boundary_wells", []))))
    diagnostics = json.loads(
        (workspace / "reports" / "v007_hmm_diagnostics.json").read_text(encoding="utf-8")
    )
    diagnostic_boundary = sorted(
        str(item["well"])
        for item in diagnostics.get("wells", [])
        if item["candidates"]["hmm_wr21_slow"]["unresolved_position_boundary"]
        or item["candidates"]["hmm_wr21_slow"]["unresolved_rate_boundary"]
    )
    if diagnostics.get("failures") != []:
        raise ValueError("v007 diagnostic artifact contains inference failures")
    if (
        len(boundary) != expected_boundary_count
        or boundary != final_boundary
        or boundary != diagnostic_boundary
    ):
        raise ValueError("v007 inherited boundary-well evidence changed")
    return (
        {
            "rows": len(frame),
            "ids_aligned": True,
            "slow_values_aligned": True,
            "unanimous_slow_selection": True,
            "boundary_wells": len(boundary),
            "violations": 0,
        },
        boundary,
    )


def _load_source_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load locked source from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def predict_visible_test(
    workspace: Path, final_weights: Sequence[float]
) -> tuple[np.ndarray | None, list[dict], list[dict]]:
    """Rebuild all four target-free test streams; return sample-aligned values only."""
    workspace = Path(workspace)
    weights = np.asarray(final_weights, dtype=np.float64)
    if weights.shape != (4,) or np.any(weights < 0.0) or not np.isfinite(weights).all():
        raise ValueError("Visible-test weights are not finite non-negative four-stream weights")
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=WEIGHT_SUM_TOLERANCE):
        raise ValueError("Visible-test weights do not sum to one")

    sample = pd.read_csv(workspace / "data" / "raw" / "sample_submission.csv")
    if list(sample.columns) != ["id", "tvt"] or sample["id"].isna().any():
        raise ValueError("Unexpected sample-submission schema")
    if sample["id"].duplicated().any():
        raise ValueError("Sample-submission IDs must be globally unique")
    parent_test = np.load(workspace / "models" / "v006" / "test_preds.npy")
    if parent_test.shape != (len(sample),) or not np.isfinite(parent_test).all():
        raise ValueError("Locked v006 test predictions do not align with the sample")
    parent_by_id = pd.Series(
        np.asarray(parent_test, dtype=np.float64), index=sample["id"].astype(str)
    )

    with (workspace / "models" / "v008" / "model.pkl").open("rb") as handle:
        formation_model = pickle.load(handle)
    requested = tuple(
        sorted(
            set(map(str, formation_model["donor_well_ids"]))
            | set(map(str, formation_model["incomplete_donor_well_ids"]))
        )
    )
    donors = V008.DonorTable(
        well_ids=tuple(map(str, formation_model["donor_well_ids"])),
        coordinates=np.asarray(formation_model["donor_coordinates"], dtype=np.float64),
        formations=np.asarray(formation_model["donor_formations"], dtype=np.float64),
        requested_well_ids=requested,
        incomplete_well_ids=tuple(map(str, formation_model["incomplete_donor_well_ids"])),
    )
    hmm = _load_source_module(
        "rogii_hmm_smoother_v007_locked",
        Path(__file__).with_name("rogii_hmm_smoother.py"),
    )
    test_paths = V008._validate_test_schemas(workspace / "data" / "raw" / "test")
    frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    diagnostics: list[dict] = []
    for path in test_paths:
        well = V008.well_id_from_path(path)
        try:
            horizontal = pd.read_csv(path, usecols=list(V008.QUERY_COLUMNS))
            visible, suffix = V008._prefix_suffix_indices(horizontal)
            del visible
            ids = pd.Index([f"{well}_{row}" for row in suffix])
            parent_values = parent_by_id.reindex(ids).to_numpy(dtype=np.float64)
            if not np.isfinite(parent_values).all():
                raise ValueError("Locked v006 test stream does not cover the well suffix")
            typewell_path = path.with_name(f"{well}{hmm.TYPEWELL_SUFFIX}")
            if not typewell_path.is_file():
                raise FileNotFoundError(typewell_path)
            hmm_frame, hmm_diagnostics = hmm.hmm_candidates(
                horizontal.loc[:, ["MD", "Z", "GR", "TVT_input"]],
                pd.read_csv(
                    typewell_path,
                    usecols=lambda column: column in {"TVT", "GR"},
                ),
                parent_tvt=parent_values,
                configs=hmm.HMM_CANDIDATES,
            )
            if not np.array_equal(hmm_frame["_row_index"].to_numpy(dtype=np.int64), suffix):
                raise ValueError("v007 test stream row order differs from the sample IDs")
            slow_diagnostics = hmm_diagnostics["candidates"]["hmm_wr21_slow"]
            if (
                slow_diagnostics["unresolved_position_boundary"]
                or slow_diagnostics["unresolved_rate_boundary"]
            ):
                raise RuntimeError("v007 slow test stream has an unresolved boundary")
            spatial = V008.spatial_streams(horizontal, donors, query_well_id=well)
            if spatial.diagnostics["plane"]["query_well_present_after_exclusion"]:
                raise RuntimeError("same-ID formation donor survived exclusion")
            if not np.array_equal(spatial.frame["_row_index"].to_numpy(dtype=np.int64), suffix):
                raise ValueError("v008 test stream row order differs from the sample IDs")
            streams = np.column_stack(
                [
                    parent_values,
                    hmm_frame["hmm_wr21_slow"].to_numpy(dtype=np.float64),
                    spatial.frame["sp_plane_ancc_k10"].to_numpy(dtype=np.float64),
                    spatial.frame["sp_plane_best6_k10"].to_numpy(dtype=np.float64),
                ]
            )
            if streams.shape != (len(ids), 4) or not np.isfinite(streams).all():
                raise ValueError("Visible-test four-stream matrix is invalid")
            frames.append(
                pd.DataFrame(
                    {
                        "_id": ids.astype(str),
                        "prediction": streams @ weights,
                    }
                )
            )
            diagnostics.append(
                {
                    "well": well,
                    "rows": len(ids),
                    "hmm": hmm_diagnostics,
                    "formation": spatial.diagnostics,
                }
            )
        except Exception as exc:
            failures.append({"well": well, "error": f"{type(exc).__name__}: {exc}"})
    if failures:
        return None, failures, diagnostics
    predictions = pd.concat(frames, ignore_index=True)
    if predictions["_id"].isna().any() or predictions["_id"].duplicated().any():
        return None, [{"error": "visible-test IDs are missing or duplicated"}], diagnostics
    aligned = sample["id"].astype(str).map(predictions.set_index("_id")["prediction"])
    values = aligned.to_numpy(dtype=np.float64)
    if aligned.isna().any() or not np.isfinite(values).all():
        return None, [{"error": "visible-test streams do not cover sample IDs"}], diagnostics
    return values, [], diagnostics


def maybe_predict_visible_test(
    workspace: Path, decision: str, final_weights: Sequence[float]
) -> tuple[str, np.ndarray | None, list[dict], list[dict], str]:
    """Enter target-free test inference only after the local candidate gate passes."""
    if decision != "candidate_ready":
        return decision, None, [], [], "not_eligible"
    values, failures, diagnostics = predict_visible_test(workspace, final_weights)
    if failures or values is None:
        return "candidate_blocked_test_failures", None, failures, diagnostics, "blocked"
    return decision, values, [], diagnostics, "completed"


def _runtime_environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _fit_payload(fit: SimplexFit) -> dict:
    return {
        "weights": dict(zip(STREAM_COLUMNS, fit.weights.tolist())),
        "rows": fit.rows,
        "squared_error": fit.squared_error,
        "selected_face": fit.selected_face,
        "feasible_candidates": fit.feasible_candidates,
        "attempted_systems": fit.attempted_systems,
        "centered_delta_gram": fit.delta_gram.tolist(),
        "centered_delta_cross": fit.delta_cross.tolist(),
        "centered_residual_squared": fit.residual_squared,
        "direct_sse_verified": fit.direct_sse_verified,
        "support_audit": list(fit.support_audit),
    }


def _write_summary(
    path: Path,
    *,
    overall: float,
    fold_scores: Sequence[float],
    folds_better: int,
    decision: str,
    final_weights: np.ndarray,
    visible_test_status: str,
) -> None:
    text = "\n".join(
        [
            "# ROGII v009 Cross-Fitted Residual Simplex",
            "",
            f"Status: `{decision}`.",
            "",
            "## Canonical Nested CV",
            "",
            f"- Pooled OOF RMSE: `{overall:.9f}`",
            f"- Canonical folds better than v006: `{folds_better} / 5`",
            f"- Fold RMSE: `{', '.join(f'{score:.6f}' for score in fold_scores)}`",
            "- Final all-OOF weights: "
            + ", ".join(
                f"`{stream}={weight:.9f}`" for stream, weight in zip(STREAM_COLUMNS, final_weights)
            ),
            f"- Visible-test status: `{visible_test_status}`",
            "",
            "Each outer fit uses canonical parent/HMM streams and formation streams whose donor "
            "universe excludes both the outer fold and each meta row's fold. Held-out evaluation "
            "uses only canonical formation columns.",
            "",
            "No submission file, notebook, kernel, API request, or competition submission is "
            "created by this experiment.",
            "",
        ]
    )
    V008._atomic_write_text(path, text)


def train(workspace: Path, *, version: str = VERSION, folds: int = 5) -> Path:
    """Run the immutable v009 protocol; callers must explicitly invoke this command."""
    if version != VERSION or folds != 5:
        raise ValueError("The fixed residual-simplex experiment is exactly v009 with five folds")
    workspace = Path(workspace).resolve()
    project_root = workspace.parents[1]
    fold_path, manifest = V008.ensure_fold_manifest(workspace, folds)
    input_hashes_start = critical_input_hashes(workspace, fold_path)
    raw_test_fingerprint_start = V008.data_fingerprint(workspace / "data" / "raw" / "test")

    model_dir = workspace / "models" / version
    staging_dir = workspace / "models" / f".{version}.partial"
    for path in (model_dir, staging_dir):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    reports = {
        "summary": workspace / "reports" / "v009_residual_simplex_summary.md",
        "folds": workspace / "reports" / "v009_residual_simplex_folds.csv",
        "by_well": workspace / "reports" / "v009_residual_simplex_by_well.csv",
        "diagnostics": workspace / "reports" / "v009_residual_simplex_diagnostics.json",
    }
    for path in reports.values():
        for candidate in (path, Path(f"{path}.partial")):
            if candidate.exists():
                raise FileExistsError(f"Refusing to overwrite {candidate}")
    staging_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    run_common = {
        "version": version,
        "status": "running",
        "parent_runs": [PARENT_VERSION, HMM_VERSION, FORMATION_VERSION],
        "template": "rogii-cross-fitted-four-stream-residual-simplex",
        "command": " ".join(sys.argv),
        "reproduction_command": (
            "uv run --with-requirements "
            "workspaces/rogii-wellbore-geology-prediction/requirements-beam.txt python "
            "workspaces/rogii-wellbore-geology-prediction/scripts/rogii_residual_simplex.py "
            f"--workspace {workspace} train --version v009 --folds 5"
        ),
        "git_commit": V008.git_commit(project_root),
        "git_dirty_paths": V008.git_dirty_paths(project_root),
        "source_sha256": input_hashes_start["source_v009"],
        "plan_sha256": input_hashes_start["plan_v009"],
        "fixed_settings": _fixed_settings(),
        "random_seed": None,
        "cv_splitter": "canonical row-balanced GroupKFold by well; nested formation meta streams",
        "fold_manifest": str(fold_path.relative_to(workspace)),
        "fold_manifest_sha256": input_hashes_start["fold_manifest"],
        "metric": "rmse",
        "metric_direction": "minimize",
        "target_column": "TVT",
        "id_column": "id",
        "input_sha256": input_hashes_start,
        "raw_test_fingerprint": raw_test_fingerprint_start,
        "raw_test_fingerprint_algorithm": V008.DATA_FINGERPRINT_ALGORITHM,
        "environment": _runtime_environment(),
        "decision": "running",
        "test_predictions_saved": False,
        "submission_path": None,
        "submitted": False,
    }
    run_path = staging_dir / "run.json"
    V008._atomic_write_json(
        run_path,
        {**run_common, "runtime_seconds": None, "error": None},
    )
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        joined, join_diagnostics = join_locked_features(
            workspace / "data" / "features" / "v004.parquet",
            workspace / "data" / "features" / "v005.parquet",
        )
        if len(joined) != 3_783_989 or joined["_well_id"].nunique() != 773:
            raise ValueError("v009 joined rows/wells differ from the fixed competition contract")
        if sorted(np.unique(_integer_fold_values(joined["fold"])).tolist()) != [1, 2, 3, 4, 5]:
            raise ValueError("v009 joined rows do not contain canonical folds 1..5")
        parent_diagnostics = validate_parent_alignment(workspace, joined, manifest)
        hmm_diagnostics, boundary_wells = validate_hmm_alignment(workspace, joined)

        selections, final_fit, oof_prediction = nested_four_stream_stack(joined)
        target = joined["_target"].to_numpy(dtype=np.float64)
        row_folds = joined["fold"].to_numpy(dtype=int)
        fold_scores = [
            rmse(target[row_folds == fold], oof_prediction[row_folds == fold])
            for fold in range(1, 6)
        ]
        overall = rmse(target, oof_prediction)
        parent_scores = V008.baseline_scores_on_manifest(workspace, manifest)
        folds_better = sum(
            score < baseline for score, baseline in zip(fold_scores, parent_scores["fold_scores"])
        )
        direct_sse_violations = sum(
            int(not selection["direct_sse_verified"]) for selection in selections
        ) + int(not final_fit.direct_sse_verified)
        support_audit_violations = sum(
            int(selection["attempted_systems"] != EXPECTED_ACTIVE_SYSTEMS)
            for selection in selections
        ) + int(final_fit.attempted_systems != EXPECTED_ACTIVE_SYSTEMS)
        decision = candidate_decision(
            overall,
            folds_better,
            alignment_violations=int(join_diagnostics["alignment_violations"])
            + int(parent_diagnostics["violations"])
            + int(hmm_diagnostics["violations"]),
            donor_provenance_violations=support_audit_violations,
            direct_sse_violations=direct_sse_violations,
        )

        (
            decision,
            test_values,
            visible_test_failures,
            visible_test_diagnostics,
            visible_test_status,
        ) = maybe_predict_visible_test(workspace, decision, final_fit.weights)

        oof_rows = joined.loc[:, ["_id", "_well_id", "_row_index", "_target", "fold"]].copy()
        scored = oof_rows.assign(
            prediction=oof_prediction,
            squared_error=np.square(target - oof_prediction),
        )
        by_well = (
            scored.groupby(["_well_id", "fold"], sort=True)
            .agg(rows=("squared_error", "size"), squared_error=("squared_error", "sum"))
            .reset_index()
            .rename(columns={"_well_id": "well"})
        )
        by_well["rmse"] = np.sqrt(by_well["squared_error"] / by_well["rows"])
        fold_records = []
        for selection, score, baseline in zip(
            selections, fold_scores, parent_scores["fold_scores"]
        ):
            fold_records.append(
                {
                    "fold": selection["fold"],
                    "validation_rmse": score,
                    "parent_v006_rmse": baseline,
                    "improved": score < baseline,
                    **{
                        f"weight_{stream}": selection["weights"][stream]
                        for stream in STREAM_COLUMNS
                    },
                    "selected_face": selection["selected_face"],
                    "feasible_candidates": selection["feasible_candidates"],
                    "attempted_systems": selection["attempted_systems"],
                    "direct_sse_verified": selection["direct_sse_verified"],
                }
            )
        folds_frame = pd.DataFrame(fold_records)
        diagnostics_payload = {
            "version": version,
            "source_sha256": input_hashes_start["source_v009"],
            "plan_sha256": input_hashes_start["plan_v009"],
            "fixed_settings": _fixed_settings(),
            "join": join_diagnostics,
            "parent_alignment": parent_diagnostics,
            "hmm_alignment": hmm_diagnostics,
            "inherited_v007_boundary_well_count": len(boundary_wells),
            "inherited_v007_boundary_wells": boundary_wells,
            "fold_fits": selections,
            "final_fit": _fit_payload(final_fit),
            "visible_test_status": visible_test_status,
            "visible_test_failures": visible_test_failures,
            "visible_test_diagnostics": visible_test_diagnostics,
        }
        V008._atomic_write_csv(reports["folds"], folds_frame)
        V008._atomic_write_csv(reports["by_well"], by_well)
        V008._atomic_write_json(reports["diagnostics"], diagnostics_payload)
        _write_summary(
            reports["summary"],
            overall=overall,
            fold_scores=fold_scores,
            folds_better=folds_better,
            decision=decision,
            final_weights=final_fit.weights,
            visible_test_status=visible_test_status,
        )

        cv_scores = {
            "metric": "rmse",
            "direction": "minimize",
            "fold_scores": fold_scores,
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
            "overall": overall,
            "valid_rows": len(joined),
            "valid_wells": int(joined["_well_id"].nunique()),
            "selection_protocol": (
                "outer k fits canonical parent/HMM plus nested_outer_k formation on folds j!=k"
            ),
            "fold_selections": selections,
            "final_fit": _fit_payload(final_fit),
            "canonical_v006": parent_scores,
            "folds_better_than_v006": folds_better,
            "candidate_gate": CANDIDATE_GATE,
            "promising_gate": PROMISING_GATE,
            "decision": decision,
        }
        model_payload = {
            "model": "cross_fitted_four_stream_residual_simplex",
            "streams": STREAM_COLUMNS,
            "fold_selections": selections,
            "final_fit": _fit_payload(final_fit),
            "fixed_settings": _fixed_settings(),
            "inherited_v007_boundary_wells": boundary_wells,
        }
        importance_rows = []
        for selection in selections:
            for stream in STREAM_COLUMNS:
                importance_rows.append(
                    {
                        "scope": "outer_fold",
                        "fold": selection["fold"],
                        "feature": stream,
                        "weight": selection["weights"][stream],
                    }
                )
        for stream, weight in zip(STREAM_COLUMNS, final_fit.weights):
            importance_rows.append(
                {"scope": "final_all_oof", "fold": 0, "feature": stream, "weight": weight}
            )
        importance = pd.DataFrame(importance_rows)
        feature_names = [
            *STREAM_COLUMNS,
            *(
                nested_stream_column(outer, stream)
                for outer in range(1, 6)
                for stream in ("sp_plane_ancc_k10", "sp_plane_best6_k10")
            ),
        ]
        V008._atomic_write_json(staging_dir / "cv_scores.json", cv_scores)
        V008._atomic_write_npy(staging_dir / "oof_preds.npy", oof_prediction)
        V008._atomic_write_parquet(staging_dir / "oof_rows.parquet", oof_rows)
        V008._atomic_write_pickle(staging_dir / "model.pkl", model_payload)
        V008._atomic_write_text(staging_dir / "feature_list.txt", "\n".join(feature_names) + "\n")
        V008._atomic_write_csv(staging_dir / "importance.csv", importance)
        if test_values is not None:
            V008._atomic_write_npy(staging_dir / "test_preds.npy", test_values)

        input_hashes_end = critical_input_hashes(workspace, fold_path)
        raw_test_fingerprint_end = V008.data_fingerprint(workspace / "data" / "raw" / "test")
        if input_hashes_end != input_hashes_start:
            raise ValueError("Critical v009 inputs changed during the run")
        if raw_test_fingerprint_end != raw_test_fingerprint_start:
            raise ValueError("Raw visible-test tree changed during the run")

        output_files = {
            **{f"report_{name}": path for name, path in reports.items()},
            "cv_scores": staging_dir / "cv_scores.json",
            "oof_predictions": staging_dir / "oof_preds.npy",
            "oof_rows": staging_dir / "oof_rows.parquet",
            "model": staging_dir / "model.pkl",
            "feature_list": staging_dir / "feature_list.txt",
            "importance": staging_dir / "importance.csv",
        }
        if test_values is not None:
            output_files["test_predictions"] = staging_dir / "test_preds.npy"
        output_sha256 = {name: sha256_file(path) for name, path in output_files.items()}
        run_payload = {
            **run_common,
            "status": "completed",
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "error": None,
            "input_sha256_end": input_hashes_end,
            "raw_test_fingerprint_end": raw_test_fingerprint_end,
            "output_sha256": output_sha256,
            "decision": decision,
            "cv_score": overall,
            "folds_better_than_v006": folds_better,
            "final_weights": dict(zip(STREAM_COLUMNS, final_fit.weights.tolist())),
            "inherited_v007_boundary_well_count": len(boundary_wells),
            "inherited_v007_boundary_wells": boundary_wells,
            "alignment_violations": 0,
            "fold_violations": 0,
            "donor_provenance_violations": support_audit_violations,
            "nonfinite_violations": 0,
            "direct_sse_violations": direct_sse_violations,
            "visible_test_status": visible_test_status,
            "test_failures": len(visible_test_failures),
            "test_predictions_saved": test_values is not None,
        }
        V008._atomic_replace_json(run_path, run_payload)
        os.replace(staging_dir, model_dir)
        return model_dir
    except BaseException as exc:
        failure_payload = {
            **run_common,
            "status": "failed",
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "decision": "failed",
            "test_predictions_saved": False,
        }
        if run_path.is_file():
            try:
                V008._atomic_replace_json(run_path, failure_payload)
            except Exception:
                pass
        if staging_dir.exists() and not model_dir.exists():
            os.replace(staging_dir, model_dir)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Run immutable v009 local evaluation")
    train_parser.add_argument("--version", default=VERSION)
    train_parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "train":
        path = train(args.workspace, version=args.version, folds=args.folds)
        print(path)
        return 0
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
