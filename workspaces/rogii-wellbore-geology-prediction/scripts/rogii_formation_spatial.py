#!/usr/bin/env python3
"""Fold-safe formation-surface spatial streams and nested simplex stack.

The module deliberately has no submission or Kaggle API path.  Its smoke command is
target-free; the train command refuses to read suffix targets until a matching smoke
artifact has passed the fixed runtime, memory, source, and raw-data gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import resource
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised by runtime preflight
    pa = None
    pq = None


HORIZONTAL_SUFFIX = "__horizontal_well.csv"
PARENT_VERSION = "v006"
PARENT_CANDIDATE = "pf_scale_12_hold_0p2"
FORMATIONS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
QUERY_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TRAIN_COLUMNS = (
    "MD",
    "X",
    "Y",
    "Z",
    *FORMATIONS,
    "TVT",
    "GR",
    "TVT_input",
)
STREAM_COLUMNS = (
    "parent_v006",
    "sp_plane_ancc_k10",
    "sp_plane_best6_k10",
)
K_NEIGHBORS = 10
COORDINATE_STD_DDOF = 0
DISTANCE_EPSILON = 1e-6
SLOPE_RIDGE = 1e-8
QUERY_CHUNK_ROWS = 4096
PROSPECTIVE_CALIBRATION_START = -640
PROSPECTIVE_CALIBRATION_STOP = -128
PROSPECTIVE_HOLDOUT_ROWS = 128
FINAL_CALIBRATION_ROWS = 512
MINIMUM_VISIBLE_PREFIX_ROWS = 851
SMOKE_WELLS = 6
MAX_PROJECTED_HOURS = 1.0
MAX_PEAK_RSS_GIB = 4.0
PROJECTION_SAFETY_FACTOR = 2.0
SMOKE_QUERY_ROWS = 168_705
FULL_NESTED_QUERY_ROWS = 21_393_545
SMOKE_SELECTION = (
    ("fba7683c", 2, 407),
    ("fef8af96", 3, 3826),
    ("8bb9c1e6", 1, 4546),
    ("722cf0d8", 3, 5143),
    ("ff8bb73a", 3, 5927),
    ("ea3a0e38", 1, 10052),
)
CANDIDATE_GATE = 10.95
PROMISING_GATE = 11.75
DATA_FINGERPRINT_ALGORITHM = "sha256(sorted_file_name_nul_file_bytes_nul)"


def _fixed_settings() -> dict:
    """Return every predeclared numerical choice that can affect predictions."""
    return {
        "formations": list(FORMATIONS),
        "streams": list(STREAM_COLUMNS),
        "donor_summary": "per-well finite column median",
        "complete_donor_rule": "finite median X,Y and all six formations",
        "coordinate_center": "eligible donor arithmetic mean after same-ID exclusion",
        "coordinate_scale": "eligible donor population std",
        "coordinate_std_ddof": COORDINATE_STD_DDOF,
        "distance": "Euclidean in donor-standardized X/Y",
        "neighbors": K_NEIGHBORS,
        "weight_formula": "raw=1/max(distance,1e-6), normalized per query",
        "distance_epsilon": DISTANCE_EPSILON,
        "design_columns": ["(X_neighbor-X_query)/X_std", "(Y_neighbor-Y_query)/Y_std", "1"],
        "slope_ridge": SLOPE_RIDGE,
        "ridge_penalty": "two slope diagonal entries only; intercept unpenalized",
        "zero_distance": "distance clamped to epsilon; no direct-value fallback",
        "linear_solver": "numpy.linalg.solve; no pseudoinverse or constant fallback",
        "query_chunk_rows": QUERY_CHUNK_ROWS,
        "ancc_calibration": "last 512 visible-prefix rows",
        "best6_prospective_calibration": "visible-prefix rows [-640:-128]",
        "best6_prospective_holdout": "last 128 visible-prefix rows",
        "best6_final_calibration": "last 512 visible-prefix rows",
        "formation_tie_break": "declared FORMATIONS order",
        "simplex": "p0=v006; enumerate exact 2D triangle interior, edges, vertices",
        "simplex_intercept": False,
        "simplex_row_sampling": False,
        "parent_oof_alignment": "exact ID, well, row index, canonical fold, and suffix target",
        "parent_test_alignment": "locked v006 test_preds mapped by sample id",
        "minimum_visible_prefix_rows": MINIMUM_VISIBLE_PREFIX_ROWS,
        "smoke_query_rows": SMOKE_QUERY_ROWS,
        "full_nested_query_rows": FULL_NESTED_QUERY_ROWS,
        "projection_safety_factor": PROJECTION_SAFETY_FACTOR,
    }


@dataclass(frozen=True)
class DonorTable:
    """Raw donor centroids; scaling is intentionally deferred until self exclusion."""

    well_ids: tuple[str, ...]
    coordinates: np.ndarray
    formations: np.ndarray
    requested_well_ids: tuple[str, ...] = ()
    incomplete_well_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        formations = np.asarray(self.formations, dtype=np.float64)
        if coordinates.shape != (len(self.well_ids), 2):
            raise ValueError("Donor coordinates must have shape (n, 2)")
        if formations.shape != (len(self.well_ids), len(FORMATIONS)):
            raise ValueError("Donor formations must have shape (n, 6)")
        if len(set(self.well_ids)) != len(self.well_ids):
            raise ValueError("Donor well IDs must be unique")
        if not np.isfinite(coordinates).all() or not np.isfinite(formations).all():
            raise ValueError("Eligible donor values must all be finite")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "formations", formations)

    @property
    def xy(self) -> np.ndarray:
        """Compatibility alias with the domain name used in the v008 plan."""
        return self.coordinates


@dataclass(frozen=True)
class SpatialPrediction:
    frame: pd.DataFrame
    diagnostics: dict

    @property
    def values(self) -> np.ndarray:
        return self.frame.loc[:, ["sp_plane_ancc_k10", "sp_plane_best6_k10"]].to_numpy(
            dtype=np.float64
        )


@dataclass(frozen=True)
class SimplexFit:
    weights: np.ndarray
    squared_error: float
    rows: int
    delta_gram: np.ndarray
    delta_cross: np.ndarray
    residual_squared: float


class TerminationRequested(RuntimeError):
    """Raised in the main thread so SIGTERM can publish a failed run artifact."""


def _raise_on_sigterm(signum, frame) -> None:
    del frame
    raise TerminationRequested(f"received signal {signum}")


def well_id_from_path(path: Path) -> str:
    if not path.name.endswith(HORIZONTAL_SUFFIX):
        raise ValueError(f"Unexpected horizontal-well filename: {path.name}")
    return path.name[: -len(HORIZONTAL_SUFFIX)]


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("RMSE arrays must be non-empty and shape-aligned")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("RMSE arrays must be finite")
    return float(np.sqrt(np.mean(np.square(target - prediction))))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(Path(root).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {root}")
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def git_dirty_paths(project_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        [line.rstrip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode == 0
        else []
    )


def validate_horizontal_schema(
    frame: pd.DataFrame, *, role: str, require_target: bool = False
) -> None:
    """Fail closed on unexpected train/test columns, including duplicate names."""
    if role not in {"train", "test", "query"}:
        raise ValueError(f"Unknown horizontal schema role: {role}")
    expected = TRAIN_COLUMNS if role == "train" or require_target else QUERY_COLUMNS
    actual = tuple(str(column) for column in frame.columns)
    if len(actual) != len(set(actual)):
        raise ValueError(f"Duplicate columns in {role} horizontal schema")
    if set(actual) != set(expected) or len(actual) != len(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"Unexpected {role} schema; missing={missing}, extra={extra}")


def _validate_csv_schema(path: Path, *, role: str) -> None:
    validate_horizontal_schema(pd.read_csv(path, nrows=0), role=role)


def _horizontal_paths(root: Path) -> list[Path]:
    paths = sorted(Path(root).glob(f"*{HORIZONTAL_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(f"No horizontal-well CSV files found under {root}")
    ids = [well_id_from_path(path) for path in paths]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate horizontal well IDs under {root}")
    return paths


def complete_six_donor_row(path: Path) -> dict | None:
    """Read no TVT target and return one eligible donor centroid, or None."""
    _validate_csv_schema(path, role="train")
    frame = pd.read_csv(path, usecols=["X", "Y", *FORMATIONS])
    medians = []
    for column in ("X", "Y", *FORMATIONS):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        medians.append(float(np.median(finite)) if finite.size else np.nan)
    if not np.isfinite(medians).all():
        return None
    return {"well": well_id_from_path(path), **dict(zip(("X", "Y", *FORMATIONS), medians))}


def build_donor_catalog(train_root: Path, expected_well_ids: Iterable[str]) -> DonorTable:
    expected = tuple(sorted(str(well) for well in expected_well_ids))
    if len(expected) != len(set(expected)):
        raise ValueError("Expected donor well IDs must be unique")
    paths = _horizontal_paths(train_root)
    path_map = {well_id_from_path(path): path for path in paths}
    if set(path_map) != set(expected):
        raise ValueError("Canonical manifest wells do not exactly match training files")
    rows: list[dict] = []
    incomplete: list[str] = []
    for well in expected:
        row = complete_six_donor_row(path_map[well])
        if row is None:
            incomplete.append(well)
        else:
            rows.append(row)
    if len(rows) < K_NEIGHBORS:
        raise ValueError("Fewer than k complete-six formation donors")
    return DonorTable(
        well_ids=tuple(str(row["well"]) for row in rows),
        coordinates=np.asarray([[row["X"], row["Y"]] for row in rows], dtype=np.float64),
        formations=np.asarray(
            [[row[name] for name in FORMATIONS] for row in rows], dtype=np.float64
        ),
        requested_well_ids=expected,
        incomplete_well_ids=tuple(incomplete),
    )


def subset_donor_table(catalog: DonorTable, donor_well_ids: Iterable[str]) -> DonorTable:
    requested = tuple(sorted(set(str(well) for well in donor_well_ids)))
    catalog_requested = set(catalog.requested_well_ids or catalog.well_ids)
    if not set(requested).issubset(catalog_requested):
        raise ValueError("Requested donors are outside the canonical catalog")
    index = {well: i for i, well in enumerate(catalog.well_ids)}
    eligible = tuple(well for well in requested if well in index)
    if len(eligible) < K_NEIGHBORS:
        raise ValueError("Fewer than k eligible donors after fold exclusion")
    indices = np.asarray([index[well] for well in eligible], dtype=np.int64)
    return DonorTable(
        well_ids=eligible,
        coordinates=catalog.coordinates[indices],
        formations=catalog.formations[indices],
        requested_well_ids=requested,
        incomplete_well_ids=tuple(well for well in requested if well not in index),
    )


def build_donor_table(train_root: Path, donor_well_ids: Iterable[str]) -> DonorTable:
    """Convenience API for tests and small tools; full runs build the catalog once."""
    requested = tuple(sorted(set(str(well) for well in donor_well_ids)))
    paths = {well_id_from_path(path): path for path in _horizontal_paths(train_root)}
    if not set(requested).issubset(paths):
        raise ValueError("Requested donor files are missing")
    rows = [complete_six_donor_row(paths[well]) for well in requested]
    eligible = [row for row in rows if row is not None]
    if len(eligible) < K_NEIGHBORS:
        raise ValueError("Fewer than k complete-six formation donors")
    eligible_ids = tuple(str(row["well"]) for row in eligible)
    return DonorTable(
        well_ids=eligible_ids,
        coordinates=np.asarray([[row["X"], row["Y"]] for row in eligible], dtype=np.float64),
        formations=np.asarray(
            [[row[name] for name in FORMATIONS] for row in eligible], dtype=np.float64
        ),
        requested_well_ids=requested,
        incomplete_well_ids=tuple(well for well in requested if well not in eligible_ids),
    )


def query_local_planes(
    donors: DonorTable, query_xy: np.ndarray, *, query_well_id: str
) -> tuple[np.ndarray, dict]:
    """Predict all six surfaces with the fixed centered k=10 local plane."""
    query = np.asarray(query_xy, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 2 or query.shape[0] == 0:
        raise ValueError("Query coordinates must have shape (n, 2) with n > 0")
    if not np.isfinite(query).all():
        raise ValueError("Query coordinates contain non-finite values")

    # Same-ID exclusion happens before scaler construction and neighbour search.
    keep = np.asarray([well != str(query_well_id) for well in donors.well_ids], dtype=bool)
    self_excluded = int((~keep).sum())
    donor_ids = tuple(well for well, include in zip(donors.well_ids, keep) if include)
    coordinates = donors.coordinates[keep]
    surfaces = donors.formations[keep]
    if len(coordinates) < K_NEIGHBORS:
        raise ValueError("Fewer than k donors after same-well exclusion")
    mean = np.mean(coordinates, axis=0, dtype=np.float64)
    scale = np.std(coordinates, axis=0, ddof=COORDINATE_STD_DDOF, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("Donor coordinate scaler is degenerate")
    normalized_donors = (coordinates - mean) / scale
    normalized_query = (query - mean) / scale

    output = np.empty((len(query), len(FORMATIONS)), dtype=np.float64)
    max_neighbor_distance = 0.0
    ridge = np.diag([SLOPE_RIDGE, SLOPE_RIDGE, 0.0])
    for start in range(0, len(query), QUERY_CHUNK_ROWS):
        stop = min(start + QUERY_CHUNK_ROWS, len(query))
        q = normalized_query[start:stop]
        delta = normalized_donors[None, :, :] - q[:, None, :]
        distance_squared = np.einsum("qdi,qdi->qd", delta, delta, optimize=True)
        nearest = np.argpartition(distance_squared, K_NEIGHBORS - 1, axis=1)[:, :K_NEIGHBORS]
        nearest_distance_squared = np.take_along_axis(distance_squared, nearest, axis=1)
        order = np.argsort(nearest_distance_squared, axis=1, kind="stable")
        nearest = np.take_along_axis(nearest, order, axis=1)
        distances = np.sqrt(np.take_along_axis(distance_squared, nearest, axis=1))
        max_neighbor_distance = max(max_neighbor_distance, float(np.max(distances)))
        weights = 1.0 / np.maximum(distances, DISTANCE_EPSILON)
        weights /= np.sum(weights, axis=1, keepdims=True)
        neighbour_coords = normalized_donors[nearest]
        centred = neighbour_coords - q[:, None, :]
        design = np.concatenate(
            [centred, np.ones((len(q), K_NEIGHBORS, 1), dtype=np.float64)], axis=2
        )
        neighbour_surfaces = surfaces[nearest]
        normal = np.einsum("qki,qk,qkj->qij", design, weights, design, optimize=True)
        normal += ridge[None, :, :]
        rhs = np.einsum("qki,qk,qkf->qif", design, weights, neighbour_surfaces, optimize=True)
        try:
            coefficients = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Centered local plane could not be solved") from exc
        prediction = coefficients[:, 2, :]
        if not np.isfinite(prediction).all():
            raise ValueError("Centered local plane produced non-finite predictions")
        output[start:stop] = prediction
    return output, {
        "requested_donor_count": len(donors.requested_well_ids or donors.well_ids),
        "eligible_donor_count_before_self_exclusion": len(donors.well_ids),
        "eligible_donor_count": len(donor_ids),
        "same_id_donors_excluded": self_excluded,
        "query_well_present_after_exclusion": str(query_well_id) in donor_ids,
        "coordinate_mean": mean.tolist(),
        "coordinate_std_ddof0": scale.tolist(),
        "max_kth_neighbor_distance": max_neighbor_distance,
    }


def _prefix_suffix_indices(horizontal: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    input_values = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if np.isinf(input_values).any():
        raise ValueError("TVT_input contains infinity")
    visible = np.flatnonzero(np.isfinite(input_values))
    suffix = np.flatnonzero(np.isnan(input_values))
    if len(visible) < MINIMUM_VISIBLE_PREFIX_ROWS:
        raise ValueError("Visible TVT_input prefix is shorter than the fixed window contract")
    if len(suffix) == 0:
        raise ValueError("Horizontal well has no missing TVT_input suffix")
    if not np.array_equal(visible, np.arange(len(visible))):
        raise ValueError("TVT_input visible rows must form one contiguous prefix")
    if not np.array_equal(suffix, np.arange(len(visible), len(horizontal))):
        raise ValueError("TVT_input missing rows must form one contiguous suffix")
    return visible, suffix


def spatial_streams(
    horizontal: pd.DataFrame, donors: DonorTable, *, query_well_id: str
) -> SpatialPrediction:
    """Create fixed ANCC and prospective best-of-six streams without reading TVT."""
    validate_horizontal_schema(horizontal, role="query")
    visible, suffix = _prefix_suffix_indices(horizontal)
    for column in ("MD", "X", "Y", "Z"):
        values = pd.to_numeric(horizontal[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Query column {column} contains non-finite values")
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(dtype=np.float64)
    if np.any(np.diff(md) <= 0.0):
        raise ValueError("MD must be strictly increasing in file order")
    gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(dtype=np.float64)
    if np.isinf(gr).any():
        raise ValueError("GR may be missing but may not contain infinity")

    # Only the fixed last-640 calibration/holdout prefix plus suffix are queried.
    # This is both the target-free contract and the exact unit used by the smoke projection.
    prefix_query = visible[-640:]
    query_indices = np.concatenate([prefix_query, suffix])
    xy = horizontal.iloc[query_indices][["X", "Y"]].to_numpy(dtype=np.float64)
    surfaces, plane_diagnostics = query_local_planes(donors, xy, query_well_id=str(query_well_id))
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(dtype=np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(dtype=np.float64)

    calibration = visible[PROSPECTIVE_CALIBRATION_START:PROSPECTIVE_CALIBRATION_STOP]
    holdout = visible[-PROSPECTIVE_HOLDOUT_ROWS:]
    if len(calibration) != 512 or len(holdout) != PROSPECTIVE_HOLDOUT_ROWS:
        raise ValueError("Prospective best-six windows do not match the fixed contract")
    local_calibration = np.arange(0, 512, dtype=np.int64)
    local_holdout = np.arange(512, 640, dtype=np.int64)
    local_final = np.arange(128, 640, dtype=np.int64)
    local_suffix = np.arange(640, len(query_indices), dtype=np.int64)
    final_window = visible[-FINAL_CALIBRATION_ROWS:]
    ancc_bias = float(
        np.median(tvt_input[final_window] + z[final_window] - surfaces[local_final, 0])
    )
    ancc_prediction = -z[suffix] + surfaces[local_suffix, 0] + ancc_bias

    prospective_rmse: list[float] = []
    prospective_bias: list[float] = []
    for formation_index in range(len(FORMATIONS)):
        bias = float(
            np.median(
                tvt_input[calibration]
                + z[calibration]
                - surfaces[local_calibration, formation_index]
            )
        )
        prediction = -z[holdout] + surfaces[local_holdout, formation_index] + bias
        prospective_bias.append(bias)
        prospective_rmse.append(rmse(tvt_input[holdout], prediction))
    selected_index = int(np.argmin(np.asarray(prospective_rmse, dtype=np.float64)))
    selected_name = FORMATIONS[selected_index]
    final_bias = float(
        np.median(tvt_input[final_window] + z[final_window] - surfaces[local_final, selected_index])
    )
    best_prediction = -z[suffix] + surfaces[local_suffix, selected_index] + final_bias
    values = np.column_stack([ancc_prediction, best_prediction])
    if not np.isfinite(values).all():
        raise ValueError("Spatial streams contain non-finite suffix predictions")
    frame = pd.DataFrame(
        {
            "_row_index": suffix.astype(np.int64),
            "sp_plane_ancc_k10": values[:, 0],
            "sp_plane_best6_k10": values[:, 1],
        }
    )
    return SpatialPrediction(
        frame=frame,
        diagnostics={
            "well": str(query_well_id),
            "visible_prefix_rows": len(visible),
            "suffix_rows": len(suffix),
            "ancc_bias": ancc_bias,
            "prospective_rmse": dict(zip(FORMATIONS, prospective_rmse)),
            "prospective_bias": dict(zip(FORMATIONS, prospective_bias)),
            "selected_formation": selected_name,
            "selected_formation_index": selected_index,
            "final_bias": final_bias,
            "plane": plane_diagnostics,
        },
    )


def _simplex_objective(
    delta_gram: np.ndarray,
    delta_cross: np.ndarray,
    residual_squared: float,
    point: tuple[float, float],
) -> float:
    weights = np.asarray(point, dtype=np.float64)
    return float(
        residual_squared - 2.0 * np.dot(delta_cross, weights) + weights @ delta_gram @ weights
    )


def simplex_nnls_from_gram(
    gram: np.ndarray, cross: np.ndarray, residual_squared: float = 0.0
) -> np.ndarray:
    """Solve the exact two-delta triangle problem; inputs are centered statistics.

    ``gram`` is D'D and ``cross`` is D'(y-p0), where D=[p1-p0,p2-p0].
    The returned order is [w0, w1, w2].  No unconstrained clipping or pseudoinverse
    is used; all three vertices, all three edges, and a solvable interior are enumerated.
    """
    matrix = np.asarray(gram, dtype=np.float64)
    vector = np.asarray(cross, dtype=np.float64)
    if matrix.shape != (2, 2) or vector.shape != (2,):
        raise ValueError("Centered simplex statistics must have shapes (2,2) and (2,)")
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise ValueError("Centered simplex statistics must be finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10):
        raise ValueError("Centered simplex Gram matrix must be symmetric")
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(matrix))))
    if float(np.min(np.linalg.eigvalsh(matrix))) < -tolerance:
        raise ValueError("Centered simplex Gram matrix is not positive semidefinite")

    candidates: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    if matrix[0, 0] > tolerance:
        candidates.append((float(np.clip(vector[0] / matrix[0, 0], 0.0, 1.0)), 0.0))
    if matrix[1, 1] > tolerance:
        candidates.append((0.0, float(np.clip(vector[1] / matrix[1, 1], 0.0, 1.0))))
    edge_curvature = float(matrix[0, 0] - 2.0 * matrix[0, 1] + matrix[1, 1])
    if edge_curvature > tolerance:
        edge_w1 = float(
            np.clip(
                (vector[0] - vector[1] - matrix[0, 1] + matrix[1, 1]) / edge_curvature,
                0.0,
                1.0,
            )
        )
        candidates.append((edge_w1, 1.0 - edge_w1))
    try:
        interior = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        interior = None
    if (
        interior is not None
        and np.isfinite(interior).all()
        and interior[0] >= 0.0
        and interior[1] >= 0.0
        and float(interior.sum()) <= 1.0
    ):
        candidates.append((float(interior[0]), float(interior[1])))

    ranked = sorted(
        (
            _simplex_objective(matrix, vector, float(residual_squared), point),
            point[0],
            point[1],
        )
        for point in candidates
    )
    _, w1, w2 = ranked[0]
    weights = np.asarray([1.0 - w1 - w2, w1, w2], dtype=np.float64)
    if np.any(weights < -1e-12) or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("Simplex enumeration returned an infeasible point")
    weights[np.abs(weights) < 1e-15] = 0.0
    return weights


def fit_simplex_weights(streams: np.ndarray, target: np.ndarray) -> SimplexFit:
    predictions = np.asarray(streams, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("Simplex streams must have shape (n, 3)")
    if truth.shape != (len(predictions),) or len(truth) == 0:
        raise ValueError("Simplex target must align with non-empty streams")
    if not np.isfinite(predictions).all() or not np.isfinite(truth).all():
        raise ValueError("Simplex inputs must be finite")
    delta = predictions[:, 1:] - predictions[:, [0]]
    residual = truth - predictions[:, 0]
    gram = delta.T @ delta
    cross = delta.T @ residual
    residual_squared = float(np.dot(residual, residual))
    weights = simplex_nnls_from_gram(gram, cross, residual_squared)
    # Direct row-level residual verification avoids cancellation in absolute TVT space.
    direct_residual = truth - predictions @ weights
    squared_error = float(np.dot(direct_residual, direct_residual))
    formula_error = _simplex_objective(gram, cross, residual_squared, (weights[1], weights[2]))
    tolerance = 1e-8 * max(1.0, squared_error)
    if not np.isfinite(squared_error) or abs(formula_error - squared_error) > tolerance:
        raise ValueError("Centered simplex statistics failed direct residual verification")
    return SimplexFit(
        weights=weights,
        squared_error=squared_error,
        rows=len(truth),
        delta_gram=gram,
        delta_cross=cross,
        residual_squared=residual_squared,
    )


def scored_row_counts(train_root: Path) -> pd.DataFrame:
    records = []
    for path in _horizontal_paths(train_root):
        frame = pd.read_csv(path, usecols=["TVT_input"])
        values = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise ValueError(f"TVT_input contains infinity for {well_id_from_path(path)}")
        records.append(
            {
                "well": well_id_from_path(path),
                "scored_rows": int(np.isnan(values).sum()),
            }
        )
    return pd.DataFrame(records).sort_values("well").reset_index(drop=True)


def assign_balanced_folds(counts: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    if n_splits < 2 or len(counts) < n_splits:
        raise ValueError("Balanced folds require at least two folds and one well per fold")
    ordered = counts[["well", "scored_rows"]].sort_values("well").reset_index(drop=True)
    weights = ordered["scored_rows"].to_numpy(dtype=np.int64)
    descending = np.argsort(weights, kind="stable")[::-1]
    fold_weights = np.zeros(n_splits, dtype=np.int64)
    assignments = np.zeros(len(ordered), dtype=np.int64)
    for group_index in descending:
        fold = int(np.argmin(fold_weights))
        fold_weights[fold] += weights[group_index]
        assignments[group_index] = fold + 1
    result = ordered.assign(fold=assignments)
    result["splitter"] = f"GroupKFold(n_splits={n_splits}, weight=scored_rows)"
    return result


def ensure_fold_manifest(workspace: Path, n_splits: int) -> tuple[Path, pd.DataFrame]:
    path = Path(workspace) / "reports" / "canonical_outer_folds_v001.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Canonical fold manifest is missing: {path}")
    expected = assign_balanced_folds(
        scored_row_counts(Path(workspace) / "data" / "raw" / "train"), n_splits
    )
    current = pd.read_csv(path)
    pd.testing.assert_frame_equal(current, expected, check_dtype=False)
    actual_folds = sorted(current["fold"].astype(int).unique().tolist())
    if actual_folds != list(range(1, n_splits + 1)):
        raise ValueError("Canonical fold labels are not contiguous one-based integers")
    return path, expected


def next_feature_path(workspace: Path) -> Path:
    root = Path(workspace) / "data" / "features"
    root.mkdir(parents=True, exist_ok=True)
    versions = []
    for path in root.glob("v[0-9][0-9][0-9].parquet"):
        try:
            versions.append(int(path.stem[1:]))
        except ValueError:
            continue
    return root / f"v{max(versions, default=0) + 1:03d}.parquet"


def nested_stream_column(outer_fold: int, stream: str) -> str:
    if stream not in {"sp_plane_ancc_k10", "sp_plane_best6_k10"}:
        raise ValueError(f"Unknown spatial stream: {stream}")
    if int(outer_fold) < 1:
        raise ValueError("Outer fold must be positive")
    return f"nested_outer_{int(outer_fold)}__{stream}"


class ParentOOFReader:
    """Strictly align the locked v006 OOF stream by ID, well, row, and fold."""

    def __init__(self, workspace: Path, manifest: pd.DataFrame):
        workspace = Path(workspace)
        self.rows_path = workspace / "models" / PARENT_VERSION / "oof_rows.parquet"
        self.oof_path = workspace / "models" / PARENT_VERSION / "oof_preds.npy"
        self.feature_path = workspace / "data" / "features" / "v003.parquet"
        scores_path = workspace / "models" / PARENT_VERSION / "cv_scores.json"
        for path in (self.rows_path, self.oof_path, self.feature_path, scores_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        selected = {str(item["selected_candidate"]) for item in scores["fold_selections"]}
        if selected != {PARENT_CANDIDATE} or scores.get("final_candidate") != PARENT_CANDIDATE:
            raise ValueError("v006 parent no longer uses the locked prediction stream")
        self.rows = pd.read_parquet(
            self.rows_path, columns=["_id", "_well_id", "_row_index", "_target", "fold"]
        )
        self.oof = np.load(self.oof_path, mmap_mode="r")
        if len(self.rows) != len(self.oof):
            raise ValueError("v006 OOF rows and predictions differ in length")
        if self.rows["_id"].duplicated().any():
            raise ValueError("v006 OOF IDs are duplicated")
        fold_map = manifest.set_index("well")["fold"].astype(int)
        mapped = self.rows["_well_id"].map(fold_map)
        if mapped.isna().any() or not np.array_equal(
            mapped.to_numpy(dtype=np.int64), self.rows["fold"].to_numpy(dtype=np.int64)
        ):
            raise ValueError("v006 OOF folds do not match the canonical manifest")
        self.cursor = 0

    def read_well(
        self,
        well: str,
        evaluation_indices: np.ndarray,
        expected_fold: int,
        expected_target: np.ndarray,
    ) -> np.ndarray:
        row_indices = np.asarray(evaluation_indices, dtype=np.int64)
        stop = self.cursor + len(row_indices)
        if stop > len(self.rows):
            raise ValueError("v006 parent ended before raw suffix rows")
        block = self.rows.iloc[self.cursor : stop]
        expected_ids = np.asarray([f"{well}_{row}" for row in row_indices], dtype=object)
        if not np.array_equal(block["_id"].astype(str).to_numpy(), expected_ids):
            raise ValueError(f"v006 OOF ID alignment failed for well {well}")
        if set(block["_well_id"].astype(str)) != {str(well)}:
            raise ValueError(f"v006 OOF well alignment failed for well {well}")
        if not np.array_equal(block["_row_index"].to_numpy(dtype=np.int64), row_indices):
            raise ValueError(f"v006 OOF row alignment failed for well {well}")
        if set(block["fold"].astype(int)) != {int(expected_fold)}:
            raise ValueError(f"v006 OOF fold alignment failed for well {well}")
        target = np.asarray(expected_target, dtype=np.float64)
        parent_target = block["_target"].to_numpy(dtype=np.float64)
        if target.shape != (len(row_indices),) or not np.isfinite(target).all():
            raise ValueError(f"Current suffix target is invalid for well {well}")
        if not np.array_equal(parent_target, target):
            raise ValueError(f"v006 OOF target alignment failed for well {well}")
        values = np.asarray(self.oof[self.cursor : stop], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"v006 OOF contains non-finite values for well {well}")
        self.cursor = stop
        return values

    def finish(self) -> None:
        if self.cursor != len(self.rows):
            raise ValueError(f"Consumed {self.cursor} of {len(self.rows)} v006 parent rows")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    partial = Path(f"{path}.partial")
    for candidate in (path, partial):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite {candidate}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    if json.loads(partial.read_text(encoding="utf-8")) != payload:
        raise ValueError(f"JSON verification changed payload for {path}")
    os.replace(partial, path)


def _atomic_replace_json(path: Path, payload: dict) -> None:
    """Atomically advance a run-state JSON while preserving the previous state on failure."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Run-state JSON does not exist: {path}")
    partial = path.with_name(f"{path.name}.update.partial")
    if partial.exists():
        raise FileExistsError(f"Refusing to overwrite {partial}")
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    if json.loads(partial.read_text(encoding="utf-8")) != payload:
        raise ValueError(f"JSON verification changed payload for {path}")
    os.replace(partial, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    partial = Path(f"{path}.partial")
    for candidate in (path, partial):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite {candidate}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if partial.read_text(encoding="utf-8") != text:
        raise ValueError(f"Text verification changed payload for {path}")
    os.replace(partial, path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    partial = Path(f"{path}.partial")
    for candidate in (path, partial):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite {candidate}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(partial, index=False)
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    verified = pd.read_csv(partial)
    if len(verified) != len(frame) or list(verified.columns) != list(frame.columns):
        raise ValueError(f"CSV verification failed for {path}")
    os.replace(partial, path)


def _atomic_write_npy(path: Path, values: np.ndarray) -> None:
    path = Path(path)
    partial = Path(f"{path}.partial")
    for candidate in (path, partial):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite {candidate}")
    with partial.open("xb") as handle:
        np.save(handle, np.asarray(values))
        handle.flush()
        os.fsync(handle.fileno())
    verified = np.load(partial, mmap_mode="r")
    if verified.shape != np.asarray(values).shape or not np.array_equal(verified, values):
        raise ValueError(f"NumPy verification failed for {path}")
    os.replace(partial, path)


def _atomic_write_pickle(path: Path, payload: object) -> None:
    path = Path(path)
    partial = Path(f"{path}.partial")
    for candidate in (path, partial):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite {candidate}")
    with partial.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    with partial.open("rb") as handle:
        pickle.load(handle)
    os.replace(partial, path)


def atomic_publish_parquet(partial_path: Path, final_path: Path) -> None:
    if pq is None:
        raise RuntimeError("pyarrow is required to validate parquet artifacts")
    partial_path = Path(partial_path)
    final_path = Path(final_path)
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite {final_path}")
    parquet = pq.ParquetFile(partial_path)
    if parquet.metadata is None or parquet.metadata.num_rows < 1:
        raise ValueError(f"Partial parquet has no rows: {partial_path}")
    os.replace(partial_path, final_path)


def donor_contexts(
    catalog: DonorTable, manifest: pd.DataFrame, folds: Sequence[int]
) -> tuple[dict[tuple[int, ...], DonorTable], list[dict]]:
    """Build exactly the five singleton and ten pair exclusion contexts."""
    fold_values = tuple(sorted(int(fold) for fold in folds))
    if fold_values != (1, 2, 3, 4, 5):
        raise ValueError("v008 requires the five canonical folds labelled 1..5")
    fold_map = manifest.set_index("well")["fold"].astype(int).to_dict()
    if set(fold_map) != set(catalog.requested_well_ids):
        raise ValueError("Donor catalog and canonical manifest wells differ")
    exclusions = [(fold,) for fold in fold_values]
    exclusions.extend(
        (left, right)
        for index, left in enumerate(fold_values)
        for right in fold_values[index + 1 :]
    )
    tables: dict[tuple[int, ...], DonorTable] = {}
    audits: list[dict] = []
    for excluded in exclusions:
        requested = tuple(well for well, fold in fold_map.items() if fold not in excluded)
        table = subset_donor_table(catalog, requested)
        donor_folds = sorted({int(fold_map[well]) for well in table.well_ids})
        expected_folds = sorted(set(fold_values) - set(excluded))
        violations = [well for well in table.well_ids if int(fold_map[well]) in excluded]
        if donor_folds != expected_folds or violations:
            raise ValueError(f"Donor-fold exclusion failed for context {excluded}")
        tables[excluded] = table
        audits.append(
            {
                "excluded_folds": list(excluded),
                "expected_donor_folds": expected_folds,
                "actual_donor_folds": donor_folds,
                "requested_donor_wells": len(requested),
                "eligible_donor_wells": len(table.well_ids),
                "incomplete_donor_wells": list(table.incomplete_well_ids),
                "donor_fold_violations": violations,
            }
        )
    if len(tables) != 15:
        raise RuntimeError("v008 must construct exactly 15 unique donor contexts")
    return tables, audits


def _context_key(query_fold: int, outer_fold: int) -> tuple[int, ...]:
    return tuple(sorted({int(query_fold), int(outer_fold)}))


def build_training_streams(
    workspace: Path,
    manifest: pd.DataFrame,
    feature_path: Path,
) -> tuple[pd.DataFrame, list[dict], list[dict], DonorTable]:
    """Publish canonical and outer-by-inner cross-fitted streams one well at a time."""
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required to stream v008 features")
    workspace = Path(workspace)
    feature_path = Path(feature_path)
    partial_path = Path(f"{feature_path}.partial")
    for candidate in (feature_path, partial_path):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite {candidate}")
    train_root = workspace / "data" / "raw" / "train"
    paths = _horizontal_paths(train_root)
    fold_map = manifest.set_index("well")["fold"].astype(int).to_dict()
    path_ids = [well_id_from_path(path) for path in paths]
    if set(path_ids) != set(fold_map):
        raise ValueError("Canonical fold manifest does not match training wells")
    folds = tuple(sorted(set(fold_map.values())))
    catalog = build_donor_catalog(train_root, fold_map)
    contexts, context_audit = donor_contexts(catalog, manifest, folds)
    parent = ParentOOFReader(workspace, manifest)
    summary: list[dict] = []
    diagnostics: list[dict] = []
    writer = None
    try:
        for path_index, path in enumerate(paths):
            well = well_id_from_path(path)
            query_fold = int(fold_map[well])
            horizontal = pd.read_csv(path, usecols=list(QUERY_COLUMNS))
            validate_horizontal_schema(horizontal, role="query")
            context_predictions: dict[int, SpatialPrediction] = {}
            for outer_fold in folds:
                key = _context_key(query_fold, outer_fold)
                prediction = spatial_streams(
                    horizontal,
                    contexts[key],
                    query_well_id=well,
                )
                if prediction.diagnostics["plane"]["query_well_present_after_exclusion"]:
                    raise ValueError(
                        f"Self donor survived exclusion for {well}, outer {outer_fold}"
                    )
                context_predictions[outer_fold] = prediction
                diagnostics.append(
                    {
                        "well": well,
                        "query_fold": query_fold,
                        "outer_fold": int(outer_fold),
                        "excluded_folds": list(key),
                        **prediction.diagnostics,
                    }
                )
            canonical = context_predictions[query_fold]
            evaluation_indices = canonical.frame["_row_index"].to_numpy(dtype=np.int64)
            target_frame = pd.read_csv(path, usecols=["TVT"])
            target = pd.to_numeric(
                target_frame.iloc[evaluation_indices]["TVT"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            if not np.isfinite(target).all():
                raise ValueError(f"Suffix TVT target is non-finite for {well}")
            parent_values = parent.read_well(
                well, evaluation_indices, query_fold, expected_target=target
            )
            frame = pd.DataFrame(
                {
                    "_id": [f"{well}_{row}" for row in evaluation_indices],
                    "_well_id": well,
                    "_row_index": evaluation_indices,
                    "_target": target,
                    "fold": query_fold,
                    "parent_v006": parent_values,
                    "sp_plane_ancc_k10": canonical.frame["sp_plane_ancc_k10"].to_numpy(
                        dtype=np.float64
                    ),
                    "sp_plane_best6_k10": canonical.frame["sp_plane_best6_k10"].to_numpy(
                        dtype=np.float64
                    ),
                }
            )
            for outer_fold, prediction in context_predictions.items():
                if not np.array_equal(
                    prediction.frame["_row_index"].to_numpy(dtype=np.int64),
                    evaluation_indices,
                ):
                    raise ValueError(f"Spatial query/order mismatch for {well}, outer {outer_fold}")
                for stream in ("sp_plane_ancc_k10", "sp_plane_best6_k10"):
                    frame[nested_stream_column(outer_fold, stream)] = prediction.frame[
                        stream
                    ].to_numpy(dtype=np.float64)
            feature_values = frame[
                [
                    "parent_v006",
                    "sp_plane_ancc_k10",
                    "sp_plane_best6_k10",
                    *(
                        nested_stream_column(outer, stream)
                        for outer in folds
                        for stream in ("sp_plane_ancc_k10", "sp_plane_best6_k10")
                    ),
                ]
            ].to_numpy(dtype=np.float64)
            if not np.isfinite(feature_values).all():
                raise ValueError(f"Spatial feature artifact is non-finite for {well}")
            for stream in STREAM_COLUMNS:
                error = frame[stream].to_numpy(dtype=np.float64) - target
                squared_error = float(np.dot(error, error))
                summary.append(
                    {
                        "well": well,
                        "fold": query_fold,
                        "stream": stream,
                        "rows": len(frame),
                        "squared_error": squared_error,
                        "rmse": float(np.sqrt(squared_error / len(frame))),
                        "selected_formation": (
                            canonical.diagnostics["selected_formation"]
                            if stream == "sp_plane_best6_k10"
                            else ""
                        ),
                    }
                )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(partial_path, table.schema, compression="zstd")
            writer.write_table(table)
            if (path_index + 1) % 10 == 0 or path_index + 1 == len(paths):
                print(
                    f"v008 feature progress: {path_index + 1}/{len(paths)} wells "
                    f"({time.strftime('%H:%M:%S')})",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
    parent.finish()
    atomic_publish_parquet(partial_path, feature_path)
    return pd.DataFrame(summary), diagnostics, context_audit, catalog


def nested_stack_weights(
    feature_path: Path, folds: Sequence[int] = (1, 2, 3, 4, 5)
) -> tuple[list[dict], SimplexFit]:
    """Fit outer weights from streams whose donors exclude outer and inner folds."""
    fold_values = tuple(sorted(int(fold) for fold in folds))
    if fold_values != (1, 2, 3, 4, 5):
        raise ValueError("Nested v008 stack requires canonical folds 1..5")
    selections: list[dict] = []
    for outer_fold in fold_values:
        columns = [
            "_target",
            "fold",
            "parent_v006",
            nested_stream_column(outer_fold, "sp_plane_ancc_k10"),
            nested_stream_column(outer_fold, "sp_plane_best6_k10"),
        ]
        rows = pd.read_parquet(feature_path, columns=columns)
        train_rows = rows.loc[rows["fold"].to_numpy(dtype=int) != outer_fold]
        expected_inner = sorted(set(fold_values) - {outer_fold})
        if sorted(train_rows["fold"].astype(int).unique().tolist()) != expected_inner:
            raise ValueError(f"Outer fold {outer_fold} is missing an inner held-out fold")
        streams = train_rows.iloc[:, 2:].to_numpy(dtype=np.float64)
        target = train_rows["_target"].to_numpy(dtype=np.float64)
        fit = fit_simplex_weights(streams, target)
        selections.append(
            {
                "fold": outer_fold,
                "meta_training_folds": expected_inner,
                "donor_exclusion": "query fold union outer fold",
                "weights": dict(zip(STREAM_COLUMNS, fit.weights.tolist())),
                "training_rows": fit.rows,
                "training_squared_error": fit.squared_error,
                "training_rmse": float(np.sqrt(fit.squared_error / fit.rows)),
                "centered_delta_gram": fit.delta_gram.tolist(),
                "centered_delta_cross": fit.delta_cross.tolist(),
                "centered_residual_squared": fit.residual_squared,
                "direct_residual_verified": True,
            }
        )
    final_rows = pd.read_parquet(feature_path, columns=["_target", *STREAM_COLUMNS])
    final_fit = fit_simplex_weights(
        final_rows.loc[:, STREAM_COLUMNS].to_numpy(dtype=np.float64),
        final_rows["_target"].to_numpy(dtype=np.float64),
    )
    return selections, final_fit


def materialize_nested_oof(
    feature_path: Path, selections: Sequence[Mapping]
) -> tuple[pd.DataFrame, np.ndarray]:
    columns = ["_id", "_well_id", "_row_index", "_target", "fold", *STREAM_COLUMNS]
    frame = pd.read_parquet(feature_path, columns=columns)
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    expected_folds = sorted(frame["fold"].astype(int).unique().tolist())
    if sorted(int(item["fold"]) for item in selections) != expected_folds:
        raise ValueError("Nested stack selections do not cover each canonical fold exactly once")
    for selection in selections:
        fold = int(selection["fold"])
        mask = frame["fold"].to_numpy(dtype=int) == fold
        weights = np.asarray(
            [selection["weights"][stream] for stream in STREAM_COLUMNS], dtype=np.float64
        )
        if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError(f"Infeasible nested weights for fold {fold}")
        prediction[mask] = frame.loc[mask, STREAM_COLUMNS].to_numpy(dtype=np.float64) @ weights
    if not np.isfinite(prediction).all():
        raise ValueError("Nested OOF predictions contain non-finite values")
    return frame.iloc[:, :5].copy(), prediction


def baseline_scores_on_manifest(workspace: Path, manifest: pd.DataFrame) -> dict:
    rows = pd.read_parquet(
        Path(workspace) / "models" / PARENT_VERSION / "oof_rows.parquet",
        columns=["_well_id", "_target", "fold"],
    )
    prediction = np.load(Path(workspace) / "models" / PARENT_VERSION / "oof_preds.npy")
    if len(rows) != len(prediction):
        raise ValueError("v006 OOF rows and predictions differ in length")
    mapped = rows["_well_id"].map(manifest.set_index("well")["fold"])
    if mapped.isna().any() or not np.array_equal(
        mapped.to_numpy(dtype=int), rows["fold"].to_numpy(dtype=int)
    ):
        raise ValueError("v006 parent folds do not match the canonical manifest")
    target = rows["_target"].to_numpy(dtype=np.float64)
    fold_values = rows["fold"].to_numpy(dtype=int)
    return {
        "overall": rmse(target, prediction),
        "fold_scores": [
            rmse(target[fold_values == fold], prediction[fold_values == fold])
            for fold in sorted(manifest["fold"].astype(int).unique())
        ],
    }


def candidate_decision(
    overall: float,
    folds_better: int,
    *,
    inference_failures: int = 0,
    nonfinite_failures: int = 0,
    donor_fold_violations: int = 0,
    self_donor_violations: int = 0,
    test_failures: int = 0,
) -> str:
    if not np.isfinite(overall) or int(folds_better) != folds_better or folds_better < 0:
        raise ValueError("Gate score and improved-fold count are invalid")
    counts = (
        inference_failures,
        nonfinite_failures,
        donor_fold_violations,
        self_donor_violations,
        test_failures,
    )
    if any(int(value) != value or value < 0 for value in counts):
        raise ValueError("Gate failure counts must be non-negative integers")
    failures = sum(int(value) for value in counts)
    if overall <= CANDIDATE_GATE and folds_better >= 4:
        return "candidate_ready" if failures == 0 else "candidate_blocked_failures"
    if overall < PROMISING_GATE and folds_better >= 3:
        return "promising_continue" if failures == 0 else "promising_blocked_failures"
    return "exhausted"


def _peak_rss_gib() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0**2 if sys.platform != "darwin" else 1024.0**3
    return usage / divisor


def _plan_path(workspace: Path) -> Path:
    return Path(workspace) / "reports" / "v008_formation_spatial_plan.md"


def _parent_source_path() -> Path:
    return Path(__file__).with_name("rogii_state_space.py").resolve()


def _smoke_path(workspace: Path) -> Path:
    return Path(workspace) / "reports" / "v008_formation_runtime_smoke.json"


def _legacy_parent_data_fingerprint(train_root: Path) -> str:
    """Reproduce the metadata fingerprint recorded by the immutable v006 run."""
    digest = hashlib.sha256()
    paths = sorted(Path(train_root).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No parent input CSV files found under {train_root}")
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _validate_parent_run_contract(
    workspace: Path, fold_manifest_path: Path, parent_run_path: Path
) -> dict:
    payload = json.loads(Path(parent_run_path).read_text(encoding="utf-8"))
    expected = {
        "status": "completed",
        "source_sha256": sha256_file(_parent_source_path()),
        "fold_manifest_sha256": sha256_file(fold_manifest_path),
        "data_fingerprint": _legacy_parent_data_fingerprint(
            Path(workspace) / "data" / "raw" / "train"
        ),
        "feature_path": "data/features/v003.parquet",
        "final_candidate": PARENT_CANDIDATE,
    }
    mismatches = [name for name, value in expected.items() if payload.get(name) != value]
    if mismatches:
        raise ValueError(f"v006 parent run contract changed: {mismatches}")
    return payload


def _critical_input_hashes(workspace: Path, fold_manifest_path: Path) -> dict:
    workspace = Path(workspace)
    paths = {
        "config": workspace / "config.yaml",
        "fold_manifest": fold_manifest_path,
        "plan": _plan_path(workspace),
        "source": Path(__file__).resolve(),
        "parent_source": _parent_source_path(),
        "parent_cv_scores": workspace / "models" / PARENT_VERSION / "cv_scores.json",
        "parent_run": workspace / "models" / PARENT_VERSION / "run.json",
        "parent_model": workspace / "models" / PARENT_VERSION / "model.pkl",
        "parent_oof_predictions": workspace / "models" / PARENT_VERSION / "oof_preds.npy",
        "parent_oof_rows": workspace / "models" / PARENT_VERSION / "oof_rows.parquet",
        "parent_test_predictions": workspace / "models" / PARENT_VERSION / "test_preds.npy",
        "parent_feature": workspace / "data" / "features" / "v003.parquet",
        "sample_submission": workspace / "data" / "raw" / "sample_submission.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Critical v008 inputs are missing: {missing}")
    _validate_parent_run_contract(workspace, fold_manifest_path, paths["parent_run"])
    return {name: sha256_file(path) for name, path in paths.items()}


def _validate_test_schemas(test_root: Path) -> list[Path]:
    paths = _horizontal_paths(test_root)
    for path in paths:
        _validate_csv_schema(path, role="test")
    return paths


def run_smoke(workspace: Path, *, wells: int = SMOKE_WELLS) -> dict:
    """Run exact five-context target-free work on the six predeclared wells."""
    if wells != SMOKE_WELLS:
        raise ValueError(f"v008 smoke requires exactly {SMOKE_WELLS} wells")
    workspace = Path(workspace).resolve()
    output_path = _smoke_path(workspace)
    partial_output = Path(f"{output_path}.partial")
    for candidate in (output_path, partial_output):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite {candidate}")
    fold_path, manifest = ensure_fold_manifest(workspace, 5)
    train_root = workspace / "data" / "raw" / "train"
    test_root = workspace / "data" / "raw" / "test"
    source_path = Path(__file__).resolve()
    plan_path = _plan_path(workspace)
    input_hashes_start = _critical_input_hashes(workspace, fold_path)
    train_fingerprint_start = data_fingerprint(train_root)
    test_fingerprint_start = data_fingerprint(test_root)
    _validate_test_schemas(test_root)
    fold_map = manifest.set_index("well")["fold"].astype(int).to_dict()
    path_map = {well_id_from_path(path): path for path in _horizontal_paths(train_root)}

    selected_records = []
    for well, expected_fold, expected_suffix in SMOKE_SELECTION:
        if well not in path_map:
            raise ValueError(f"Fixed smoke well is missing: {well}")
        if int(fold_map.get(well, -1)) != expected_fold:
            raise ValueError(f"Fixed smoke fold changed for {well}")
        values = pd.read_csv(path_map[well], usecols=["TVT_input"])["TVT_input"]
        suffix_rows = int(pd.to_numeric(values, errors="coerce").isna().sum())
        if suffix_rows != expected_suffix:
            raise ValueError(f"Fixed smoke suffix count changed for {well}")
        selected_records.append(
            {
                "well": well,
                "fold": expected_fold,
                "suffix_rows": expected_suffix,
            }
        )

    setup_started = time.perf_counter()
    catalog = build_donor_catalog(train_root, fold_map)
    contexts, context_audit = donor_contexts(catalog, manifest, (1, 2, 3, 4, 5))
    context_build_seconds = time.perf_counter() - setup_started
    if len(contexts) != 15:
        raise RuntimeError("Smoke did not construct exactly 15 donor contexts")

    query_started = time.perf_counter()
    failures: list[dict] = []
    records: list[dict] = []
    query_rows = 0
    for well, expected_fold, expected_suffix in SMOKE_SELECTION:
        path = path_map[well]
        horizontal = pd.read_csv(path, usecols=list(QUERY_COLUMNS))
        context_records = []
        for outer_fold in range(1, 6):
            started = time.perf_counter()
            key = _context_key(expected_fold, outer_fold)
            try:
                prediction = spatial_streams(horizontal, contexts[key], query_well_id=well)
                if len(prediction.frame) != expected_suffix:
                    raise ValueError("Smoke suffix output count changed")
                if prediction.diagnostics["plane"]["query_well_present_after_exclusion"]:
                    raise ValueError("Smoke detected a self donor")
                context_records.append(
                    {
                        "outer_fold": outer_fold,
                        "excluded_folds": list(key),
                        "query_rows": 640 + expected_suffix,
                        "runtime_seconds": round(time.perf_counter() - started, 6),
                        "selected_formation": prediction.diagnostics["selected_formation"],
                        "same_id_donors_excluded": prediction.diagnostics["plane"][
                            "same_id_donors_excluded"
                        ],
                    }
                )
                query_rows += 640 + expected_suffix
            except Exception as exc:
                failures.append(
                    {
                        "well": well,
                        "outer_fold": outer_fold,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        records.append(
            {
                "well": well,
                "fold": expected_fold,
                "suffix_rows": expected_suffix,
                "contexts": context_records,
            }
        )
    query_seconds = time.perf_counter() - query_started
    if query_rows != SMOKE_QUERY_ROWS and not failures:
        raise ValueError(f"Smoke queried {query_rows} rows, expected {SMOKE_QUERY_ROWS}")

    total_suffix_rows = int(manifest["scored_rows"].sum())
    full_query_rows = 5 * (total_suffix_rows + 640 * len(manifest))
    if full_query_rows != FULL_NESTED_QUERY_ROWS:
        raise ValueError(
            f"Full v008 work units changed: {full_query_rows} != {FULL_NESTED_QUERY_ROWS}"
        )
    projected_hours = (
        PROJECTION_SAFETY_FACTOR
        * (context_build_seconds + query_seconds * (full_query_rows / SMOKE_QUERY_ROWS))
        / 3600.0
    )
    peak_rss = _peak_rss_gib()
    input_hashes_end = _critical_input_hashes(workspace, fold_path)
    train_fingerprint_end = data_fingerprint(train_root)
    test_fingerprint_end = data_fingerprint(test_root)
    hashes_stable = (
        input_hashes_start == input_hashes_end
        and train_fingerprint_start == train_fingerprint_end
        and test_fingerprint_start == test_fingerprint_end
    )
    donor_fold_violations = sum(len(item["donor_fold_violations"]) for item in context_audit)
    eligible = bool(
        not failures
        and hashes_stable
        and donor_fold_violations == 0
        and query_rows == SMOKE_QUERY_ROWS
        and projected_hours <= MAX_PROJECTED_HOURS
        and peak_rss <= MAX_PEAK_RSS_GIB
    )
    payload = {
        "version": "v008",
        "intended_use": "target-free runtime, memory, provenance, and donor-isolation gate",
        "target_read": False,
        "fixed_settings": _fixed_settings(),
        "required_smoke_wells": SMOKE_WELLS,
        "selected_wells": selected_records,
        "expected_selection": [
            {"well": well, "fold": fold, "suffix_rows": rows}
            for well, fold, rows in SMOKE_SELECTION
        ],
        "selected_query_rows": [
            {"well": well, "query_rows": 5 * (640 + rows)} for well, _, rows in SMOKE_SELECTION
        ],
        "smoke_contexts_per_well": 5,
        "unique_donor_contexts": len(contexts),
        "donor_context_audit": context_audit,
        "smoke_query_rows": query_rows,
        "full_nested_query_rows": full_query_rows,
        "total_suffix_rows": total_suffix_rows,
        "total_training_wells": len(manifest),
        "context_build_seconds": context_build_seconds,
        "tree_build_seconds": context_build_seconds,
        "query_seconds": query_seconds,
        "projected_full_hours": projected_hours,
        "projection_safety_factor": PROJECTION_SAFETY_FACTOR,
        "peak_rss_gib": peak_rss,
        "max_projected_hours": MAX_PROJECTED_HOURS,
        "max_peak_rss_gib": MAX_PEAK_RSS_GIB,
        "eligible_for_full_cv": eligible,
        "source_path": str(source_path),
        "plan_path": str(plan_path),
        "source_sha256": input_hashes_start["source"],
        "source_sha256_end": input_hashes_end["source"],
        "plan_sha256": input_hashes_start["plan"],
        "plan_sha256_end": input_hashes_end["plan"],
        "fold_manifest_sha256": input_hashes_start["fold_manifest"],
        "fold_manifest_sha256_end": input_hashes_end["fold_manifest"],
        "input_sha256_start": input_hashes_start,
        "input_sha256_end": input_hashes_end,
        "train_data_fingerprint": train_fingerprint_start,
        "train_data_fingerprint_end": train_fingerprint_end,
        "test_data_fingerprint": test_fingerprint_start,
        "test_data_fingerprint_end": test_fingerprint_end,
        "data_fingerprint_algorithm": DATA_FINGERPRINT_ALGORITHM,
        "hashes_stable": hashes_stable,
        "records": records,
        "failures": failures,
    }
    _atomic_write_json(output_path, payload)
    return payload


def _require_runtime_smoke(workspace: Path) -> tuple[Path, dict]:
    workspace = Path(workspace).resolve()
    path = _smoke_path(workspace)
    if not path.is_file():
        raise FileNotFoundError(f"Required v008 runtime smoke is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fold_path, _ = ensure_fold_manifest(workspace, 5)
    expected_hashes = _critical_input_hashes(workspace, fold_path)
    expected_selection = [
        {"well": well, "fold": fold, "suffix_rows": rows} for well, fold, rows in SMOKE_SELECTION
    ]
    current_train_fingerprint = data_fingerprint(workspace / "data" / "raw" / "train")
    current_test_fingerprint = data_fingerprint(workspace / "data" / "raw" / "test")
    checks = {
        "version": payload.get("version") == "v008",
        "target_free": payload.get("target_read") is False,
        "eligible": payload.get("eligible_for_full_cv") is True,
        "selection": payload.get("selected_wells") == expected_selection,
        "contexts": payload.get("unique_donor_contexts") == 15,
        "smoke_rows": payload.get("smoke_query_rows") == SMOKE_QUERY_ROWS,
        "full_rows": payload.get("full_nested_query_rows") == FULL_NESTED_QUERY_ROWS,
        "settings": payload.get("fixed_settings") == _fixed_settings(),
        "input_start": payload.get("input_sha256_start") == expected_hashes,
        "input_end": payload.get("input_sha256_end") == expected_hashes,
        "source_aliases": payload.get("source_sha256") == expected_hashes["source"]
        and payload.get("source_sha256_end") == expected_hashes["source"],
        "plan_aliases": payload.get("plan_sha256") == expected_hashes["plan"]
        and payload.get("plan_sha256_end") == expected_hashes["plan"],
        "fold_manifest_aliases": payload.get("fold_manifest_sha256")
        == expected_hashes["fold_manifest"]
        and payload.get("fold_manifest_sha256_end") == expected_hashes["fold_manifest"],
        "train_data": payload.get("train_data_fingerprint") == current_train_fingerprint,
        "train_data_end": payload.get("train_data_fingerprint_end") == current_train_fingerprint,
        "test_data": payload.get("test_data_fingerprint") == current_test_fingerprint,
        "test_data_end": payload.get("test_data_fingerprint_end") == current_test_fingerprint,
        "hashes_stable": payload.get("hashes_stable") is True,
        "failures": payload.get("failures") == [],
        "runtime": float(payload.get("projected_full_hours", np.inf)) <= MAX_PROJECTED_HOURS,
        "memory": float(payload.get("peak_rss_gib", np.inf)) <= MAX_PEAK_RSS_GIB,
        "context_audit": len(payload.get("donor_context_audit", [])) == 15
        and all(
            item.get("donor_fold_violations") == []
            for item in payload.get("donor_context_audit", [])
        ),
        "record_contexts": len(payload.get("records", [])) == SMOKE_WELLS
        and all(len(item.get("contexts", [])) == 5 for item in payload.get("records", [])),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"v008 runtime smoke no longer satisfies: {failed}")
    return path, payload


def predict_visible_test(
    workspace: Path,
    donors: DonorTable,
    weights: Sequence[float],
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Predict visible test only; never materialize a submission file."""
    workspace = Path(workspace)
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (3,) or np.any(weight_array < 0.0):
        raise ValueError("Visible-test simplex weights are invalid")
    if not np.isclose(weight_array.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Visible-test simplex weights do not sum to one")
    frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    diagnostics: list[dict] = []
    sample = pd.read_csv(workspace / "data" / "raw" / "sample_submission.csv")
    locked_parent = np.load(workspace / "models" / PARENT_VERSION / "test_preds.npy")
    if list(sample.columns) != ["id", "tvt"] or sample["id"].duplicated().any():
        raise ValueError("Unexpected sample submission contract")
    if locked_parent.shape != (len(sample),) or not np.isfinite(locked_parent).all():
        raise ValueError("Locked v006 visible-test predictions do not match the sample")
    parent_by_id = pd.Series(
        np.asarray(locked_parent, dtype=np.float64), index=sample["id"].astype(str)
    )
    for path in _validate_test_schemas(workspace / "data" / "raw" / "test"):
        well = well_id_from_path(path)
        horizontal = pd.read_csv(path, usecols=list(QUERY_COLUMNS))
        try:
            spatial = spatial_streams(horizontal, donors, query_well_id=well)
            if spatial.diagnostics["plane"]["query_well_present_after_exclusion"]:
                raise ValueError("Same-ID donor survived visible-test exclusion")
            spatial_rows = spatial.frame["_row_index"].to_numpy(dtype=np.int64)
            ids = pd.Index([f"{well}_{row}" for row in spatial_rows])
            parent_values = parent_by_id.reindex(ids).to_numpy(dtype=np.float64)
            if not np.isfinite(parent_values).all():
                raise ValueError("Locked v006 parent does not cover visible-test IDs")
            streams = np.column_stack(
                [
                    parent_values,
                    spatial.frame["sp_plane_ancc_k10"].to_numpy(dtype=np.float64),
                    spatial.frame["sp_plane_best6_k10"].to_numpy(dtype=np.float64),
                ]
            )
            prediction = streams @ weight_array
            if not np.isfinite(prediction).all():
                raise ValueError("Visible-test stacked prediction is non-finite")
            frames.append(
                pd.DataFrame(
                    {
                        "_id": [f"{well}_{row}" for row in spatial_rows],
                        "prediction": prediction,
                    }
                )
            )
            diagnostics.append({"well": well, **spatial.diagnostics})
        except Exception as exc:
            failures.append({"well": well, "error": f"{type(exc).__name__}: {exc}"})
    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["_id", "prediction"])
    )
    return result, failures, diagnostics


def _align_test_predictions(workspace: Path, predictions: pd.DataFrame) -> np.ndarray:
    sample = pd.read_csv(Path(workspace) / "data" / "raw" / "sample_submission.csv")
    if list(sample.columns) != ["id", "tvt"] or sample["id"].duplicated().any():
        raise ValueError("Unexpected sample submission contract")
    if predictions["_id"].duplicated().any():
        raise ValueError("Visible-test prediction IDs are duplicated")
    mapped = sample["id"].map(predictions.set_index("_id")["prediction"])
    values = mapped.to_numpy(dtype=np.float64)
    if mapped.isna().any() or not np.isfinite(values).all():
        raise ValueError("Visible-test predictions do not exactly cover sample IDs")
    return values


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    if pq is None:
        raise RuntimeError("pyarrow is required to write parquet artifacts")
    path = Path(path)
    partial = Path(f"{path}.partial")
    for candidate in (path, partial):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite {candidate}")
    frame.to_parquet(partial, index=False, compression="zstd")
    parquet = pq.ParquetFile(partial)
    if parquet.metadata is None or parquet.metadata.num_rows != len(frame):
        raise ValueError(f"Parquet verification failed for {path}")
    os.replace(partial, path)


def _runtime_environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": getattr(pa, "__version__", None),
    }


def _selection_weights(selection: Mapping) -> np.ndarray:
    return np.asarray([selection["weights"][stream] for stream in STREAM_COLUMNS], dtype=np.float64)


def _write_summary(
    path: Path,
    *,
    overall: float,
    fold_scores: Sequence[float],
    folds_better: int,
    decision: str,
    final_weights: np.ndarray,
    donor_context_count: int,
) -> None:
    text = "\n".join(
        [
            "# ROGII v008 Fold-Safe Formation Spatial Stack",
            "",
            f"Status: `{decision}`.",
            "",
            "## Grouped CV",
            "",
            f"- Nested pooled OOF RMSE: `{overall:.9f}`",
            f"- Canonical folds better than v006: `{folds_better} / 5`",
            f"- Fold RMSE: `{', '.join(f'{value:.6f}' for value in fold_scores)}`",
            "- Final all-OOF simplex weights: "
            + ", ".join(
                f"`{name}={weight:.9f}`" for name, weight in zip(STREAM_COLUMNS, final_weights)
            ),
            f"- Audited donor contexts: `{donor_context_count}`",
            "",
            "## Leakage Boundary",
            "",
            "Each outer-fold meta fit uses inner streams whose donor universe excludes both "
            "the outer fold and the inner query fold. Validation streams exclude their own "
            "outer fold. Final test inference excludes a same-ID donor before coordinate scaling.",
            "",
            "No submission CSV, notebook, Kaggle API request, or competition submission is "
            "created by this experiment.",
            "",
        ]
    )
    _atomic_write_text(path, text)


def train(workspace: Path, *, version: str = "v008", folds: int = 5) -> Path:
    """Run the fixed v008 protocol after the immutable target-free smoke gate."""
    if version != "v008" or folds != 5:
        raise ValueError("The predeclared formation experiment is exactly v008 with five folds")
    workspace = Path(workspace).resolve()
    project_root = workspace.parents[1]
    smoke_path, smoke = _require_runtime_smoke(workspace)
    fold_path, manifest = ensure_fold_manifest(workspace, folds)
    input_hashes_start = _critical_input_hashes(workspace, fold_path)
    train_fingerprint_start = data_fingerprint(workspace / "data" / "raw" / "train")
    test_fingerprint_start = data_fingerprint(workspace / "data" / "raw" / "test")

    model_dir = workspace / "models" / version
    staging_dir = workspace / "models" / f".{version}.partial"
    for path in (model_dir, staging_dir):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    feature_path = next_feature_path(workspace)
    report_paths = {
        "candidate_scan": workspace / "reports" / "v008_formation_candidate_scan.csv",
        "diagnostics": workspace / "reports" / "v008_formation_diagnostics.json",
        "summary": workspace / "reports" / "v008_formation_spatial_summary.md",
    }
    for path in (feature_path, *report_paths.values()):
        for candidate in (path, Path(f"{path}.partial")):
            if candidate.exists():
                raise FileExistsError(f"Refusing to overwrite {candidate}")
    staging_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    command = " ".join(sys.argv)
    run_common = {
        "version": version,
        "parent_run": PARENT_VERSION,
        "template": "rogii-fold-safe-formation-local-plane-nested-simplex",
        "command": command,
        "reproduction_command": (
            "uv run --with-requirements "
            "workspaces/rogii-wellbore-geology-prediction/requirements-beam.txt python "
            "workspaces/rogii-wellbore-geology-prediction/scripts/rogii_formation_spatial.py "
            f"--workspace {workspace} train --version v008 --folds 5"
        ),
        "git_commit": git_commit(project_root),
        "git_dirty_paths": git_dirty_paths(project_root),
        "source_sha256": input_hashes_start["source"],
        "plan_sha256": input_hashes_start["plan"],
        "parent_source_sha256": input_hashes_start["parent_source"],
        "fixed_settings": _fixed_settings(),
        "random_seed": None,
        "cv_splitter": "canonical row-balanced GroupKFold by well; outer-by-inner donor exclusion",
        "fold_manifest": str(fold_path.relative_to(workspace)),
        "fold_manifest_sha256": input_hashes_start["fold_manifest"],
        "metric": "rmse",
        "metric_direction": "minimize",
        "target_column": "TVT",
        "id_column": "id",
        "feature_path": str(feature_path.relative_to(workspace)),
        "runtime_smoke": str(smoke_path.relative_to(workspace)),
        "runtime_smoke_sha256": sha256_file(smoke_path),
        "runtime_projection_hours": smoke["projected_full_hours"],
        "input_sha256": input_hashes_start,
        "train_data_fingerprint": train_fingerprint_start,
        "test_data_fingerprint": test_fingerprint_start,
        "data_fingerprint_algorithm": DATA_FINGERPRINT_ALGORITHM,
        "environment": _runtime_environment(),
        "submission_path": None,
        "submitted": False,
    }
    run_path = staging_dir / "run.json"
    _atomic_write_json(
        run_path,
        {
            **run_common,
            "status": "running",
            "runtime_seconds": None,
            "error": None,
            "decision": "running",
            "test_predictions_saved": False,
        },
    )
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        stream_summary, diagnostics, context_audit, catalog = build_training_streams(
            workspace, manifest, feature_path
        )
        selections, final_fit = nested_stack_weights(feature_path)
        oof_rows, oof_prediction = materialize_nested_oof(feature_path, selections)
        target = oof_rows["_target"].to_numpy(dtype=np.float64)
        fold_values = oof_rows["fold"].to_numpy(dtype=int)
        fold_scores = [
            rmse(target[fold_values == fold], oof_prediction[fold_values == fold])
            for fold in range(1, folds + 1)
        ]
        overall = rmse(target, oof_prediction)
        parent_scores = baseline_scores_on_manifest(workspace, manifest)
        folds_better = sum(
            score < baseline for score, baseline in zip(fold_scores, parent_scores["fold_scores"])
        )
        for selection, validation_score in zip(selections, fold_scores):
            fold = int(selection["fold"])
            selection["validation_rmse"] = validation_score
            selection["validation_rows"] = int((fold_values == fold).sum())
            selection["validation_wells"] = int(
                oof_rows.loc[fold_values == fold, "_well_id"].nunique()
            )

        donor_fold_violations = sum(len(item["donor_fold_violations"]) for item in context_audit)
        self_donor_violations = sum(
            int(item["plane"]["query_well_present_after_exclusion"]) for item in diagnostics
        )
        decision = candidate_decision(
            overall,
            folds_better,
            donor_fold_violations=donor_fold_violations,
            self_donor_violations=self_donor_violations,
        )
        visible_test_failures: list[dict] = []
        visible_test_diagnostics: list[dict] = []
        test_values: np.ndarray | None = None
        if decision == "candidate_ready":
            visible, visible_test_failures, visible_test_diagnostics = predict_visible_test(
                workspace, catalog, final_fit.weights
            )
            decision = candidate_decision(
                overall,
                folds_better,
                donor_fold_violations=donor_fold_violations,
                self_donor_violations=self_donor_violations,
                test_failures=len(visible_test_failures),
            )
            if decision == "candidate_ready":
                test_values = _align_test_predictions(workspace, visible)

        scored = oof_rows.assign(
            prediction=oof_prediction,
            squared_error=np.square(target - oof_prediction),
        )
        stack_by_well = (
            scored.groupby(["_well_id", "fold"], sort=True)
            .agg(rows=("squared_error", "size"), squared_error=("squared_error", "sum"))
            .reset_index()
            .rename(columns={"_well_id": "well"})
        )
        stack_by_well["rmse"] = np.sqrt(stack_by_well["squared_error"] / stack_by_well["rows"])
        stack_by_well["stream"] = "sp_nnls_v006"
        stack_by_well["selected_formation"] = ""
        candidate_scan = pd.concat(
            [stream_summary, stack_by_well[stream_summary.columns]], ignore_index=True
        ).sort_values(["well", "stream"])

        diagnostic_payload = {
            "version": version,
            "source_sha256": input_hashes_start["source"],
            "plan_sha256": input_hashes_start["plan"],
            "fixed_settings": _fixed_settings(),
            "donor_catalog": {
                "requested_wells": len(catalog.requested_well_ids),
                "eligible_complete_six_wells": len(catalog.well_ids),
                "incomplete_wells": list(catalog.incomplete_well_ids),
            },
            "donor_context_audit": context_audit,
            "training_inference": diagnostics,
            "visible_test_inference": visible_test_diagnostics,
            "visible_test_failures": visible_test_failures,
        }
        _atomic_write_csv(report_paths["candidate_scan"], candidate_scan)
        _atomic_write_json(report_paths["diagnostics"], diagnostic_payload)
        _write_summary(
            report_paths["summary"],
            overall=overall,
            fold_scores=fold_scores,
            folds_better=folds_better,
            decision=decision,
            final_weights=final_fit.weights,
            donor_context_count=len(context_audit),
        )

        cv_scores = {
            "metric": "rmse",
            "direction": "minimize",
            "fold_scores": fold_scores,
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
            "overall": overall,
            "valid_rows": len(oof_rows),
            "valid_wells": int(oof_rows["_well_id"].nunique()),
            "selection_protocol": (
                "outer k weights fit on inner-heldout streams whose formation donors exclude "
                "both outer k and inner j"
            ),
            "fold_selections": selections,
            "final_weights": dict(zip(STREAM_COLUMNS, final_fit.weights.tolist())),
            "final_weight_fit_rows": final_fit.rows,
            "final_weight_fit_squared_error": final_fit.squared_error,
            "final_centered_delta_gram": final_fit.delta_gram.tolist(),
            "final_centered_delta_cross": final_fit.delta_cross.tolist(),
            "final_centered_residual_squared": final_fit.residual_squared,
            "final_direct_residual_verified": True,
            "canonical_v006": parent_scores,
            "folds_better_than_v006": folds_better,
            "candidate_gate": CANDIDATE_GATE,
            "promising_gate": PROMISING_GATE,
            "training_inference_failures": 0,
            "nonfinite_failures": 0,
            "donor_fold_violations": donor_fold_violations,
            "self_donor_violations": self_donor_violations,
            "visible_test_failures": visible_test_failures,
            "decision": decision,
        }
        model_payload = {
            "model": "fold_safe_formation_local_plane_simplex",
            "streams": STREAM_COLUMNS,
            "fold_selections": selections,
            "final_weights": final_fit.weights,
            "fixed_settings": _fixed_settings(),
            "donor_well_ids": catalog.well_ids,
            "donor_coordinates": catalog.coordinates,
            "donor_formations": catalog.formations,
            "incomplete_donor_well_ids": catalog.incomplete_well_ids,
            "donor_context_audit": context_audit,
        }
        importance_rows = []
        for selection in selections:
            for stream in STREAM_COLUMNS:
                importance_rows.append(
                    {
                        "scope": "outer_fold",
                        "fold": int(selection["fold"]),
                        "feature": stream,
                        "weight": float(selection["weights"][stream]),
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

        _atomic_write_json(staging_dir / "cv_scores.json", cv_scores)
        _atomic_write_npy(staging_dir / "oof_preds.npy", oof_prediction)
        _atomic_write_parquet(staging_dir / "oof_rows.parquet", oof_rows)
        _atomic_write_pickle(staging_dir / "model.pkl", model_payload)
        _atomic_write_text(staging_dir / "feature_list.txt", "\n".join(feature_names) + "\n")
        _atomic_write_csv(staging_dir / "importance.csv", importance)
        if test_values is not None:
            _atomic_write_npy(staging_dir / "test_preds.npy", test_values)

        input_hashes_end = _critical_input_hashes(workspace, fold_path)
        train_fingerprint_end = data_fingerprint(workspace / "data" / "raw" / "train")
        test_fingerprint_end = data_fingerprint(workspace / "data" / "raw" / "test")
        if (
            input_hashes_start != input_hashes_end
            or train_fingerprint_start != train_fingerprint_end
            or test_fingerprint_start != test_fingerprint_end
        ):
            raise ValueError("v008 source or input data changed during training")

        output_files = {
            "feature": feature_path,
            **{f"report_{name}": path for name, path in report_paths.items()},
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
            "train_data_fingerprint_end": train_fingerprint_end,
            "test_data_fingerprint_end": test_fingerprint_end,
            "output_sha256": output_sha256,
            "decision": decision,
            "cv_score": overall,
            "folds_better_than_v006": folds_better,
            "final_weights": dict(zip(STREAM_COLUMNS, final_fit.weights.tolist())),
            "training_inference_failures": 0,
            "nonfinite_failures": 0,
            "donor_fold_violations": donor_fold_violations,
            "self_donor_violations": self_donor_violations,
            "test_failures": len(visible_test_failures),
            "test_predictions_saved": test_values is not None,
        }
        _atomic_replace_json(run_path, run_payload)
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
                _atomic_replace_json(run_path, failure_payload)
            except Exception:
                # The already-fsynced running state remains as crash evidence.
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
    smoke_parser = subparsers.add_parser("smoke", help="Run fixed target-free six-well gate")
    smoke_parser.add_argument("--wells", type=int, default=SMOKE_WELLS)
    train_parser = subparsers.add_parser("train", help="Run immutable nested v008 experiment")
    train_parser.add_argument("--version", default="v008")
    train_parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "smoke":
        payload = run_smoke(args.workspace, wells=args.wells)
        print(json.dumps(payload, indent=2))
        return 0 if payload["eligible_for_full_cv"] else 1
    if args.command == "train":
        path = train(args.workspace, version=args.version, folds=args.folds)
        print(path)
        return 0
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
