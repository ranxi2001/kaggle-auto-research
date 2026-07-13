#!/usr/bin/env python3
"""Leakage-aware grouped baseline for ROGII wellbore geology prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        del args, kwargs

        def decorator(function):
            return function

        return decorator


TARGET_COLUMN = "TVT"
INPUT_TARGET_COLUMN = "TVT_input"
DEPTH_COLUMN = "MD"
HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"

FEATURE_COLUMNS = [
    "baseline_tvt",
    "MD",
    "X",
    "Y",
    "Z",
    "GR",
    "previous_tvt",
    "distance_from_previous_md",
    "dX_dMD",
    "dY_dMD",
    "dZ_dMD",
    "gr_rolling_mean_5",
    "gr_rolling_std_5",
    "gr_rolling_mean_21",
    "gr_rolling_std_21",
    "typewell_gr_at_baseline",
    "typewell_gr_gradient",
    "gr_typewell_delta",
    "md_since_anchor",
    "x_since_anchor",
    "y_since_anchor",
    "z_since_anchor",
    "evaluation_fraction",
    "known_rows",
    "evaluation_rows",
    "anchor_gr",
    "known_gr_mean",
    "known_gr_std",
    "evaluation_gr_mean",
    "evaluation_gr_std",
    "typewell_tvt_min",
    "typewell_tvt_max",
    "typewell_tvt_range",
    "typewell_gr_mean",
    "typewell_gr_std",
    "beam_tvt",
    "beam_delta_from_last",
    "beam_delta_from_baseline",
]

BEAM_CONFIGS = [
    {"name": "balanced", "beam_size": 10, "move_cost": 20.0, "emission_scale": 144.0, "smooth_radius": 2},
    {"name": "loose", "beam_size": 10, "move_cost": 8.0, "emission_scale": 64.0, "smooth_radius": 2},
    {"name": "conservative", "beam_size": 8, "move_cost": 35.0, "emission_scale": 220.0, "smooth_radius": 1},
    {"name": "smooth", "beam_size": 10, "move_cost": 14.0, "emission_scale": 90.0, "smooth_radius": 5},
    {"name": "very_loose", "beam_size": 20, "move_cost": 4.0, "emission_scale": 36.0, "smooth_radius": 3},
    {"name": "middle", "beam_size": 12, "move_cost": 12.0, "emission_scale": 100.0, "smooth_radius": 3},
]


@dataclass
class BuildStats:
    wells: int = 0
    rows: int = 0
    evaluation_rows: int = 0


def well_id_from_path(path: Path) -> str:
    if not path.name.endswith(HORIZONTAL_SUFFIX):
        raise ValueError(f"Unexpected horizontal-well filename: {path.name}")
    return path.name[: -len(HORIZONTAL_SUFFIX)]


def interpolate_known_values(values: pd.Series, coordinate: pd.Series) -> np.ndarray:
    """Interpolate values over coordinate using only finite observed values."""
    value_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    coordinate_array = pd.to_numeric(coordinate, errors="coerce").to_numpy(dtype=float)
    fallback_coordinate = np.arange(len(values), dtype=float)
    coordinate_array = np.where(np.isfinite(coordinate_array), coordinate_array, fallback_coordinate)
    observed = np.isfinite(value_array)
    if not observed.any():
        raise ValueError(f"{INPUT_TARGET_COLUMN} has no observed values")

    known = pd.DataFrame(
        {"coordinate": coordinate_array[observed], "value": value_array[observed]}
    )
    known = known.groupby("coordinate", as_index=False, sort=True)["value"].mean()
    if len(known) == 1:
        return np.full(len(values), known["value"].iloc[0], dtype=float)
    return np.interp(
        coordinate_array,
        known["coordinate"].to_numpy(dtype=float),
        known["value"].to_numpy(dtype=float),
    )


@njit(cache=True)
def _beam_path(
    horizontal_gr: np.ndarray,
    typewell_gr: np.ndarray,
    start_index: int,
    beam_size: int,
    move_cost: float,
    emission_scale: float,
    max_step: int,
) -> np.ndarray:
    """Find a GR-matching path through the typewell grid."""
    n_rows = len(horizontal_gr)
    n_reference = len(typewell_gr)
    max_candidates = beam_size * (2 * max_step + 1)
    beam_indices = np.empty(beam_size, dtype=np.int64)
    beam_costs = np.full(beam_size, 1e30)
    beam_indices[0] = start_index
    beam_costs[0] = 0.0
    beam_count = 1
    history_indices = np.zeros((n_rows, beam_size), dtype=np.int64)
    history_parents = np.zeros((n_rows, beam_size), dtype=np.int64)

    for row_index in range(n_rows):
        candidate_indices = np.empty(max_candidates, dtype=np.int64)
        candidate_costs = np.full(max_candidates, 1e30)
        candidate_parents = np.zeros(max_candidates, dtype=np.int64)
        candidate_count = 0
        for parent_index in range(beam_count):
            current_index = beam_indices[parent_index]
            current_cost = beam_costs[parent_index]
            for step in range(-max_step, max_step + 1):
                next_index = current_index + step
                if next_index < 0 or next_index >= n_reference:
                    continue
                emission = horizontal_gr[row_index] - typewell_gr[next_index]
                total_cost = (
                    current_cost
                    + emission * emission / emission_scale
                    + move_cost * abs(step)
                )
                existing = -1
                for candidate_index in range(candidate_count):
                    if candidate_indices[candidate_index] == next_index:
                        existing = candidate_index
                        break
                if existing >= 0:
                    if total_cost < candidate_costs[existing]:
                        candidate_costs[existing] = total_cost
                        candidate_parents[existing] = parent_index
                elif candidate_count < max_candidates:
                    candidate_indices[candidate_count] = next_index
                    candidate_costs[candidate_count] = total_cost
                    candidate_parents[candidate_count] = parent_index
                    candidate_count += 1

        kept = min(beam_size, candidate_count)
        for position in range(kept):
            best_position = position
            for candidate_position in range(position + 1, candidate_count):
                if candidate_costs[candidate_position] < candidate_costs[best_position]:
                    best_position = candidate_position
            if best_position != position:
                candidate_indices[position], candidate_indices[best_position] = (
                    candidate_indices[best_position],
                    candidate_indices[position],
                )
                candidate_costs[position], candidate_costs[best_position] = (
                    candidate_costs[best_position],
                    candidate_costs[position],
                )
                candidate_parents[position], candidate_parents[best_position] = (
                    candidate_parents[best_position],
                    candidate_parents[position],
                )
        for position in range(kept):
            beam_indices[position] = candidate_indices[position]
            beam_costs[position] = candidate_costs[position]
            history_indices[row_index, position] = candidate_indices[position]
            history_parents[row_index, position] = candidate_parents[position]
        beam_count = kept

    best = 0
    for position in range(1, beam_count):
        if beam_costs[position] < beam_costs[best]:
            best = position
    path = np.zeros(n_rows, dtype=np.int64)
    parent = best
    for row_index in range(n_rows - 1, -1, -1):
        path[row_index] = history_indices[row_index, parent]
        parent = history_parents[row_index, parent]
    return path


def beam_prediction(
    frame: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict,
    max_step: int = 2,
) -> np.ndarray:
    """Predict missing suffix TVT by constrained GR sequence alignment."""
    input_target = _numeric_column(frame, INPUT_TARGET_COLUMN).to_numpy(dtype=float)
    evaluation_indices = np.flatnonzero(~np.isfinite(input_target))
    observed_indices = np.flatnonzero(np.isfinite(input_target))
    if not len(evaluation_indices):
        return input_target.copy()
    if not len(observed_indices):
        raise ValueError(f"{INPUT_TARGET_COLUMN} has no observed prefix")
    if evaluation_indices[0] <= observed_indices[-1]:
        raise ValueError("Beam baseline requires one missing suffix after the observed prefix")

    reference = pd.DataFrame(
        {
            "TVT": pd.to_numeric(typewell["TVT"], errors="coerce"),
            "GR": pd.to_numeric(typewell["GR"], errors="coerce"),
        }
    ).dropna()
    reference = reference.groupby("TVT", as_index=False, sort=True)["GR"].mean()
    if len(reference) < 2:
        raise ValueError("Typewell requires at least two finite TVT/GR rows")

    horizontal_gr = _numeric_column(frame, "GR").interpolate(limit_direction="both")
    radius = int(config["smooth_radius"])
    if radius > 0:
        horizontal_gr = horizontal_gr.rolling(
            window=2 * radius + 1,
            center=True,
            min_periods=1,
        ).mean()
    evaluation_gr = horizontal_gr.iloc[evaluation_indices].to_numpy(dtype=float)
    if not np.isfinite(evaluation_gr).all():
        evaluation_gr = np.nan_to_num(
            evaluation_gr,
            nan=float(np.nanmedian(reference["GR"])),
        )

    reference_tvt = reference["TVT"].to_numpy(dtype=float)
    reference_gr = reference["GR"].to_numpy(dtype=float)
    last_tvt = input_target[observed_indices[-1]]
    start_index = int(np.argmin(np.abs(reference_tvt - last_tvt)))
    path = _beam_path(
        evaluation_gr,
        reference_gr,
        start_index,
        int(config["beam_size"]),
        float(config["move_cost"]),
        float(config["emission_scale"]),
        max_step,
    )
    prediction = input_target.copy()
    prediction[evaluation_indices] = reference_tvt[path]
    return prediction


def scan_beam_configs(
    workspace: Path,
    max_wells: int = 0,
    blends: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75),
    output_name: str = "beam_scan.csv",
) -> Path:
    """Score generic beam configurations without using target for path selection."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Beam scan requires numba; run with the workspace beam dependencies")
    train_root = workspace / "data" / "raw" / "train"
    paths = sorted(train_root.glob(f"*{HORIZONTAL_SUFFIX}"))
    if max_wells > 0:
        paths = paths[:max_wells]
    totals = {
        (config["name"], blend): [0.0, 0, 0]
        for config in BEAM_CONFIGS
        for blend in blends
    }
    totals[("last_known", 1.0)] = [0.0, 0, 0]
    detail_records: list[dict] = []
    for horizontal_path in paths:
        well_id = well_id_from_path(horizontal_path)
        typewell_path = horizontal_path.with_name(f"{well_id}{TYPEWELL_SUFFIX}")
        frame = pd.read_csv(horizontal_path)
        typewell = pd.read_csv(typewell_path)
        evaluation = frame[INPUT_TARGET_COLUMN].isna().to_numpy()
        target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
        last_tvt = float(frame.loc[~evaluation, INPUT_TARGET_COLUMN].iloc[-1])
        last_known_error = last_tvt - target[evaluation]
        last_known_squared_error = float(np.dot(last_known_error, last_known_error))
        totals[("last_known", 1.0)][0] += last_known_squared_error
        totals[("last_known", 1.0)][1] += len(last_known_error)
        detail_records.append(
            {
                "well": well_id,
                "config": "last_known",
                "blend_to_last_tvt": 1.0,
                "squared_error": last_known_squared_error,
                "rows": len(last_known_error),
                "failed": False,
            }
        )
        for config in BEAM_CONFIGS:
            try:
                beam = beam_prediction(frame, typewell, config)[evaluation]
            except Exception:
                for blend in blends:
                    totals[(config["name"], blend)][2] += 1
                    detail_records.append(
                        {
                            "well": well_id,
                            "config": config["name"],
                            "blend_to_last_tvt": blend,
                            "squared_error": None,
                            "rows": len(last_known_error),
                            "failed": True,
                        }
                    )
                continue
            for blend in blends:
                prediction = (1.0 - blend) * beam + blend * last_tvt
                error = prediction - target[evaluation]
                squared_error = float(np.dot(error, error))
                totals[(config["name"], blend)][0] += squared_error
                totals[(config["name"], blend)][1] += len(error)
                detail_records.append(
                    {
                        "well": well_id,
                        "config": config["name"],
                        "blend_to_last_tvt": blend,
                        "squared_error": squared_error,
                        "rows": len(error),
                        "failed": False,
                    }
                )

    records = []
    for config in BEAM_CONFIGS:
        for blend in blends:
            squared_error, rows, failures = totals[(config["name"], blend)]
            records.append(
                {
                    "config": config["name"],
                    "blend_to_last_tvt": blend,
                    "rmse": float(np.sqrt(squared_error / rows)) if rows else None,
                    "rows": rows,
                    "wells": len(paths) - failures,
                    "failures": failures,
                    **{key: value for key, value in config.items() if key != "name"},
                }
            )
    last_squared_error, last_rows, _ = totals[("last_known", 1.0)]
    records.append(
        {
            "config": "last_known",
            "blend_to_last_tvt": 1.0,
            "rmse": float(np.sqrt(last_squared_error / last_rows)),
            "rows": last_rows,
            "wells": len(paths),
            "failures": 0,
            "beam_size": None,
            "move_cost": None,
            "emission_scale": None,
            "smooth_radius": None,
        }
    )
    report = pd.DataFrame(records).sort_values("rmse", na_position="last")
    report_path = workspace / "reports" / output_name
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    detail_path = report_path.with_name(f"{report_path.stem}_by_well.parquet")
    if detail_path.exists():
        raise FileExistsError(f"Refusing to overwrite {detail_path}")
    report.to_csv(report_path, index=False)
    pd.DataFrame(detail_records).to_parquet(detail_path, index=False)
    return report_path


def nested_beam_selection(details_path: Path, output_path: Path, n_splits: int = 5) -> dict:
    """Select beam parameters on training wells and score unseen wells per fold."""
    details = pd.read_parquet(details_path)
    details = details.loc[~details["failed"] & details["squared_error"].notna()].copy()
    wells = np.array(sorted(details["well"].unique()))
    if len(wells) < n_splits:
        raise ValueError(f"Need at least {n_splits} wells, found {len(wells)}")
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    pooled_squared_error = 0.0
    pooled_rows = 0
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(wells, groups=wells), start=1
    ):
        train_wells = set(wells[train_index])
        validation_wells = set(wells[validation_index])
        train = details.loc[details["well"].isin(train_wells)]
        candidates = (
            train.groupby(["config", "blend_to_last_tvt"], as_index=False)
            .agg(squared_error=("squared_error", "sum"), rows=("rows", "sum"))
        )
        candidates["rmse"] = np.sqrt(candidates["squared_error"] / candidates["rows"])
        selected = candidates.sort_values(
            ["rmse", "config", "blend_to_last_tvt"]
        ).iloc[0]
        validation = details.loc[
            details["well"].isin(validation_wells)
            & details["config"].eq(selected["config"])
            & details["blend_to_last_tvt"].eq(selected["blend_to_last_tvt"])
        ]
        fold_squared_error = float(validation["squared_error"].sum())
        fold_rows = int(validation["rows"].sum())
        pooled_squared_error += fold_squared_error
        pooled_rows += fold_rows
        folds.append(
            {
                "fold": fold,
                "selected_config": selected["config"],
                "selected_blend_to_last_tvt": float(selected["blend_to_last_tvt"]),
                "train_rmse": float(selected["rmse"]),
                "validation_rmse": float(np.sqrt(fold_squared_error / fold_rows)),
                "train_wells": len(train_wells),
                "validation_wells": len(validation_wells),
                "validation_rows": fold_rows,
            }
        )
    result = {
        "metric": "rmse",
        "direction": "minimize",
        "splitter": f"GroupKFold(n_splits={n_splits}, group=well_id)",
        "folds": folds,
        "pooled_nested_rmse": float(np.sqrt(pooled_squared_error / pooled_rows)),
        "rows": pooled_rows,
        "wells": len(wells),
    }
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _beam_config(name: str) -> dict:
    for config in BEAM_CONFIGS:
        if config["name"] == name:
            return config
    raise ValueError(f"Unknown beam config: {name}")


def _typewell_tvt_median(typewell_path: Path) -> float | None:
    try:
        typewell = pd.read_csv(
            typewell_path, usecols=lambda column: column == TARGET_COLUMN
        )
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError):
        return None
    if TARGET_COLUMN not in typewell:
        return None
    values = pd.to_numeric(typewell[TARGET_COLUMN], errors="coerce").dropna()
    return float(values.median()) if not values.empty else None


def _well_fallback_tvt(horizontal_path: Path, default: float) -> float:
    try:
        horizontal = pd.read_csv(
            horizontal_path, usecols=lambda column: column == INPUT_TARGET_COLUMN
        )
        if INPUT_TARGET_COLUMN in horizontal:
            observed = pd.to_numeric(
                horizontal[INPUT_TARGET_COLUMN], errors="coerce"
            ).dropna()
            if not observed.empty:
                return float(observed.iloc[-1])
    except (OSError, ValueError, pd.errors.ParserError):
        pass

    well_id = well_id_from_path(horizontal_path)
    typewell_fallback = _typewell_tvt_median(
        horizontal_path.with_name(f"{well_id}{TYPEWELL_SUFFIX}")
    )
    return typewell_fallback if typewell_fallback is not None else default


def sample_fallback_predictions(test_root: Path, sample: pd.DataFrame) -> pd.Series:
    """Return finite sample-aligned fallbacks before model-specific overrides."""
    typewell_medians = [
        value
        for path in sorted(test_root.glob(f"*{TYPEWELL_SUFFIX}"))
        if (value := _typewell_tvt_median(path)) is not None
    ]
    global_fallback = float(np.median(typewell_medians)) if typewell_medians else 0.0
    predictions = pd.Series(
        np.full(len(sample), global_fallback, dtype=float),
        index=sample["id"].astype(str),
    )
    sample_wells = sample["id"].astype(str).str.rsplit("_", n=1).str[0]
    for horizontal_path in sorted(test_root.glob(f"*{HORIZONTAL_SUFFIX}")):
        well_id = well_id_from_path(horizontal_path)
        predictions.loc[sample.loc[sample_wells.eq(well_id), "id"].astype(str)] = (
            _well_fallback_tvt(horizontal_path, global_fallback)
        )
    if not np.isfinite(predictions.to_numpy(dtype=float)).all():
        raise ValueError("Fallback predictions contain non-finite values")
    return predictions


def _beam_rows_for_well(
    horizontal_path: Path,
    config_name: str,
    blend_to_last_tvt: float,
    require_target: bool,
) -> tuple[pd.DataFrame, bool]:
    well_id = well_id_from_path(horizontal_path)
    frame = pd.read_csv(horizontal_path)
    evaluation = frame[INPUT_TARGET_COLUMN].isna().to_numpy()
    row_indices = np.flatnonzero(evaluation)
    observed_values = pd.to_numeric(
        frame.loc[~evaluation, INPUT_TARGET_COLUMN], errors="coerce"
    ).dropna()
    typewell_path = horizontal_path.with_name(f"{well_id}{TYPEWELL_SUFFIX}")
    typewell = pd.read_csv(typewell_path)
    if not observed_values.empty:
        fallback = float(observed_values.iloc[-1])
    else:
        fallback = float(pd.to_numeric(typewell["TVT"], errors="coerce").median())
    failed = False
    try:
        beam = beam_prediction(frame, typewell, _beam_config(config_name))[evaluation]
        prediction = (1.0 - blend_to_last_tvt) * beam + blend_to_last_tvt * fallback
    except Exception:
        prediction = np.full(len(row_indices), fallback, dtype=float)
        failed = True
    rows = pd.DataFrame(
        {
            "_id": [f"{well_id}_{index}" for index in row_indices],
            "_well_id": well_id,
            "_row_index": row_indices,
            "prediction": prediction,
        }
    )
    if require_target:
        if TARGET_COLUMN not in frame:
            raise ValueError(f"{horizontal_path.name} does not contain {TARGET_COLUMN}")
        rows["_target"] = pd.to_numeric(
            frame.loc[evaluation, TARGET_COLUMN], errors="coerce"
        ).to_numpy(dtype=float)
    return rows, failed


def _residual_rows_for_well(
    path: Path,
    require_target: bool,
    beam_config: str,
    beam_blend: float,
) -> tuple[pd.DataFrame, int, bool]:
    features = build_well_features(path, require_target=require_target)
    raw = pd.read_csv(path)
    evaluation = raw[INPUT_TARGET_COLUMN].isna().to_numpy()
    typewell = pd.read_csv(path.with_name(f"{well_id_from_path(path)}{TYPEWELL_SUFFIX}"))
    observed = pd.to_numeric(
        raw.loc[~evaluation, INPUT_TARGET_COLUMN], errors="coerce"
    ).dropna()
    if observed.empty:
        raise ValueError(f"{path.name} has no finite {INPUT_TARGET_COLUMN} prefix")
    last_tvt = float(observed.iloc[-1])
    beam_failed = False
    try:
        raw_beam = beam_prediction(raw, typewell, _beam_config(beam_config))[evaluation]
    except Exception:
        raw_beam = np.full(evaluation.sum(), last_tvt, dtype=float)
        beam_failed = True
    selected_beam = (1.0 - beam_blend) * raw_beam + beam_blend * last_tvt
    features["beam_tvt"] = selected_beam
    features["beam_delta_from_last"] = selected_beam - last_tvt
    features["beam_delta_from_baseline"] = selected_beam - features["baseline_tvt"]
    features[FEATURE_COLUMNS] = features[FEATURE_COLUMNS].astype(np.float32)
    return features, len(raw), beam_failed


def build_residual_rows(
    root: Path,
    require_target: bool,
    beam_config: str = "very_loose",
    beam_blend: float = 0.7,
) -> tuple[pd.DataFrame, BuildStats, int]:
    """Build target-independent row features and attach target only after feature creation."""
    paths = sorted(root.glob(f"*{HORIZONTAL_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(f"No horizontal-well CSV files found under {root}")
    frames = []
    stats = BuildStats(wells=len(paths))
    failures = 0
    for index, path in enumerate(paths, start=1):
        features, raw_rows, beam_failed = _residual_rows_for_well(
            path,
            require_target=require_target,
            beam_config=beam_config,
            beam_blend=beam_blend,
        )
        failures += int(beam_failed)
        stats.rows += raw_rows
        stats.evaluation_rows += len(features)
        frames.append(features)
        if index % 100 == 0 or index == len(paths):
            print(f"feature progress: {index}/{len(paths)} wells", flush=True)
    return pd.concat(frames, ignore_index=True), stats, failures


def next_feature_path(workspace: Path) -> Path:
    features_root = workspace / "data" / "features"
    features_root.mkdir(parents=True, exist_ok=True)
    versions = [int(path.stem[1:]) for path in features_root.glob("v[0-9][0-9][0-9].parquet")]
    version = max(versions, default=0) + 1
    return features_root / f"v{version:03d}.parquet"


def _write_sample_aligned_submission(
    workspace: Path,
    model_dir: Path,
    predictions: pd.DataFrame,
    suffix: str,
    metadata: dict,
) -> Path:
    predictions_by_id = pd.Series(
        predictions["prediction"].to_numpy(dtype=float), index=predictions["_id"]
    )
    sample_path = workspace / "data" / "raw" / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"Unexpected sample submission columns: {list(sample.columns)}")
    if sample["id"].duplicated().any() or predictions_by_id.index.duplicated().any():
        raise ValueError("Submission IDs must be unique")
    missing_ids = sample.loc[~sample["id"].isin(predictions_by_id.index), "id"]
    if not missing_ids.empty:
        raise ValueError(f"Missing predictions for {len(missing_ids)} sample IDs")
    submission = sample.copy()
    submission["tvt"] = submission["id"].map(predictions_by_id)
    if not np.isfinite(submission["tvt"].to_numpy(dtype=float)).all():
        raise ValueError("Submission predictions contain non-finite values")
    submission_path = workspace / "submissions" / f"sub_{model_dir.name}_{suffix}.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    if submission_path.exists():
        raise FileExistsError(f"Refusing to overwrite {submission_path}")
    submission.to_csv(submission_path, index=False)
    np.save(model_dir / "test_preds.npy", submission["tvt"].to_numpy(dtype=float))
    submission_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return submission_path


def train_residual(
    workspace: Path,
    n_splits: int = 5,
    n_estimators: int = 600,
    learning_rate: float = 0.03,
) -> Path:
    """Train a strongly regularized LightGBM residual model with well GroupKFold."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Residual training requires the workspace beam dependencies")
    import lightgbm as lgb

    train_root = workspace / "data" / "raw" / "train"
    model_dir = next_model_dir(workspace)
    feature_path = next_feature_path(workspace)
    started = time.time()
    run_path = model_dir / "run.json"
    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": 31,
        "min_child_samples": 500,
        "max_bin": 127,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    run = {
        "version": model_dir.name,
        "status": "running",
        "parent_run": "v002",
        "template": "rogii-lightgbm-residual",
        "command": (
            "uv run --with-requirements requirements-beam.txt python "
            "scripts/rogii_baseline.py train-residual"
        ),
        "git_commit": _git_commit(workspace.parents[1]),
        "params": params,
        "random_seed": 42,
        "cv_splitter": f"GroupKFold(n_splits={n_splits}, group=well_id)",
        "metric": "rmse",
        "metric_direction": "minimize",
        "target_column": TARGET_COLUMN,
        "id_column": "id",
        "feature_path": str(feature_path.relative_to(workspace)),
        "data_fingerprint": "",
        "runtime_seconds": None,
        "error": None,
    }
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

    try:
        print("building leakage-safe residual features", flush=True)
        rows, stats, beam_failures = build_residual_rows(train_root, require_target=True)
        rows[["_id", "_well_id", "_row_index", *FEATURE_COLUMNS]].to_parquet(
            feature_path, index=False
        )
        features = rows[FEATURE_COLUMNS]
        target = rows["_target"].to_numpy(dtype=float)
        baseline = rows["baseline_tvt"].to_numpy(dtype=float)
        residual_target = target - baseline
        groups = rows["_well_id"].to_numpy()
        splitter = GroupKFold(n_splits=n_splits)
        oof = np.zeros(len(rows), dtype=float)
        fold_scores = []
        models = []
        importances = []
        for fold, (train_index, validation_index) in enumerate(
            splitter.split(features, groups=groups), start=1
        ):
            model = lgb.LGBMRegressor(**params)
            model.fit(
                features.iloc[train_index],
                residual_target[train_index],
                eval_set=[(features.iloc[validation_index], residual_target[validation_index])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            residual_prediction = model.predict(
                features.iloc[validation_index],
                num_iteration=model.best_iteration_,
            )
            oof[validation_index] = baseline[validation_index] + residual_prediction
            fold_scores.append(rmse(target[validation_index], oof[validation_index]))
            models.append(model)
            importances.append(model.feature_importances_.astype(float))
            print(
                f"fold {fold}/{n_splits}: rmse={fold_scores[-1]:.6f} "
                f"best_iteration={model.best_iteration_}",
                flush=True,
            )

        np.save(model_dir / "oof_preds.npy", oof)
        rows[["_id", "_well_id", "_row_index", "_target"]].to_parquet(
            model_dir / "oof_rows.parquet", index=False
        )
        (model_dir / "feature_list.txt").write_text(
            "\n".join(FEATURE_COLUMNS) + "\n", encoding="utf-8"
        )
        importance = np.mean(np.stack(importances), axis=0)
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importance}).sort_values(
            "importance", ascending=False
        ).to_csv(model_dir / "importance.csv", index=False)
        with (model_dir / "models.pkl").open("wb") as handle:
            pickle.dump(models, handle)
        scores = {
            "metric": "rmse",
            "direction": "minimize",
            "fold_scores": fold_scores,
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
            "overall": rmse(target, oof),
            "valid_rows": len(rows),
            "valid_wells": int(rows["_well_id"].nunique()),
            "beam_feature_fallback_wells": beam_failures,
        }
        (model_dir / "cv_scores.json").write_text(
            json.dumps(scores, indent=2), encoding="utf-8"
        )
        run.update(
            {
                "status": "completed",
                "data_fingerprint": data_fingerprint(train_root),
                "runtime_seconds": round(time.time() - started, 3),
                "data_stats": {**asdict(stats), "beam_feature_fallback_wells": beam_failures},
            }
        )
    except Exception as exc:
        run.update(
            {
                "status": "failed",
                "runtime_seconds": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
        raise

    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    return model_dir


def predict_residual(workspace: Path, model_dir: Path) -> Path:
    """Generate visible-test predictions using all residual fold models."""
    with (model_dir / "models.pkl").open("rb") as handle:
        models = pickle.load(handle)
    test_root = workspace / "data" / "raw" / "test"
    sample = pd.read_csv(workspace / "data" / "raw" / "sample_submission.csv")
    if list(sample.columns) != ["id", "tvt"] or sample["id"].duplicated().any():
        raise ValueError("Unexpected sample submission contract")
    predictions_by_id = sample_fallback_predictions(test_root, sample)
    beam_failures = 0
    feature_fallback_wells = 0
    test_paths = sorted(test_root.glob(f"*{HORIZONTAL_SUFFIX}"))
    if not test_paths:
        raise FileNotFoundError(f"No horizontal-well CSV files found under {test_root}")
    for path in test_paths:
        try:
            rows, _, beam_failed = _residual_rows_for_well(
                path,
                require_target=False,
                beam_config="very_loose",
                beam_blend=0.7,
            )
            features = rows[FEATURE_COLUMNS]
            baseline = rows["baseline_tvt"].to_numpy(dtype=float)
            residual_prediction = np.zeros(len(rows), dtype=float)
            for model in models:
                residual_prediction += model.predict(
                    features, num_iteration=model.best_iteration_
                )
            well_prediction = baseline + residual_prediction / len(models)
            if not np.isfinite(well_prediction).all():
                raise ValueError("Residual prediction contains non-finite values")
            well_predictions = pd.Series(well_prediction, index=rows["_id"].astype(str))
            shared_ids = predictions_by_id.index.intersection(well_predictions.index)
            predictions_by_id.loc[shared_ids] = well_predictions.loc[shared_ids]
            beam_failures += int(beam_failed)
        except Exception as exc:
            feature_fallback_wells += 1
            print(f"residual fallback for {path.name}: {type(exc).__name__}: {exc}", flush=True)
    rows = pd.DataFrame(
        {"_id": predictions_by_id.index, "prediction": predictions_by_id.to_numpy(float)}
    )
    scores = json.loads((model_dir / "cv_scores.json").read_text(encoding="utf-8"))
    metadata = {
        "source_model_versions": [model_dir.name],
        "ensemble_weights": None,
        "local_cv_score": scores["overall"],
        "fold_mean_score": scores["mean"],
        "metric": "rmse",
        "generated_command": (
            "uv run --with-requirements requirements-beam.txt python "
            f"scripts/rogii_baseline.py predict-residual --model-dir models/{model_dir.name}"
        ),
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "dry_run": {
            "valid": True,
            "checks": ["sample ID order", "unique IDs", "finite predictions", "row count"],
            "beam_feature_fallback_wells": beam_failures,
            "feature_fallback_wells": feature_fallback_wells,
        },
        "kaggle_submission_id": None,
        "lb_score": None,
        "rank": None,
        "submitted": False,
        "submission_mode": "notebook",
    }
    return _write_sample_aligned_submission(
        workspace, model_dir, rows, "residual_lgbm", metadata
    )


def train_beam(
    workspace: Path,
    selection_path: Path,
    final_config: str = "very_loose",
    final_blend: float = 0.7,
) -> Path:
    """Persist nested-CV beam predictions and a final hidden-test configuration."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Beam training requires numba; run with the workspace beam dependencies")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    train_root = workspace / "data" / "raw" / "train"
    paths_by_well = {
        well_id_from_path(path): path
        for path in sorted(train_root.glob(f"*{HORIZONTAL_SUFFIX}"))
    }
    wells = np.array(sorted(paths_by_well))
    n_splits = len(selection["folds"])
    splitter = GroupKFold(n_splits=n_splits)
    model_dir = next_model_dir(workspace)
    started = time.time()
    run_path = model_dir / "run.json"
    run = {
        "version": model_dir.name,
        "status": "running",
        "parent_run": "v001",
        "template": "rogii-gr-beam-alignment",
        "command": (
            "uv run --with-requirements requirements-beam.txt python "
            "scripts/rogii_baseline.py train-beam"
        ),
        "git_commit": _git_commit(workspace.parents[1]),
        "params": {
            "selection_path": str(selection_path.relative_to(workspace)),
            "fold_selections": selection["folds"],
            "final_config": final_config,
            "final_blend_to_last_tvt": final_blend,
        },
        "random_seed": 42,
        "cv_splitter": f"GroupKFold(n_splits={n_splits}, group=well_id)",
        "metric": "rmse",
        "metric_direction": "minimize",
        "target_column": TARGET_COLUMN,
        "id_column": "id",
        "data_fingerprint": "",
        "runtime_seconds": None,
        "error": None,
    }
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

    try:
        oof_frames = []
        fold_scores = []
        failures = 0
        for fold, (_, validation_index) in enumerate(
            splitter.split(wells, groups=wells), start=1
        ):
            fold_selection = selection["folds"][fold - 1]
            config_name = fold_selection["selected_config"]
            blend = float(fold_selection["selected_blend_to_last_tvt"])
            fold_frames = []
            for well_id in wells[validation_index]:
                rows, failed = _beam_rows_for_well(
                    paths_by_well[well_id],
                    config_name,
                    blend,
                    require_target=True,
                )
                rows["fold"] = fold
                fold_frames.append(rows)
                failures += int(failed)
            fold_frame = pd.concat(fold_frames, ignore_index=True)
            fold_scores.append(
                rmse(
                    fold_frame["_target"].to_numpy(dtype=float),
                    fold_frame["prediction"].to_numpy(dtype=float),
                )
            )
            oof_frames.append(fold_frame)

        oof = pd.concat(oof_frames, ignore_index=True)
        predictions = oof["prediction"].to_numpy(dtype=float)
        target = oof["_target"].to_numpy(dtype=float)
        overall = rmse(target, predictions)
        np.save(model_dir / "oof_preds.npy", predictions)
        oof[["_id", "_well_id", "_row_index", "_target", "fold"]].to_parquet(
            model_dir / "oof_rows.parquet", index=False
        )
        input_features = [
            "horizontal.MD",
            "horizontal.GR",
            "horizontal.TVT_input",
            "typewell.TVT",
            "typewell.GR",
        ]
        (model_dir / "feature_list.txt").write_text(
            "\n".join(input_features) + "\n", encoding="utf-8"
        )
        pd.DataFrame(columns=["feature", "importance"]).to_csv(
            model_dir / "importance.csv", index=False
        )
        model_config = {
            "model": "gr_beam_alignment",
            "final_config": final_config,
            "final_blend_to_last_tvt": final_blend,
            "fold_selections": selection["folds"],
        }
        with (model_dir / "model.pkl").open("wb") as handle:
            pickle.dump(model_config, handle)
        scores = {
            "metric": "rmse",
            "direction": "minimize",
            "fold_scores": fold_scores,
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
            "overall": overall,
            "valid_rows": len(oof),
            "valid_wells": len(wells),
            "fallback_wells": failures,
            "selection_protocol": "nested GroupKFold parameter selection",
        }
        (model_dir / "cv_scores.json").write_text(
            json.dumps(scores, indent=2), encoding="utf-8"
        )
        run.update(
            {
                "status": "completed",
                "data_fingerprint": data_fingerprint(train_root),
                "runtime_seconds": round(time.time() - started, 3),
                "data_stats": {
                    "wells": len(wells),
                    "evaluation_rows": len(oof),
                    "fallback_wells": failures,
                },
            }
        )
    except Exception as exc:
        run.update(
            {
                "status": "failed",
                "runtime_seconds": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
        raise

    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    return model_dir


def predict_beam(workspace: Path, model_dir: Path) -> Path:
    """Generate sample-aligned predictions from a persisted beam configuration."""
    with (model_dir / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    test_root = workspace / "data" / "raw" / "test"
    prediction_frames = []
    failures = 0
    for path in sorted(test_root.glob(f"*{HORIZONTAL_SUFFIX}")):
        rows, failed = _beam_rows_for_well(
            path,
            model["final_config"],
            float(model["final_blend_to_last_tvt"]),
            require_target=False,
        )
        prediction_frames.append(rows)
        failures += int(failed)
    if not prediction_frames:
        raise FileNotFoundError(f"No test wells found under {test_root}")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions_by_id = pd.Series(
        predictions["prediction"].to_numpy(dtype=float), index=predictions["_id"]
    )
    sample_path = workspace / "data" / "raw" / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"Unexpected sample submission columns: {list(sample.columns)}")
    if sample["id"].duplicated().any() or predictions_by_id.index.duplicated().any():
        raise ValueError("Submission IDs must be unique")
    missing_ids = sample.loc[~sample["id"].isin(predictions_by_id.index), "id"]
    if not missing_ids.empty:
        raise ValueError(f"Missing predictions for {len(missing_ids)} sample IDs")
    submission = sample.copy()
    submission["tvt"] = submission["id"].map(predictions_by_id)
    if not np.isfinite(submission["tvt"].to_numpy(dtype=float)).all():
        raise ValueError("Submission predictions contain non-finite values")

    submission_path = workspace / "submissions" / f"sub_{model_dir.name}_beam.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    if submission_path.exists():
        raise FileExistsError(f"Refusing to overwrite {submission_path}")
    submission.to_csv(submission_path, index=False)
    np.save(model_dir / "test_preds.npy", submission["tvt"].to_numpy(dtype=float))
    scores = json.loads((model_dir / "cv_scores.json").read_text(encoding="utf-8"))
    metadata = {
        "source_model_versions": [model_dir.name],
        "ensemble_weights": None,
        "local_cv_score": scores["overall"],
        "fold_mean_score": scores["mean"],
        "metric": "rmse",
        "generated_command": (
            "uv run --with-requirements requirements-beam.txt python "
            f"scripts/rogii_baseline.py predict-beam --model-dir models/{model_dir.name}"
        ),
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "dry_run": {
            "valid": True,
            "checks": ["sample ID order", "unique IDs", "finite predictions", "row count"],
            "fallback_wells": failures,
        },
        "kaggle_submission_id": None,
        "lb_score": None,
        "rank": None,
        "submitted": False,
        "submission_mode": "notebook",
    }
    submission_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return submission_path


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _rate_of_change(values: pd.Series, depth: pd.Series) -> pd.Series:
    denominator = depth.diff().replace(0, np.nan)
    return values.diff().divide(denominator).replace([np.inf, -np.inf], np.nan)


def _typewell_features(
    baseline: np.ndarray,
    horizontal_gr: pd.Series,
    typewell_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty = np.full(len(baseline), np.nan, dtype=float)
    if not typewell_path.exists():
        return empty.copy(), empty.copy(), empty.copy()

    typewell = pd.read_csv(typewell_path, usecols=lambda column: column in {"TVT", "GR"})
    if "TVT" not in typewell or "GR" not in typewell:
        return empty.copy(), empty.copy(), empty.copy()

    reference = pd.DataFrame(
        {
            "TVT": pd.to_numeric(typewell["TVT"], errors="coerce"),
            "GR": pd.to_numeric(typewell["GR"], errors="coerce"),
        }
    ).dropna()
    if reference.empty:
        return empty.copy(), empty.copy(), empty.copy()

    reference = reference.groupby("TVT", as_index=False, sort=True)["GR"].mean()
    reference_tvt = reference["TVT"].to_numpy(dtype=float)
    reference_gr = reference["GR"].to_numpy(dtype=float)
    gr_at_baseline = np.interp(baseline, reference_tvt, reference_gr)
    if len(reference) > 1:
        gradient = np.gradient(reference_gr, reference_tvt)
        gradient_at_baseline = np.interp(baseline, reference_tvt, gradient)
    else:
        gradient_at_baseline = np.zeros(len(baseline), dtype=float)
    delta = horizontal_gr.to_numpy(dtype=float) - gr_at_baseline
    return gr_at_baseline, gradient_at_baseline, delta


def build_well_features(horizontal_path: Path, require_target: bool) -> pd.DataFrame:
    """Build target-independent features for one well and retain evaluation rows."""
    frame = pd.read_csv(horizontal_path)
    required = {INPUT_TARGET_COLUMN, DEPTH_COLUMN}
    if require_target:
        required.add(TARGET_COLUMN)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{horizontal_path.name} missing columns: {missing}")

    well_id = well_id_from_path(horizontal_path)
    feature_source = frame.drop(
        columns=[TARGET_COLUMN, "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"],
        errors="ignore",
    )
    depth = _numeric_column(feature_source, DEPTH_COLUMN)
    input_target = _numeric_column(feature_source, INPUT_TARGET_COLUMN)
    evaluation_indices = np.flatnonzero(input_target.isna().to_numpy())
    observed_indices = np.flatnonzero(input_target.notna().to_numpy())
    if len(evaluation_indices) and len(observed_indices):
        if evaluation_indices[0] <= observed_indices[-1]:
            raise ValueError(f"{horizontal_path.name} does not contain one missing TVT_input suffix")
    baseline = interpolate_known_values(input_target, depth)
    observed = input_target.notna()

    observed_depth = depth.where(observed)
    previous_depth = observed_depth.ffill()
    next_depth = observed_depth.bfill()
    previous_tvt = input_target.ffill()
    next_tvt = input_target.bfill()
    gap_span = next_depth - previous_depth
    gap_fraction = (depth - previous_depth).divide(gap_span.replace(0, np.nan))

    horizontal_gr = _numeric_column(feature_source, "GR")
    typewell_path = horizontal_path.with_name(f"{well_id}{TYPEWELL_SUFFIX}")
    typewell_gr, typewell_gradient, gr_delta = _typewell_features(
        baseline, horizontal_gr, typewell_path
    )

    features = pd.DataFrame(index=frame.index)
    features["baseline_tvt"] = baseline
    for column in ["MD", "X", "Y", "Z", "GR"]:
        features[column] = _numeric_column(feature_source, column)
    features["previous_tvt"] = previous_tvt
    features["next_tvt"] = next_tvt
    features["distance_from_previous_md"] = depth - previous_depth
    features["distance_to_next_md"] = next_depth - depth
    features["gap_span_md"] = gap_span
    features["gap_fraction"] = gap_fraction
    features["dX_dMD"] = _rate_of_change(_numeric_column(feature_source, "X"), depth)
    features["dY_dMD"] = _rate_of_change(_numeric_column(feature_source, "Y"), depth)
    features["dZ_dMD"] = _rate_of_change(_numeric_column(feature_source, "Z"), depth)
    for window in (5, 21):
        rolling = horizontal_gr.rolling(window=window, center=True, min_periods=1)
        features[f"gr_rolling_mean_{window}"] = rolling.mean()
        features[f"gr_rolling_std_{window}"] = rolling.std()
    features["typewell_gr_at_baseline"] = typewell_gr
    features["typewell_gr_gradient"] = typewell_gradient
    features["gr_typewell_delta"] = gr_delta

    last_observed_index = int(observed_indices[-1])
    anchor_depth = float(depth.iloc[last_observed_index])
    anchor_x = float(_numeric_column(feature_source, "X").iloc[last_observed_index])
    anchor_y = float(_numeric_column(feature_source, "Y").iloc[last_observed_index])
    anchor_z = float(_numeric_column(feature_source, "Z").iloc[last_observed_index])
    anchor_gr = float(horizontal_gr.iloc[last_observed_index])
    features["md_since_anchor"] = depth - anchor_depth
    features["x_since_anchor"] = _numeric_column(feature_source, "X") - anchor_x
    features["y_since_anchor"] = _numeric_column(feature_source, "Y") - anchor_y
    features["z_since_anchor"] = _numeric_column(feature_source, "Z") - anchor_z
    evaluation_length = len(frame) - last_observed_index - 1
    features["evaluation_fraction"] = (
        np.arange(len(frame), dtype=float) - last_observed_index
    ) / max(evaluation_length, 1)
    features["known_rows"] = len(observed_indices)
    features["evaluation_rows"] = len(evaluation_indices)
    features["anchor_gr"] = anchor_gr
    features["known_gr_mean"] = float(horizontal_gr.iloc[observed_indices].mean())
    features["known_gr_std"] = float(horizontal_gr.iloc[observed_indices].std())
    features["evaluation_gr_mean"] = float(horizontal_gr.iloc[evaluation_indices].mean())
    features["evaluation_gr_std"] = float(horizontal_gr.iloc[evaluation_indices].std())
    typewell = pd.read_csv(typewell_path, usecols=lambda column: column in {"TVT", "GR"})
    typewell_tvt = _numeric_column(typewell, "TVT")
    typewell_gr_series = _numeric_column(typewell, "GR")
    features["typewell_tvt_min"] = float(typewell_tvt.min())
    features["typewell_tvt_max"] = float(typewell_tvt.max())
    features["typewell_tvt_range"] = float(typewell_tvt.max() - typewell_tvt.min())
    features["typewell_gr_mean"] = float(typewell_gr_series.mean())
    features["typewell_gr_std"] = float(typewell_gr_series.std())
    features["beam_tvt"] = baseline
    features["beam_delta_from_last"] = 0.0
    features["beam_delta_from_baseline"] = 0.0

    evaluation_mask = input_target.isna()
    if require_target:
        target = _numeric_column(frame, TARGET_COLUMN)
        evaluation_mask &= target.notna()
        features["_target"] = target
    features["_well_id"] = well_id
    features["_row_index"] = np.arange(len(frame), dtype=int)
    features["_id"] = [f"{well_id}_{index}" for index in features["_row_index"]]
    return features.loc[evaluation_mask].reset_index(drop=True)


def load_feature_rows(root: Path, require_target: bool) -> tuple[pd.DataFrame, BuildStats]:
    paths = sorted(root.glob(f"*{HORIZONTAL_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(f"No horizontal-well CSV files found under {root}")

    rows: list[pd.DataFrame] = []
    stats = BuildStats(wells=len(paths))
    for path in paths:
        raw_row_count = sum(1 for _ in path.open("rb")) - 1
        well_rows = build_well_features(path, require_target=require_target)
        stats.rows += max(raw_row_count, 0)
        stats.evaluation_rows += len(well_rows)
        rows.append(well_rows)
    return pd.concat(rows, ignore_index=True), stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def fold_indices(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two wells are required for grouped validation")
    effective_splits = min(n_splits, len(unique_groups))
    splitter = GroupKFold(n_splits=effective_splits)
    placeholder = np.zeros(len(groups), dtype=float)
    return list(splitter.split(placeholder, groups=groups))


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def data_fingerprint(train_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(train_root.glob("*.csv")):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def next_model_dir(workspace: Path) -> Path:
    models_root = workspace / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    versions = [int(path.name[1:]) for path in models_root.glob("v[0-9][0-9][0-9]")]
    version = max(versions, default=0) + 1
    model_dir = models_root / f"v{version:03d}"
    model_dir.mkdir()
    return model_dir


def train_last_known(workspace: Path, n_splits: int = 5) -> Path:
    """Evaluate and persist the deterministic last-known-TVT baseline."""
    train_root = workspace / "data" / "raw" / "train"
    model_dir = next_model_dir(workspace)
    started = time.time()
    run_path = model_dir / "run.json"
    run = {
        "version": model_dir.name,
        "status": "running",
        "parent_run": None,
        "template": "rogii-last-known-tvt",
        "command": "uv run python scripts/rogii_baseline.py train --model last_known",
        "git_commit": _git_commit(workspace.parents[1]),
        "params": {"model": "last_known", "cv_folds": n_splits},
        "random_seed": 42,
        "cv_splitter": "GroupKFold(well_id)",
        "metric": "rmse",
        "metric_direction": "minimize",
        "target_column": TARGET_COLUMN,
        "id_column": "id",
        "data_fingerprint": "",
        "runtime_seconds": None,
        "error": None,
    }
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

    try:
        feature_rows, stats = load_feature_rows(train_root, require_target=True)
        if feature_rows.empty:
            raise ValueError("No rows have missing TVT_input with an available TVT target")
        target = feature_rows["_target"].to_numpy(dtype=float)
        predictions = feature_rows["baseline_tvt"].to_numpy(dtype=float)
        groups = feature_rows["_well_id"].to_numpy()
        splits = fold_indices(groups, n_splits)
        fold_scores = [rmse(target[val_idx], predictions[val_idx]) for _, val_idx in splits]

        np.save(model_dir / "oof_preds.npy", predictions)
        feature_rows[["_id", "_well_id", "_row_index", "_target"]].to_parquet(
            model_dir / "oof_rows.parquet", index=False
        )
        (model_dir / "feature_list.txt").write_text("baseline_tvt\n", encoding="utf-8")
        pd.DataFrame(columns=["feature", "importance"]).to_csv(
            model_dir / "importance.csv", index=False
        )
        with (model_dir / "model.pkl").open("wb") as handle:
            pickle.dump({"model": "last_known", "coordinate": DEPTH_COLUMN}, handle)
        scores = {
            "metric": "rmse",
            "direction": "minimize",
            "fold_scores": fold_scores,
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
            "overall": rmse(target, predictions),
            "valid_rows": len(feature_rows),
            "valid_wells": int(feature_rows["_well_id"].nunique()),
        }
        (model_dir / "cv_scores.json").write_text(
            json.dumps(scores, indent=2), encoding="utf-8"
        )
        run.update(
            {
                "status": "completed",
                "data_fingerprint": data_fingerprint(train_root),
                "runtime_seconds": round(time.time() - started, 3),
                "data_stats": asdict(stats),
            }
        )
    except Exception as exc:
        run.update(
            {
                "status": "failed",
                "runtime_seconds": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
        raise

    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    return model_dir


def predict_last_known(workspace: Path, model_dir: Path) -> Path:
    """Generate hidden-test-compatible predictions from the last-known model."""
    test_root = workspace / "data" / "raw" / "test"
    sample_path = workspace / "data" / "raw" / "sample_submission.csv"
    rows, _ = load_feature_rows(test_root, require_target=False)
    predictions_by_id = pd.Series(rows["baseline_tvt"].to_numpy(), index=rows["_id"])
    sample = pd.read_csv(sample_path)
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"Unexpected sample submission columns: {list(sample.columns)}")
    missing_ids = sample.loc[~sample["id"].isin(predictions_by_id.index), "id"]
    if not missing_ids.empty:
        raise ValueError(f"Missing predictions for {len(missing_ids)} sample IDs")

    submission = sample.copy()
    submission["tvt"] = submission["id"].map(predictions_by_id)
    if not np.isfinite(submission["tvt"].to_numpy(dtype=float)).all():
        raise ValueError("Submission predictions contain non-finite values")
    submission_path = workspace / "submissions" / f"sub_{model_dir.name}_last_known.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    if submission_path.exists():
        raise FileExistsError(f"Refusing to overwrite {submission_path}")
    submission.to_csv(submission_path, index=False)
    np.save(model_dir / "test_preds.npy", submission["tvt"].to_numpy(dtype=float))

    scores = json.loads((model_dir / "cv_scores.json").read_text(encoding="utf-8"))
    metadata = {
        "source_model_versions": [model_dir.name],
        "ensemble_weights": None,
        "local_cv_score": scores["overall"],
        "metric": "rmse",
        "generated_command": (
            "uv run python scripts/rogii_baseline.py predict "
            f"--model-dir models/{model_dir.name}"
        ),
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "dry_run": {
            "valid": True,
            "checks": ["sample ID order", "finite predictions", "row count"],
        },
        "kaggle_submission_id": None,
        "lb_score": None,
        "rank": None,
        "submitted": False,
        "submission_mode": "notebook",
    }
    submission_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return submission_path


def inspect_data(workspace: Path) -> dict:
    train_root = workspace / "data" / "raw" / "train"
    test_root = workspace / "data" / "raw" / "test"
    train_paths = sorted(train_root.glob(f"*{HORIZONTAL_SUFFIX}"))
    test_paths = sorted(test_root.glob(f"*{HORIZONTAL_SUFFIX}"))
    if not train_paths:
        raise FileNotFoundError(f"No training wells found under {train_root}")
    first = pd.read_csv(train_paths[0], nrows=5)
    result = {
        "train_wells": len(train_paths),
        "public_test_wells": len(test_paths),
        "columns": list(first.columns),
        "target_column": TARGET_COLUMN,
        "input_target_column": INPUT_TARGET_COLUMN,
    }
    state_path = workspace / ".state" / "schema.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", choices=["last_known"], default="last_known")
    train_parser.add_argument("--folds", type=int, default=5)
    scan_parser = subparsers.add_parser("scan-beam")
    scan_parser.add_argument("--max-wells", type=int, default=0)
    scan_parser.add_argument("--blends", default="0,0.1,0.25,0.5,0.75")
    scan_parser.add_argument("--output", default="beam_scan.csv")
    select_parser = subparsers.add_parser("select-beam")
    select_parser.add_argument("--details", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument("--folds", type=int, default=5)
    train_beam_parser = subparsers.add_parser("train-beam")
    train_beam_parser.add_argument("--selection", type=Path, required=True)
    train_beam_parser.add_argument("--final-config", default="very_loose")
    train_beam_parser.add_argument("--final-blend", type=float, default=0.7)
    predict_beam_parser = subparsers.add_parser("predict-beam")
    predict_beam_parser.add_argument("--model-dir", type=Path, required=True)
    train_residual_parser = subparsers.add_parser("train-residual")
    train_residual_parser.add_argument("--folds", type=int, default=5)
    train_residual_parser.add_argument("--estimators", type=int, default=600)
    train_residual_parser.add_argument("--learning-rate", type=float, default=0.03)
    predict_residual_parser = subparsers.add_parser("predict-residual")
    predict_residual_parser.add_argument("--model-dir", type=Path, required=True)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.resolve()
    try:
        if args.command == "inspect":
            print(json.dumps(inspect_data(workspace), indent=2))
        elif args.command == "train":
            model_dir = train_last_known(workspace, n_splits=args.folds)
            print(model_dir)
        elif args.command == "scan-beam":
            blends = tuple(float(value) for value in args.blends.split(","))
            print(
                scan_beam_configs(
                    workspace,
                    max_wells=args.max_wells,
                    blends=blends,
                    output_name=args.output,
                )
            )
        elif args.command == "select-beam":
            details_path = args.details
            output_path = args.output
            if not details_path.is_absolute():
                details_path = workspace / details_path
            if not output_path.is_absolute():
                output_path = workspace / output_path
            result = nested_beam_selection(details_path, output_path, n_splits=args.folds)
            print(json.dumps(result, indent=2))
        elif args.command == "train-beam":
            selection_path = args.selection
            if not selection_path.is_absolute():
                selection_path = workspace / selection_path
            print(
                train_beam(
                    workspace,
                    selection_path,
                    final_config=args.final_config,
                    final_blend=args.final_blend,
                )
            )
        elif args.command == "predict-beam":
            model_dir = args.model_dir
            if not model_dir.is_absolute():
                model_dir = workspace / model_dir
            print(predict_beam(workspace, model_dir))
        elif args.command == "train-residual":
            print(
                train_residual(
                    workspace,
                    n_splits=args.folds,
                    n_estimators=args.estimators,
                    learning_rate=args.learning_rate,
                )
            )
        elif args.command == "predict-residual":
            model_dir = args.model_dir
            if not model_dir.is_absolute():
                model_dir = workspace / model_dir
            print(predict_residual(workspace, model_dir))
        elif args.command == "predict":
            model_dir = args.model_dir
            if not model_dir.is_absolute():
                model_dir = workspace / model_dir
            print(predict_last_known(workspace, model_dir))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
