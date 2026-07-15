#!/usr/bin/env python3
"""Leakage-safe latent-surface particle filtering for the ROGII competition."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import signal
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - pyarrow is a project dependency
    pa = None
    pq = None

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # Unit tests use the small pure-Python fallback.
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        del kwargs
        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"
REFERENCE_URL = "https://www.kaggle.com/code/amerhu/rogii-wellbore-geology-exact-hmm-smoother"
INPUT_FEATURES = [
    "horizontal.MD",
    "horizontal.Z",
    "horizontal.GR",
    "horizontal.TVT_input",
    "typewell.TVT",
    "typewell.GR",
]


def well_id_from_path(path: Path) -> str:
    if not path.name.endswith(HORIZONTAL_SUFFIX):
        raise ValueError(f"Unexpected horizontal-well filename: {path.name}")
    return path.name[: -len(HORIZONTAL_SUFFIX)]


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(target - prediction))))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
    if result.returncode != 0:
        return []
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def data_fingerprint(train_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(train_root.glob("*.csv")):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def scored_row_counts(train_root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(train_root.glob(f"*{HORIZONTAL_SUFFIX}")):
        frame = pd.read_csv(path, usecols=["TVT_input"])
        records.append(
            {
                "well": well_id_from_path(path),
                "scored_rows": int(frame["TVT_input"].isna().sum()),
            }
        )
    if not records:
        raise FileNotFoundError(f"No horizontal-well CSV files found under {train_root}")
    return pd.DataFrame(records).sort_values("well").reset_index(drop=True)


def assign_balanced_folds(counts: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """Reproduce non-shuffled GroupKFold using scored-row counts as group weights."""
    if n_splits < 2:
        raise ValueError("At least two outer folds are required")
    ordered = counts[["well", "scored_rows"]].sort_values("well").reset_index(drop=True)
    if len(ordered) < n_splits:
        raise ValueError("Number of wells must be at least the number of folds")

    weights = ordered["scored_rows"].to_numpy(dtype=np.int64)
    descending = np.argsort(weights, kind="stable")[::-1]
    fold_weights = np.zeros(n_splits, dtype=np.int64)
    group_to_fold = np.zeros(len(ordered), dtype=np.int64)
    for group_index in descending:
        fold = int(np.argmin(fold_weights))
        fold_weights[fold] += weights[group_index]
        group_to_fold[group_index] = fold + 1

    result = ordered.assign(fold=group_to_fold)
    result["splitter"] = "GroupKFold(n_splits=%d, weight=scored_rows)" % n_splits
    return result


def ensure_fold_manifest(workspace: Path, n_splits: int) -> tuple[Path, pd.DataFrame]:
    path = workspace / "reports" / "canonical_outer_folds_v001.csv"
    expected = assign_balanced_folds(
        scored_row_counts(workspace / "data" / "raw" / "train"), n_splits
    )
    if path.exists():
        current = pd.read_csv(path)
        pd.testing.assert_frame_equal(current, expected, check_dtype=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        expected.to_csv(path, index=False)
    return path, expected


def prepare_typewell(typewell: pd.DataFrame, step: float) -> tuple[np.ndarray, float, float]:
    reference = pd.DataFrame(
        {
            "TVT": pd.to_numeric(typewell["TVT"], errors="coerce"),
            "GR": pd.to_numeric(typewell["GR"], errors="coerce"),
        }
    ).dropna()
    reference = reference.groupby("TVT", as_index=False, sort=True)["GR"].mean()
    if len(reference) < 3:
        raise ValueError("Typewell requires at least three finite TVT/GR rows")
    tvt = reference["TVT"].to_numpy(dtype=float)
    gr = reference["GR"].to_numpy(dtype=float)
    minimum = float(tvt[0])
    maximum = float(tvt[-1])
    grid = np.arange(minimum, maximum + step, step, dtype=float)
    return np.interp(grid, tvt, gr), minimum, maximum


def prefix_state(horizontal: pd.DataFrame, reference_gr: np.ndarray, minimum: float, step: float):
    known = horizontal[horizontal["TVT_input"].notna()]
    if len(known) < 3:
        raise ValueError("Horizontal well requires at least three known TVT_input rows")
    last = known.iloc[-1]
    known_tvt = known["TVT_input"].to_numpy(dtype=float)
    known_gr = pd.to_numeric(known["GR"], errors="coerce").to_numpy(dtype=float)
    reference_at_known = np.interp(
        known_tvt,
        minimum + np.arange(len(reference_gr), dtype=float) * step,
        reference_gr,
    )
    valid = np.isfinite(known_gr) & np.isfinite(reference_at_known)
    residual = known_gr[valid] - reference_at_known[valid]
    gr_sigma = float(np.clip(np.nanstd(residual), 10.0, 60.0)) if valid.any() else 30.0

    tail = known.tail(30)
    dtvt = np.diff(tail["TVT_input"].to_numpy(dtype=float))
    dz = np.diff(tail["Z"].to_numpy(dtype=float))
    dmd = np.diff(tail["MD"].to_numpy(dtype=float))
    valid_rate = np.isfinite(dtvt) & np.isfinite(dz) & np.isfinite(dmd) & (dmd > 0)
    initial_rate = (
        float(np.median((dtvt[valid_rate] + dz[valid_rate]) / dmd[valid_rate]))
        if valid_rate.sum() >= 3
        else 0.0
    )
    return last, gr_sigma, initial_rate


@njit(cache=True, nogil=True)
def _interp_regular(values, position, minimum, step):
    raw = (position - minimum) / step
    if raw <= 0.0:
        return values[0]
    index = int(raw)
    last = len(values) - 1
    if index >= last:
        return values[last]
    fraction = raw - index
    return values[index] * (1.0 - fraction) + values[index + 1] * fraction


@njit(cache=True, nogil=True)
def _particle_paths(
    md,
    z,
    gr,
    reference_gr,
    minimum_tvt,
    maximum_tvt,
    reference_step,
    gr_sigma,
    initial_surface,
    initial_rate,
    initial_md,
    particles,
    seeds,
    initial_spread,
    momentum,
    velocity_noise,
    position_noise,
    resample_position_noise,
    resample_velocity_noise,
    resample_threshold,
):
    n_rows = len(md)
    predictions = np.empty((seeds, n_rows), dtype=np.float64)
    log_likelihoods = np.empty(seeds, dtype=np.float64)

    for seed in range(seeds):
        np.random.seed(seed)
        surface = np.empty(particles, dtype=np.float64)
        rate = np.empty(particles, dtype=np.float64)
        weight = np.empty(particles, dtype=np.float64)
        for particle in range(particles):
            surface[particle] = initial_surface + initial_spread * np.random.randn()
            rate[particle] = initial_rate + 0.01 * np.random.randn()
            weight[particle] = 1.0 / particles

        previous_md = initial_md
        log_likelihood = 0.0
        for row in range(n_rows):
            delta_md = md[row] - previous_md
            if delta_md < 1.0:
                delta_md = 1.0

            has_observation = np.isfinite(gr[row])
            average_likelihood = 0.0
            for particle in range(particles):
                rate[particle] = momentum * rate[particle] + velocity_noise * np.random.randn()
                surface[particle] += rate[particle] * delta_md + position_noise * np.random.randn()
                tvt = surface[particle] - z[row]
                if tvt < minimum_tvt - 100.0:
                    tvt = minimum_tvt - 100.0
                elif tvt > maximum_tvt + 100.0:
                    tvt = maximum_tvt + 100.0
                surface[particle] = tvt + z[row]

                if has_observation:
                    expected_gr = _interp_regular(reference_gr, tvt, minimum_tvt, reference_step)
                    standardized = (gr[row] - expected_gr) / gr_sigma
                    squared = standardized * standardized
                    if squared > 600.0:
                        squared = 600.0
                    likelihood = np.exp(-0.5 * squared)
                    if likelihood < 1e-300:
                        likelihood = 1e-300
                    average_likelihood += weight[particle] * likelihood
                    weight[particle] *= likelihood

            if has_observation:
                if average_likelihood < 1e-300:
                    average_likelihood = 1e-300
                log_likelihood += np.log(average_likelihood)

                total_weight = 0.0
                for particle in range(particles):
                    total_weight += weight[particle]
                if total_weight <= 0.0:
                    for particle in range(particles):
                        weight[particle] = 1.0 / particles
                else:
                    for particle in range(particles):
                        weight[particle] /= total_weight

                squared_weight = 0.0
                for particle in range(particles):
                    squared_weight += weight[particle] * weight[particle]
                effective = 1.0 / squared_weight if squared_weight > 0.0 else 0.0
                if effective < resample_threshold * particles:
                    cumulative = np.empty(particles, dtype=np.float64)
                    running = 0.0
                    for particle in range(particles):
                        running += weight[particle]
                        cumulative[particle] = running
                    new_surface = np.empty(particles, dtype=np.float64)
                    new_rate = np.empty(particles, dtype=np.float64)
                    offset = np.random.uniform(0.0, 1.0 / particles)
                    source = 0
                    for particle in range(particles):
                        threshold = offset + particle / particles
                        while source < particles - 1 and cumulative[source] < threshold:
                            source += 1
                        new_surface[particle] = (
                            surface[source] + resample_position_noise * np.random.randn()
                        )
                        new_rate[particle] = (
                            rate[source] + resample_velocity_noise * np.random.randn()
                        )
                    for particle in range(particles):
                        surface[particle] = new_surface[particle]
                        rate[particle] = new_rate[particle]
                        weight[particle] = 1.0 / particles

            estimate = 0.0
            for particle in range(particles):
                estimate += weight[particle] * (surface[particle] - z[row])
            predictions[seed, row] = estimate
            previous_md = md[row]

        log_likelihoods[seed] = log_likelihood

    return predictions, log_likelihoods


def particle_candidates(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    particles: int,
    seeds: int,
    scales: tuple[float, ...],
    holds: tuple[float, ...],
    reference_step: float = 0.2,
) -> tuple[pd.DataFrame, dict]:
    evaluation = horizontal["TVT_input"].isna().to_numpy()
    evaluation_indices = np.flatnonzero(evaluation)
    if not len(evaluation_indices):
        raise ValueError("Horizontal well has no missing TVT_input suffix")

    reference_gr, minimum_tvt, maximum_tvt = prepare_typewell(typewell, reference_step)
    last, gr_sigma, initial_rate = prefix_state(
        horizontal, reference_gr, minimum_tvt, reference_step
    )
    md = horizontal.loc[evaluation, "MD"].to_numpy(dtype=float)
    z = horizontal.loc[evaluation, "Z"].to_numpy(dtype=float)
    gr = pd.to_numeric(horizontal.loc[evaluation, "GR"], errors="coerce").to_numpy(dtype=float)
    paths, likelihoods = _particle_paths(
        md,
        z,
        gr,
        reference_gr,
        minimum_tvt,
        maximum_tvt,
        reference_step,
        gr_sigma,
        float(last["TVT_input"] + last["Z"]),
        initial_rate,
        float(last["MD"]),
        particles,
        seeds,
        4.5,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
    )

    result = pd.DataFrame({"_row_index": evaluation_indices})
    centered_likelihood = likelihoods - np.max(likelihoods)
    last_tvt = float(last["TVT_input"])
    for scale in scales:
        weights = np.exp(centered_likelihood / scale)
        weights /= weights.sum()
        prediction = weights @ paths
        for hold in holds:
            name = candidate_name(scale, hold)
            result[name] = ((1.0 - hold) * prediction + hold * last_tvt).astype(np.float32)
    diagnostics = {
        "gr_sigma": gr_sigma,
        "initial_surface_rate": initial_rate,
        "observed_gr_rows": int(np.isfinite(gr).sum()),
        "missing_gr_rows": int((~np.isfinite(gr)).sum()),
        "log_likelihood_min": float(np.min(likelihoods)),
        "log_likelihood_max": float(np.max(likelihoods)),
    }
    return result, diagnostics


def candidate_name(scale: float, hold: float) -> str:
    scale_tag = f"{scale:g}".replace(".", "p")
    hold_tag = f"{hold:g}".replace(".", "p")
    return f"pf_scale_{scale_tag}_hold_{hold_tag}"


def candidate_names(scales: tuple[float, ...], holds: tuple[float, ...]) -> list[str]:
    return [candidate_name(scale, hold) for scale in scales for hold in holds]


def parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("At least one numeric value is required")
    return result


def choose_nested_candidates(summary: pd.DataFrame) -> tuple[list[dict], str]:
    selections = []
    folds = sorted(int(value) for value in summary["fold"].unique())
    for fold in folds:
        training = summary.loc[summary["fold"] != fold]
        aggregate = training.groupby("candidate", as_index=False).agg(
            squared_error=("squared_error", "sum"), rows=("rows", "sum")
        )
        aggregate["rmse"] = np.sqrt(aggregate["squared_error"] / aggregate["rows"])
        selected = aggregate.sort_values(["rmse", "candidate"]).iloc[0]
        validation = summary.loc[
            (summary["fold"] == fold) & (summary["candidate"] == selected["candidate"])
        ]
        selections.append(
            {
                "fold": fold,
                "selected_candidate": str(selected["candidate"]),
                "train_rmse": float(selected["rmse"]),
                "validation_rmse": float(
                    np.sqrt(validation["squared_error"].sum() / validation["rows"].sum())
                ),
                "validation_rows": int(validation["rows"].sum()),
                "validation_wells": int(validation["well"].nunique()),
            }
        )

    aggregate_all = summary.groupby("candidate", as_index=False).agg(
        squared_error=("squared_error", "sum"), rows=("rows", "sum")
    )
    aggregate_all["rmse"] = np.sqrt(aggregate_all["squared_error"] / aggregate_all["rows"])
    final_candidate = str(aggregate_all.sort_values(["rmse", "candidate"]).iloc[0]["candidate"])
    return selections, final_candidate


def next_feature_path(workspace: Path) -> Path:
    root = workspace / "data" / "features"
    root.mkdir(parents=True, exist_ok=True)
    versions = [int(path.stem[1:]) for path in root.glob("v[0-9][0-9][0-9].parquet")]
    return root / f"v{max(versions, default=0) + 1:03d}.parquet"


def baseline_scores_on_manifest(workspace: Path, manifest: pd.DataFrame, version: str) -> dict:
    model_dir = workspace / "models" / version
    rows = pd.read_parquet(model_dir / "oof_rows.parquet")
    prediction = np.load(model_dir / "oof_preds.npy")
    if len(rows) != len(prediction):
        raise ValueError(f"{version} OOF length mismatch")
    fold_map = manifest.set_index("well")["fold"]
    folds = rows["_well_id"].map(fold_map).to_numpy(dtype=int)
    target = rows["_target"].to_numpy(dtype=float)
    return {
        "overall": rmse(target, prediction),
        "fold_scores": [
            rmse(target[folds == fold], prediction[folds == fold])
            for fold in sorted(manifest["fold"].unique())
        ],
    }


def build_training_candidates(
    workspace: Path,
    manifest: pd.DataFrame,
    feature_path: Path,
    *,
    particles: int,
    seeds: int,
    scales: tuple[float, ...],
    holds: tuple[float, ...],
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required to stream the candidate feature parquet")
    if feature_path.exists():
        raise FileExistsError(f"Refusing to overwrite {feature_path}")

    train_root = workspace / "data" / "raw" / "train"
    fold_map = manifest.set_index("well")["fold"].to_dict()
    names = candidate_names(scales, holds)
    summary_records = []
    diagnostics = []
    failures = []
    writer = None
    paths = sorted(train_root.glob(f"*{HORIZONTAL_SUFFIX}"))
    try:
        for index, horizontal_path in enumerate(paths, start=1):
            well = well_id_from_path(horizontal_path)
            horizontal = pd.read_csv(horizontal_path)
            typewell_path = horizontal_path.with_name(f"{well}{TYPEWELL_SUFFIX}")
            typewell = pd.read_csv(typewell_path)
            evaluation = horizontal["TVT_input"].isna().to_numpy()
            evaluation_indices = np.flatnonzero(evaluation)
            target = pd.to_numeric(horizontal.loc[evaluation, "TVT"], errors="coerce").to_numpy(
                dtype=float
            )
            last_tvt = float(horizontal.loc[~evaluation, "TVT_input"].iloc[-1])
            try:
                candidates, well_diagnostics = particle_candidates(
                    horizontal,
                    typewell,
                    particles=particles,
                    seeds=seeds,
                    scales=scales,
                    holds=holds,
                )
            except Exception as exc:
                candidates = pd.DataFrame({"_row_index": evaluation_indices})
                for name in names:
                    candidates[name] = np.full(len(evaluation_indices), last_tvt, dtype=np.float32)
                well_diagnostics = {}
                failures.append({"well": well, "error": f"{type(exc).__name__}: {exc}"})

            frame = pd.DataFrame(
                {
                    "_id": [f"{well}_{row}" for row in evaluation_indices],
                    "_well_id": well,
                    "_row_index": evaluation_indices,
                    "_target": target,
                    "fold": int(fold_map[well]),
                }
            )
            for name in names:
                prediction = candidates[name].to_numpy(dtype=np.float32)
                if len(prediction) != len(frame) or not np.isfinite(prediction).all():
                    raise ValueError(f"Invalid candidate {name} for well {well}")
                frame[name] = prediction
                squared_error = float(np.sum(np.square(prediction.astype(float) - target)))
                summary_records.append(
                    {
                        "well": well,
                        "fold": int(fold_map[well]),
                        "candidate": name,
                        "squared_error": squared_error,
                        "rows": len(frame),
                        "rmse": float(np.sqrt(squared_error / len(frame))),
                    }
                )

            diagnostics.append({"well": well, **well_diagnostics})
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(feature_path, table.schema, compression="zstd")
            writer.write_table(table)
            if index % 10 == 0 or index == len(paths):
                print(
                    f"particle progress: {index}/{len(paths)} wells ({time.strftime('%H:%M:%S')})",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    return pd.DataFrame(summary_records), diagnostics, failures


def materialize_oof(
    feature_path: Path, selections: list[dict], names: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_parquet(
        feature_path,
        columns=["_id", "_well_id", "_row_index", "_target", "fold", *names],
    )
    prediction = np.full(len(frame), np.nan, dtype=float)
    for selection in selections:
        fold = int(selection["fold"])
        candidate = str(selection["selected_candidate"])
        mask = frame["fold"].to_numpy(dtype=int) == fold
        prediction[mask] = frame.loc[mask, candidate].to_numpy(dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("Nested OOF predictions contain non-finite values")
    return frame[["_id", "_well_id", "_row_index", "_target", "fold"]], prediction


def predict_visible_test(
    workspace: Path,
    candidate: str,
    *,
    particles: int,
    seeds: int,
    scales: tuple[float, ...],
    holds: tuple[float, ...],
) -> tuple[pd.DataFrame, list[dict]]:
    test_root = workspace / "data" / "raw" / "test"
    frames = []
    failures = []
    for horizontal_path in sorted(test_root.glob(f"*{HORIZONTAL_SUFFIX}")):
        well = well_id_from_path(horizontal_path)
        horizontal = pd.read_csv(horizontal_path)
        evaluation = horizontal["TVT_input"].isna().to_numpy()
        evaluation_indices = np.flatnonzero(evaluation)
        last_tvt = float(horizontal.loc[~evaluation, "TVT_input"].iloc[-1])
        try:
            predictions, _ = particle_candidates(
                horizontal,
                pd.read_csv(horizontal_path.with_name(f"{well}{TYPEWELL_SUFFIX}")),
                particles=particles,
                seeds=seeds,
                scales=scales,
                holds=holds,
            )
            values = predictions[candidate].to_numpy(dtype=float)
        except Exception as exc:
            values = np.full(len(evaluation_indices), last_tvt, dtype=float)
            failures.append({"well": well, "error": f"{type(exc).__name__}: {exc}"})
        frames.append(
            pd.DataFrame(
                {
                    "_id": [f"{well}_{row}" for row in evaluation_indices],
                    "prediction": values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True), failures


def write_submission(
    workspace: Path,
    version: str,
    predictions: pd.DataFrame,
    metadata: dict,
) -> tuple[Path, np.ndarray]:
    sample = pd.read_csv(workspace / "data" / "raw" / "sample_submission.csv")
    if list(sample.columns) != ["id", "tvt"] or sample["id"].duplicated().any():
        raise ValueError("Unexpected sample submission contract")
    prediction_map = predictions.set_index("_id")["prediction"]
    submission = sample.copy()
    submission["tvt"] = submission["id"].map(prediction_map)
    if submission["tvt"].isna().any() or not np.isfinite(submission["tvt"]).all():
        raise ValueError("Submission has missing or non-finite predictions")
    path = workspace / "submissions" / f"sub_{version}_particle_state_space.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    submission.to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path, submission["tvt"].to_numpy(dtype=float)


def finalize_candidate_decision(cv_decision: str, test_failures: list[dict]) -> str:
    if cv_decision == "candidate_ready" and test_failures:
        return "candidate_blocked_test_failures"
    return cv_decision


def run_smoke(
    workspace: Path,
    *,
    particles: int,
    seeds: int,
    scales: tuple[float, ...],
    holds: tuple[float, ...],
    wells: int,
) -> dict:
    train_root = workspace / "data" / "raw" / "train"
    paths = sorted(train_root.glob(f"*{HORIZONTAL_SUFFIX}"))
    counts = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["TVT_input"])
        counts.append((int(frame["TVT_input"].isna().sum()), path))
    counts.sort(key=lambda item: (item[0], item[1].name))
    positions = np.linspace(0, len(counts) - 1, min(wells, len(counts)), dtype=int)
    selected = [counts[position][1] for position in positions]
    totals = {name: [0.0, 0] for name in candidate_names(scales, holds)}
    records = []
    for path in selected:
        horizontal = pd.read_csv(path)
        well = well_id_from_path(path)
        target = horizontal.loc[horizontal["TVT_input"].isna(), "TVT"].to_numpy(dtype=float)
        candidates, _ = particle_candidates(
            horizontal,
            pd.read_csv(path.with_name(f"{well}{TYPEWELL_SUFFIX}")),
            particles=particles,
            seeds=seeds,
            scales=scales,
            holds=holds,
        )
        for name in candidate_names(scales, holds):
            prediction = candidates[name].to_numpy(dtype=float)
            squared_error = float(np.sum(np.square(prediction - target)))
            totals[name][0] += squared_error
            totals[name][1] += len(target)
            records.append(
                {
                    "well": well,
                    "rows": len(target),
                    "candidate": name,
                    "rmse": float(np.sqrt(squared_error / len(target))),
                }
            )
        print(f"smoke: {well} ({len(target)} scored rows)", flush=True)
    pooled = sorted(
        (
            {"candidate": name, "pooled_rmse": float(np.sqrt(values[0] / values[1]))}
            for name, values in totals.items()
        ),
        key=lambda item: (item["pooled_rmse"], item["candidate"]),
    )
    return {
        "intended_use": "runtime diagnostic only; do not select full-CV parameters",
        "wells": [well_id_from_path(path) for path in selected],
        "pooled": pooled,
        "rows": records,
    }


def train(
    workspace: Path,
    *,
    version: str,
    folds: int,
    particles: int,
    seeds: int,
    scales: tuple[float, ...],
    holds: tuple[float, ...],
    candidate_gate: float,
    exhausted_gate: float,
) -> Path:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Full particle training requires numba")
    model_dir = workspace / "models" / version
    if model_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {model_dir}")
    model_dir.mkdir(parents=True)
    run_path = model_dir / "run.json"
    started = time.time()
    run = {
        "version": version,
        "status": "running",
        "parent_run": "v004",
        "template": "rogii-latent-surface-particle-filter",
        "runtime_seconds": None,
        "error": None,
    }
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(signum, frame):
        del signum, frame
        raise KeyboardInterrupt("received SIGTERM")

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        project_root = workspace.parents[1]
        script_path = Path(__file__).resolve()
        feature_path = next_feature_path(workspace)
        manifest_path, manifest = ensure_fold_manifest(workspace, folds)
        names = candidate_names(scales, holds)
        run.update(
            {
                "command": shlex.join([sys.executable, *sys.argv]),
                "git_commit": git_commit(project_root),
                "git_dirty_paths": git_dirty_paths(project_root),
                "source_sha256": sha256_file(script_path),
                "external_method_reference": REFERENCE_URL,
                "params": {
                    "particles": particles,
                    "seeds": seeds,
                    "likelihood_scales": scales,
                    "hold_weights": holds,
                    "reference_grid_step": 0.2,
                    "momentum": 0.998,
                    "velocity_noise": 0.002,
                    "position_noise": 0.005,
                },
                "random_seed": 0,
                "cv_splitter": "canonical row-balanced GroupKFold by well",
                "fold_manifest": str(manifest_path.relative_to(workspace)),
                "fold_manifest_sha256": sha256_file(manifest_path),
                "metric": "rmse",
                "metric_direction": "minimize",
                "target_column": "TVT",
                "id_column": "id",
                "feature_path": str(feature_path.relative_to(workspace)),
                "data_fingerprint": data_fingerprint(workspace / "data" / "raw" / "train"),
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                },
            }
        )
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

        summary, diagnostics, failures = build_training_candidates(
            workspace,
            manifest,
            feature_path,
            particles=particles,
            seeds=seeds,
            scales=scales,
            holds=holds,
        )
        by_well_path = workspace / "reports" / f"{version}_particle_candidates_by_well.csv"
        scan_path = workspace / "reports" / f"{version}_particle_candidate_scan.csv"
        diagnostics_path = workspace / "reports" / f"{version}_particle_diagnostics.json"
        for path in (by_well_path, scan_path, diagnostics_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
        summary.to_csv(by_well_path, index=False)
        scan = summary.groupby("candidate", as_index=False).agg(
            squared_error=("squared_error", "sum"), rows=("rows", "sum")
        )
        scan["pooled_rmse"] = np.sqrt(scan["squared_error"] / scan["rows"])
        scan.sort_values(["pooled_rmse", "candidate"]).to_csv(scan_path, index=False)
        diagnostics_path.write_text(
            json.dumps({"wells": diagnostics, "failures": failures}, indent=2),
            encoding="utf-8",
        )

        selections, final_candidate = choose_nested_candidates(summary)
        oof_rows, oof_prediction = materialize_oof(feature_path, selections, names)
        target = oof_rows["_target"].to_numpy(dtype=float)
        fold_values = oof_rows["fold"].to_numpy(dtype=int)
        fold_scores = [
            rmse(target[fold_values == fold], oof_prediction[fold_values == fold])
            for fold in sorted(manifest["fold"].unique())
        ]
        overall = rmse(target, oof_prediction)
        v004 = baseline_scores_on_manifest(workspace, manifest, "v004")
        folds_better = int(
            sum(current < baseline for current, baseline in zip(fold_scores, v004["fold_scores"]))
        )
        decision = (
            "candidate_ready"
            if overall <= candidate_gate and folds_better >= 4 and not failures
            else ("promising_continue" if overall < exhausted_gate else "exhausted")
        )

        np.save(model_dir / "oof_preds.npy", oof_prediction)
        oof_rows.to_parquet(model_dir / "oof_rows.parquet", index=False)
        (model_dir / "feature_list.txt").write_text(
            "\n".join(INPUT_FEATURES) + "\n", encoding="utf-8"
        )
        pd.DataFrame(columns=["feature", "importance"]).to_csv(
            model_dir / "importance.csv", index=False
        )
        model = {
            "model": "latent_surface_particle_filter",
            "final_candidate": final_candidate,
            "fold_selections": selections,
            "params": run["params"],
        }
        with (model_dir / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)

        scores = {
            "metric": "rmse",
            "direction": "minimize",
            "fold_scores": fold_scores,
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
            "overall": overall,
            "valid_rows": len(oof_rows),
            "valid_wells": int(oof_rows["_well_id"].nunique()),
            "selection_protocol": "outer-fold training wells select fixed particle candidate grid",
            "fold_selections": selections,
            "final_candidate": final_candidate,
            "fallback_wells": len(failures),
            "canonical_v004": v004,
            "folds_better_than_v004": folds_better,
            "candidate_gate": candidate_gate,
            "exhausted_gate": exhausted_gate,
            "decision": decision,
        }
        submission_path = None
        test_failures = []
        if decision == "candidate_ready":
            test_frame, test_failures = predict_visible_test(
                workspace,
                final_candidate,
                particles=particles,
                seeds=seeds,
                scales=scales,
                holds=holds,
            )
            decision = finalize_candidate_decision(decision, test_failures)
            if decision == "candidate_ready":
                metadata = {
                    "source_model_versions": [version],
                    "ensemble_weights": None,
                    "local_cv_score": overall,
                    "fold_scores": fold_scores,
                    "metric": "rmse",
                    "generated_command": run["command"],
                    "generated_at": pd.Timestamp.now("UTC").isoformat(),
                    "dry_run": {
                        "valid": True,
                        "checks": ["sample ID order", "finite predictions", "row count"],
                        "fallback_wells": 0,
                    },
                    "kaggle_submission_id": None,
                    "lb_score": None,
                    "rank": None,
                    "submitted": False,
                    "submission_mode": "notebook",
                    "decision": decision,
                }
                submission_path, test_prediction = write_submission(
                    workspace, version, test_frame, metadata
                )
                np.save(model_dir / "test_preds.npy", test_prediction)

        scores["decision"] = decision
        (model_dir / "cv_scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")

        run.update(
            {
                "status": "completed",
                "runtime_seconds": round(time.time() - started, 3),
                "decision": decision,
                "cv_score": overall,
                "folds_better_than_v004": folds_better,
                "final_candidate": final_candidate,
                "fallback_wells": len(failures),
                "test_fallback_wells": len(test_failures),
                "submission_path": (
                    str(submission_path.relative_to(workspace)) if submission_path else None
                ),
                "submitted": False,
            }
        )
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    except BaseException as exc:
        run.update(
            {
                "status": "failed",
                "runtime_seconds": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    return model_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--particles", type=int, default=128)
    smoke.add_argument("--seeds", type=int, default=8)
    smoke.add_argument("--scales", type=parse_float_tuple, default=(3.0, 5.0, 8.0, 12.0))
    smoke.add_argument("--holds", type=parse_float_tuple, default=(0.0, 0.2, 0.5))
    smoke.add_argument("--wells", type=int, default=12)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--version", default="v005")
    train_parser.add_argument("--folds", type=int, default=5)
    train_parser.add_argument("--particles", type=int, default=500)
    train_parser.add_argument("--seeds", type=int, default=32)
    train_parser.add_argument("--scales", type=parse_float_tuple, default=(3.0, 5.0, 8.0, 12.0))
    train_parser.add_argument("--holds", type=parse_float_tuple, default=(0.0, 0.2, 0.5))
    train_parser.add_argument("--candidate-gate", type=float, default=12.5)
    train_parser.add_argument("--exhausted-gate", type=float, default=14.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.resolve()
    try:
        if args.command == "smoke":
            result = run_smoke(
                workspace,
                particles=args.particles,
                seeds=args.seeds,
                scales=args.scales,
                holds=args.holds,
                wells=args.wells,
            )
            print(json.dumps(result, indent=2))
        else:
            print(
                train(
                    workspace,
                    version=args.version,
                    folds=args.folds,
                    particles=args.particles,
                    seeds=args.seeds,
                    scales=args.scales,
                    holds=args.holds,
                    candidate_gate=args.candidate_gate,
                    exhausted_gate=args.exhausted_gate,
                )
            )
    except KeyboardInterrupt as exc:
        print(f"KeyboardInterrupt: {exc}", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
