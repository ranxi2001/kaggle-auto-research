#!/usr/bin/env python3
"""Nested latent-surface HMM smoothing for the ROGII competition."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
import pickle
import platform
import resource
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - pyarrow is a project dependency
    pa = None
    pq = None

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # Unit tests exercise small arrays through the Python fallback.
    NUMBA_AVAILABLE = False
    prange = range

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
REFERENCE_SHA256 = "2321997c8bcbca442d7d7abfc5b9b7eeed251bac8c671b3554b54c9355c231dc"
PARENT_VERSION = "v006"
PARENT_CANDIDATE = "pf_scale_12_hold_0p2"
BLOCK_SIZE = 21
OFFSET_STEP = 0.5
INITIAL_HALF_WIDTH = 64.0
EXPANDED_HALF_WIDTH = 128.0
RATE_STATES = 33
RATE_SPAN = 0.12
EMISSION_DF = 4.0
EMISSION_TEMPER = 20.0
OUT_OF_SUPPORT_LOSS = 12.0
CALIBRATION_SLOPE_MIN = 0.25
CALIBRATION_SLOPE_MAX = 2.5
CALIBRATION_HUBER_MULTIPLIER = 1.5
CALIBRATION_SCALE_MIN = 8.0
CALIBRATION_SCALE_MAX = 60.0
CALIBRATION_DEFAULT_SCALE = 30.0
POSITION_MIN_SIGMA_STEP_FRACTION = 0.35
POSITION_KERNEL_SIGMA_RADIUS = 4.0
POSITION_KERNEL_MIN_RADIUS = 2
EDGE_MASS_THRESHOLD = 0.01
POSITION_EDGE_CELLS = 3
RATE_EDGE_CELLS = 2
CANDIDATE_GATE = 10.95
PROMISING_GATE = 11.75
MAX_PROJECTED_HOURS = 8.0
MAX_PEAK_RSS_GIB = 8.0
SMOKE_WELLS = 6
LATTICE_SENTINEL = -1e300
LATTICE_UNREACHABLE = -1e299
DATA_FINGERPRINT_ALGORITHM = "sha256(sorted_file_name_nul_file_bytes_nul)"
INPUT_FEATURES = (
    "horizontal.MD",
    "horizontal.Z",
    "horizontal.GR",
    "horizontal.TVT_input",
    "typewell.TVT",
    "typewell.GR",
    "parent.v006_oof_or_inference_path",
)


@dataclass(frozen=True)
class HMMConfig:
    """One predeclared HMM dynamics candidate."""

    name: str
    momentum: float
    rate_noise: float
    position_noise: float
    start_sigma: float
    initial_rate_sigma: float


HMM_CANDIDATES = (
    HMMConfig("hmm_wr21_slow", 0.998, 0.002, 0.175, 0.75, 0.01),
    HMMConfig("hmm_wr21_flex", 0.998, 0.004, 0.35, 0.75, 0.02),
)


def _fixed_settings() -> dict:
    """Stable JSON representation of every material non-candidate model constant."""
    return {
        "block_size": BLOCK_SIZE,
        "offset_step": OFFSET_STEP,
        "initial_half_width": INITIAL_HALF_WIDTH,
        "expanded_half_width": EXPANDED_HALF_WIDTH,
        "rate_states": RATE_STATES,
        "base_rate_span": RATE_SPAN,
        "emission_df": EMISSION_DF,
        "emission_temper": EMISSION_TEMPER,
        "out_of_support_loss": OUT_OF_SUPPORT_LOSS,
        "calibration_slope_min": CALIBRATION_SLOPE_MIN,
        "calibration_slope_max": CALIBRATION_SLOPE_MAX,
        "calibration_huber_multiplier": CALIBRATION_HUBER_MULTIPLIER,
        "calibration_scale_min": CALIBRATION_SCALE_MIN,
        "calibration_scale_max": CALIBRATION_SCALE_MAX,
        "calibration_default_scale": CALIBRATION_DEFAULT_SCALE,
        "position_min_sigma_step_fraction": POSITION_MIN_SIGMA_STEP_FRACTION,
        "position_kernel_sigma_radius": POSITION_KERNEL_SIGMA_RADIUS,
        "position_kernel_min_radius": POSITION_KERNEL_MIN_RADIUS,
        "edge_mass_threshold": EDGE_MASS_THRESHOLD,
        "position_edge_cells": POSITION_EDGE_CELLS,
        "rate_edge_cells": RATE_EDGE_CELLS,
    }


@dataclass(frozen=True)
class RegistrationBlock:
    """One non-overlapping suffix block and its real observed GR rows."""

    start: int
    stop: int
    anchor: int
    observed_indices: np.ndarray


@dataclass(frozen=True)
class SparseEmissions:
    """Block-anchor log emissions over the offset x rate state lattice."""

    rows: np.ndarray
    values: np.ndarray
    observed_counts: np.ndarray
    blocks: tuple[RegistrationBlock, ...]


@dataclass(frozen=True)
class ForwardBackwardResult:
    """Posterior moments and boundary diagnostics at registration anchors."""

    mean_offset: np.ndarray
    std_offset: np.ndarray
    mean_rate: np.ndarray
    std_rate: np.ndarray
    position_edge_mass: np.ndarray
    rate_edge_mass: np.ndarray
    log_likelihood: float


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


def data_fingerprint(root: Path) -> str:
    """Hash every raw CSV name and byte so provenance does not depend on file metadata."""
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.csv")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def build_registration_blocks(
    gr: np.ndarray | pd.Series, block_size: int = BLOCK_SIZE
) -> list[RegistrationBlock]:
    """Partition suffix rows once; every finite GR row belongs to exactly one block."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    values = np.asarray(gr, dtype=float)
    if values.ndim != 1:
        raise ValueError("gr must be one-dimensional")
    blocks: list[RegistrationBlock] = []
    for start in range(0, len(values), block_size):
        stop = min(start + block_size, len(values))
        observed = np.flatnonzero(np.isfinite(values[start:stop])).astype(np.int64) + start
        anchor = int(observed[len(observed) // 2]) if len(observed) else (start + stop - 1) // 2
        blocks.append(RegistrationBlock(start, stop, anchor, observed))
    return blocks


def _prepare_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    reference = pd.DataFrame(
        {
            "TVT": pd.to_numeric(typewell["TVT"], errors="coerce"),
            "GR": pd.to_numeric(typewell["GR"], errors="coerce"),
        }
    ).dropna()
    reference = reference.groupby("TVT", as_index=False, sort=True)["GR"].mean()
    if len(reference) < 3:
        raise ValueError("Typewell requires at least three finite TVT/GR rows")
    return (
        reference["TVT"].to_numpy(dtype=np.float64),
        reference["GR"].to_numpy(dtype=np.float64),
    )


def _prefix_calibration(
    horizontal: pd.DataFrame, reference_tvt: np.ndarray, reference_gr: np.ndarray
) -> tuple[float, float, float, float, pd.Series]:
    input_tvt = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
    prefix_mask = input_tvt.notna()
    if int(prefix_mask.sum()) < 3:
        raise ValueError("Horizontal well requires at least three visible TVT_input rows")
    known_tvt = input_tvt.loc[prefix_mask].to_numpy(dtype=float)
    known_gr = pd.to_numeric(horizontal.loc[prefix_mask, "GR"], errors="coerce").to_numpy(
        dtype=float
    )
    in_reference = (known_tvt >= reference_tvt[0]) & (known_tvt <= reference_tvt[-1])
    expected = np.interp(known_tvt, reference_tvt, reference_gr)
    valid = np.isfinite(known_gr) & np.isfinite(expected) & in_reference
    if int(valid.sum()) < 3:
        slope, intercept, scale = 1.0, 0.0, CALIBRATION_DEFAULT_SCALE
    else:
        x = expected[valid]
        y = known_gr[valid]
        if len(x) >= 20 and float(np.std(x)) > 1e-6:
            slope, intercept = (float(value) for value in np.polyfit(x, y, 1))
            slope = float(np.clip(slope, CALIBRATION_SLOPE_MIN, CALIBRATION_SLOPE_MAX))
        else:
            slope = 1.0
            intercept = float(np.median(y - x))
        for _ in range(3):
            residual = y - (slope * x + intercept)
            median = float(np.median(residual))
            mad = float(1.4826 * np.median(np.abs(residual - median)))
            robust_scale = max(mad, 1.0)
            weights = np.minimum(
                1.0,
                CALIBRATION_HUBER_MULTIPLIER * robust_scale / np.maximum(np.abs(residual), 1e-12),
            )
            design = np.column_stack((x, np.ones(len(x))))
            weighted = design * np.sqrt(weights)[:, None]
            solution, *_ = np.linalg.lstsq(weighted, y * np.sqrt(weights), rcond=None)
            slope = float(np.clip(solution[0], CALIBRATION_SLOPE_MIN, CALIBRATION_SLOPE_MAX))
            intercept = float(solution[1])
        residual = y - (slope * x + intercept)
        median = float(np.median(residual))
        scale = float(
            np.clip(
                1.4826 * np.median(np.abs(residual - median)),
                CALIBRATION_SCALE_MIN,
                CALIBRATION_SCALE_MAX,
            )
        )

    prefix = horizontal.loc[prefix_mask]
    tail = prefix.tail(30)
    dtvt = np.diff(pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(float))
    dz = np.diff(pd.to_numeric(tail["Z"], errors="coerce").to_numpy(float))
    dmd = np.diff(pd.to_numeric(tail["MD"], errors="coerce").to_numpy(float))
    valid_rate = np.isfinite(dtvt) & np.isfinite(dz) & np.isfinite(dmd) & (dmd > 0)
    initial_rate = (
        float(np.median((dtvt[valid_rate] + dz[valid_rate]) / dmd[valid_rate]))
        if int(valid_rate.sum()) >= 3
        else 0.0
    )
    return slope, intercept, scale, initial_rate, prefix_mask


@njit(cache=True, nogil=True)
def _interp_reference(tvt, reference_tvt, reference_gr):
    if tvt < reference_tvt[0] or tvt > reference_tvt[-1]:
        return np.nan
    low = 0
    high = len(reference_tvt) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if reference_tvt[middle] <= tvt:
            low = middle
        else:
            high = middle
    span = reference_tvt[high] - reference_tvt[low]
    if span <= 0.0:
        return reference_gr[low]
    fraction = (tvt - reference_tvt[low]) / span
    return reference_gr[low] * (1.0 - fraction) + reference_gr[high] * fraction


@njit(cache=True, nogil=True, parallel=True)
def _registration_emission_core(
    md,
    z,
    gr,
    center_u,
    offset_grid,
    rates,
    reference_tvt,
    reference_gr,
    slope,
    intercept,
    scale,
    starts,
    stops,
    anchors,
    degrees_freedom,
    temper,
    out_of_support_loss,
):
    n_blocks = len(starts)
    n_offsets = len(offset_grid)
    n_rates = len(rates)
    values = np.zeros((n_blocks, n_offsets, n_rates), dtype=np.float32)
    counts = np.zeros(n_blocks, dtype=np.int64)
    for block in prange(n_blocks):
        anchor = anchors[block]
        for row in range(starts[block], stops[block]):
            if np.isfinite(gr[row]):
                counts[block] += 1
        if counts[block] == 0:
            continue
        for position in range(n_offsets):
            anchor_u = center_u[anchor] + offset_grid[position]
            for rate_index in range(n_rates):
                total_loss = 0.0
                local_rate = rates[rate_index]
                for row in range(starts[block], stops[block]):
                    if not np.isfinite(gr[row]):
                        continue
                    row_u = anchor_u + local_rate * (md[row] - md[anchor])
                    expected = _interp_reference(row_u - z[row], reference_tvt, reference_gr)
                    if np.isfinite(expected):
                        residual = (gr[row] - (slope * expected + intercept)) / scale
                        total_loss += (
                            0.5
                            * (degrees_freedom + 1.0)
                            * np.log1p(residual * residual / degrees_freedom)
                        )
                    else:
                        total_loss += out_of_support_loss
                values[block, position, rate_index] = -total_loss / temper
    return values, counts


def build_registration_emissions(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    center_tvt: np.ndarray,
    offset_grid: np.ndarray,
    rates: np.ndarray,
    *,
    block_size: int = BLOCK_SIZE,
    degrees_freedom: float = EMISSION_DF,
    temper: float = EMISSION_TEMPER,
    out_of_support_loss: float = OUT_OF_SUPPORT_LOSS,
) -> tuple[SparseEmissions, dict]:
    """Build missing-safe window emissions without reading the suffix target."""
    if (
        block_size < 1
        or not np.isfinite([degrees_freedom, temper, out_of_support_loss]).all()
        or degrees_freedom <= 0.0
        or temper <= 0.0
        or out_of_support_loss < 0.0
    ):
        raise ValueError("Registration emission parameters are invalid")
    allowed = horizontal[["MD", "Z", "GR", "TVT_input"]].copy()
    evaluation = pd.to_numeric(allowed["TVT_input"], errors="coerce").isna().to_numpy()
    if not evaluation.any():
        raise ValueError("Horizontal well has no missing TVT_input suffix")
    if np.flatnonzero(evaluation)[0] <= np.flatnonzero(~evaluation)[-1]:
        raise ValueError("Horizontal well must contain one missing suffix")
    center_tvt = np.asarray(center_tvt, dtype=float)
    if center_tvt.shape != (int(evaluation.sum()),) or not np.isfinite(center_tvt).all():
        raise ValueError("parent center must contain one finite value per suffix row")

    reference_tvt, reference_gr = _prepare_typewell(typewell)
    slope, intercept, scale, initial_rate, _ = _prefix_calibration(
        allowed, reference_tvt, reference_gr
    )
    md = pd.to_numeric(allowed.loc[evaluation, "MD"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(allowed.loc[evaluation, "Z"], errors="coerce").to_numpy(float)
    gr = pd.to_numeric(allowed.loc[evaluation, "GR"], errors="coerce").to_numpy(float)
    if not np.isfinite(md).all() or not np.isfinite(z).all():
        raise ValueError("Suffix MD and Z must be finite")
    blocks = build_registration_blocks(gr, block_size)
    starts = np.asarray([block.start for block in blocks], dtype=np.int64)
    stops = np.asarray([block.stop for block in blocks], dtype=np.int64)
    anchors = np.asarray([block.anchor for block in blocks], dtype=np.int64)
    center_u = center_tvt + z
    values, counts = _registration_emission_core(
        md,
        z,
        gr,
        center_u,
        np.asarray(offset_grid, dtype=np.float64),
        np.asarray(rates, dtype=np.float64),
        reference_tvt,
        reference_gr,
        slope,
        intercept,
        scale,
        starts,
        stops,
        anchors,
        float(degrees_freedom),
        float(temper),
        float(out_of_support_loss),
    )
    sparse = SparseEmissions(anchors, values, counts, tuple(blocks))
    diagnostics = {
        "observed_gr_rows": int(np.isfinite(gr).sum()),
        "missing_gr_rows": int((~np.isfinite(gr)).sum()),
        "registration_blocks": len(blocks),
        "neutral_blocks": int((counts == 0).sum()),
        "affine_slope": slope,
        "affine_intercept": intercept,
        "gr_scale": scale,
        "initial_rate": initial_rate,
    }
    return sparse, diagnostics


@njit(cache=True, nogil=True)
def _logsumexp_matrix(values):
    maximum = LATTICE_SENTINEL
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if values[row, column] > maximum:
                maximum = values[row, column]
    if not np.isfinite(maximum) or maximum <= LATTICE_UNREACHABLE:
        return LATTICE_SENTINEL
    total = 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            total += np.exp(values[row, column] - maximum)
    if not np.isfinite(total) or total <= 0.0:
        return LATTICE_SENTINEL
    result = maximum + np.log(total)
    return result if np.isfinite(result) else LATTICE_SENTINEL


@njit(cache=True, nogil=True)
def _rate_log_transition(rates, delta_md, momentum, rate_noise):
    """Aggregated OU Gaussian transition over the actual anchor MD distance."""
    count = len(rates)
    transition = np.empty((count, count), dtype=np.float64)
    persistence = momentum**delta_md
    denominator = 1.0 - momentum * momentum
    if denominator > 1e-12:
        variance = rate_noise * rate_noise * (1.0 - persistence * persistence) / denominator
    else:
        variance = rate_noise * rate_noise * delta_md
    sigma = np.sqrt(max(variance, 1e-12))
    for source in range(count):
        expected = persistence * rates[source]
        maximum = LATTICE_SENTINEL
        for destination in range(count):
            standardized = (rates[destination] - expected) / sigma
            value = -0.5 * standardized * standardized
            transition[source, destination] = value
            if value > maximum:
                maximum = value
        total = 0.0
        for destination in range(count):
            total += np.exp(transition[source, destination] - maximum)
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("rate transition has no reachable destination")
        normalizer = maximum + np.log(total)
        if not np.isfinite(normalizer):
            raise ValueError("rate transition normalizer is non-finite")
        for destination in range(count):
            transition[source, destination] -= normalizer
    return transition


@njit(cache=True, nogil=True)
def _position_kernel(offset_step, delta_md, center_delta, source_rate, momentum, noise):
    if abs(1.0 - momentum) > 1e-12:
        integrated_rate = source_rate * (1.0 - momentum**delta_md) / (1.0 - momentum)
    else:
        integrated_rate = source_rate * delta_md
    mean_cells = (integrated_rate - center_delta) / offset_step
    sigma = max(noise * np.sqrt(delta_md), POSITION_MIN_SIGMA_STEP_FRACTION * offset_step)
    sigma_cells = sigma / offset_step
    radius = max(
        POSITION_KERNEL_MIN_RADIUS,
        int(np.ceil(POSITION_KERNEL_SIGMA_RADIUS * sigma_cells)),
    )
    shifts = np.arange(-radius, radius + 1, dtype=np.int64)
    log_weights = np.empty(len(shifts), dtype=np.float64)
    center_shift = int(np.floor(mean_cells + 0.5))
    maximum = LATTICE_SENTINEL
    for index in range(len(shifts)):
        shifts[index] += center_shift
        difference = (shifts[index] - mean_cells) / sigma_cells
        value = -0.5 * difference * difference
        log_weights[index] = value
        if value > maximum:
            maximum = value
    total = 0.0
    for value in log_weights:
        total += np.exp(value - maximum)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("position kernel is empty")
    normalizer = maximum + np.log(total)
    if not np.isfinite(normalizer):
        raise ValueError("position kernel normalizer is non-finite")
    for index in range(len(log_weights)):
        log_weights[index] -= normalizer
    return shifts, log_weights


@njit(cache=True, nogil=True)
def _position_log_normalizers(shifts, log_weights, positions):
    """Renormalize truncated offset transitions for every source position."""
    normalizers = np.empty(positions, dtype=np.float64)
    for source_position in range(positions):
        maximum = LATTICE_SENTINEL
        for kernel in range(len(shifts)):
            destination_position = source_position + shifts[kernel]
            if 0 <= destination_position < positions and log_weights[kernel] > maximum:
                maximum = log_weights[kernel]
        if maximum <= LATTICE_UNREACHABLE:
            normalizers[source_position] = LATTICE_SENTINEL
            continue
        total = 0.0
        for kernel in range(len(shifts)):
            destination_position = source_position + shifts[kernel]
            if 0 <= destination_position < positions:
                total += np.exp(log_weights[kernel] - maximum)
        if not np.isfinite(total) or total <= 0.0:
            normalizers[source_position] = LATTICE_SENTINEL
        else:
            normalizers[source_position] = maximum + np.log(total)
    return normalizers


@njit(cache=True, nogil=True)
def _posterior_summary(alpha, beta, offset_grid, rates, position_edge_cells, rate_edge_cells):
    maximum = LATTICE_SENTINEL
    for position in range(alpha.shape[0]):
        for rate in range(alpha.shape[1]):
            value = alpha[position, rate] + beta[position, rate]
            if value > maximum:
                maximum = value
    if not np.isfinite(maximum) or maximum <= LATTICE_UNREACHABLE:
        raise ValueError("posterior lattice is unreachable")
    total = 0.0
    offset_first = 0.0
    offset_second = 0.0
    rate_first = 0.0
    rate_second = 0.0
    position_edge = 0.0
    rate_edge = 0.0
    for position in range(alpha.shape[0]):
        for rate in range(alpha.shape[1]):
            probability = np.exp(alpha[position, rate] + beta[position, rate] - maximum)
            total += probability
            offset_first += probability * offset_grid[position]
            offset_second += probability * offset_grid[position] * offset_grid[position]
            rate_first += probability * rates[rate]
            rate_second += probability * rates[rate] * rates[rate]
            if position < position_edge_cells or position >= alpha.shape[0] - position_edge_cells:
                position_edge += probability
            if rate < rate_edge_cells or rate >= alpha.shape[1] - rate_edge_cells:
                rate_edge += probability
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("posterior normalization failed")
    offset_mean = offset_first / total
    rate_mean = rate_first / total
    offset_variance = max(offset_second / total - offset_mean * offset_mean, 0.0)
    rate_variance = max(rate_second / total - rate_mean * rate_mean, 0.0)
    return (
        offset_mean,
        np.sqrt(offset_variance),
        rate_mean,
        np.sqrt(rate_variance),
        position_edge / total,
        rate_edge / total,
    )


@njit(cache=True, nogil=True, parallel=True)
def _forward_backward_core(
    log_emission,
    anchor_md,
    center_u,
    offset_grid,
    rates,
    initial_md,
    initial_center_u,
    initial_offset,
    initial_rate,
    momentum,
    rate_noise,
    position_noise,
    start_sigma,
    initial_rate_sigma,
    position_edge_cells,
    rate_edge_cells,
):
    steps, positions, rate_count = log_emission.shape
    negative = LATTICE_SENTINEL
    alpha = np.empty((steps, positions, rate_count), dtype=np.float32)
    previous = np.empty((positions, rate_count), dtype=np.float64)
    for position in range(positions):
        position_score = -0.5 * ((offset_grid[position] - initial_offset) / start_sigma) ** 2
        for rate in range(rate_count):
            rate_score = -0.5 * ((rates[rate] - initial_rate) / initial_rate_sigma) ** 2
            previous[position, rate] = position_score + rate_score
    normalizer = _logsumexp_matrix(previous)
    if not np.isfinite(normalizer) or normalizer <= LATTICE_UNREACHABLE:
        raise ValueError("initial HMM lattice is unreachable")
    previous -= normalizer
    log_likelihood = 0.0

    for step in range(steps):
        previous_md = initial_md if step == 0 else anchor_md[step - 1]
        previous_center = initial_center_u if step == 0 else center_u[step - 1]
        delta_md = max(anchor_md[step] - previous_md, 1.0)
        center_delta = center_u[step] - previous_center
        rate_transition = _rate_log_transition(rates, delta_md, momentum, rate_noise)
        after_position = np.full((positions, rate_count), negative, dtype=np.float64)
        for source_rate in prange(rate_count):
            shifts, position_weights = _position_kernel(
                offset_grid[1] - offset_grid[0],
                delta_md,
                center_delta,
                rates[source_rate],
                momentum,
                position_noise,
            )
            position_normalizers = _position_log_normalizers(shifts, position_weights, positions)
            for destination_position in range(positions):
                maximum = negative
                for kernel in range(len(shifts)):
                    source_position = destination_position - shifts[kernel]
                    if (
                        0 <= source_position < positions
                        and position_normalizers[source_position] > LATTICE_UNREACHABLE
                    ):
                        value = (
                            previous[source_position, source_rate]
                            + position_weights[kernel]
                            - position_normalizers[source_position]
                        )
                        if value > maximum:
                            maximum = value
                if maximum <= negative / 2:
                    continue
                total = 0.0
                for kernel in range(len(shifts)):
                    source_position = destination_position - shifts[kernel]
                    if (
                        0 <= source_position < positions
                        and position_normalizers[source_position] > LATTICE_UNREACHABLE
                    ):
                        total += np.exp(
                            previous[source_position, source_rate]
                            + position_weights[kernel]
                            - position_normalizers[source_position]
                            - maximum
                        )
                after_position[destination_position, source_rate] = maximum + np.log(total)

        current = np.full((positions, rate_count), negative, dtype=np.float64)
        for position in prange(positions):
            for destination_rate in range(rate_count):
                maximum = negative
                for source_rate in range(rate_count):
                    value = (
                        after_position[position, source_rate]
                        + rate_transition[source_rate, destination_rate]
                    )
                    if value > maximum:
                        maximum = value
                if maximum <= LATTICE_UNREACHABLE:
                    continue
                total = 0.0
                for source_rate in range(rate_count):
                    total += np.exp(
                        after_position[position, source_rate]
                        + rate_transition[source_rate, destination_rate]
                        - maximum
                    )
                current[position, destination_rate] = (
                    maximum + np.log(total) + log_emission[step, position, destination_rate]
                )
        normalizer = _logsumexp_matrix(current)
        if not np.isfinite(normalizer) or normalizer <= LATTICE_UNREACHABLE:
            raise ValueError("forward HMM lattice is unreachable")
        log_likelihood += normalizer
        current -= normalizer
        alpha[step] = current.astype(np.float32)
        previous = current

    mean_offset = np.empty(steps, dtype=np.float64)
    std_offset = np.empty(steps, dtype=np.float64)
    mean_rate = np.empty(steps, dtype=np.float64)
    std_rate = np.empty(steps, dtype=np.float64)
    position_edge = np.empty(steps, dtype=np.float64)
    rate_edge = np.empty(steps, dtype=np.float64)
    beta_next = np.zeros((positions, rate_count), dtype=np.float64)
    summary = _posterior_summary(
        alpha[steps - 1],
        beta_next,
        offset_grid,
        rates,
        position_edge_cells,
        rate_edge_cells,
    )
    (
        mean_offset[steps - 1],
        std_offset[steps - 1],
        mean_rate[steps - 1],
        std_rate[steps - 1],
        position_edge[steps - 1],
        rate_edge[steps - 1],
    ) = summary

    for step in range(steps - 1, 0, -1):
        delta_md = max(anchor_md[step] - anchor_md[step - 1], 1.0)
        center_delta = center_u[step] - center_u[step - 1]
        rate_transition = _rate_log_transition(rates, delta_md, momentum, rate_noise)
        after_rate = np.full((positions, rate_count), negative, dtype=np.float64)
        for destination_position in prange(positions):
            for source_rate in range(rate_count):
                maximum = negative
                for destination_rate in range(rate_count):
                    value = (
                        rate_transition[source_rate, destination_rate]
                        + log_emission[step, destination_position, destination_rate]
                        + beta_next[destination_position, destination_rate]
                    )
                    if value > maximum:
                        maximum = value
                if maximum <= LATTICE_UNREACHABLE:
                    continue
                total = 0.0
                for destination_rate in range(rate_count):
                    total += np.exp(
                        rate_transition[source_rate, destination_rate]
                        + log_emission[step, destination_position, destination_rate]
                        + beta_next[destination_position, destination_rate]
                        - maximum
                    )
                after_rate[destination_position, source_rate] = maximum + np.log(total)

        beta_current = np.full((positions, rate_count), negative, dtype=np.float64)
        for source_rate in prange(rate_count):
            shifts, position_weights = _position_kernel(
                offset_grid[1] - offset_grid[0],
                delta_md,
                center_delta,
                rates[source_rate],
                momentum,
                position_noise,
            )
            position_normalizers = _position_log_normalizers(shifts, position_weights, positions)
            for source_position in range(positions):
                maximum = negative
                if position_normalizers[source_position] <= LATTICE_UNREACHABLE:
                    continue
                for kernel in range(len(shifts)):
                    destination_position = source_position + shifts[kernel]
                    if 0 <= destination_position < positions:
                        value = (
                            position_weights[kernel]
                            - position_normalizers[source_position]
                            + after_rate[destination_position, source_rate]
                        )
                        if value > maximum:
                            maximum = value
                if maximum <= negative / 2:
                    continue
                total = 0.0
                for kernel in range(len(shifts)):
                    destination_position = source_position + shifts[kernel]
                    if 0 <= destination_position < positions:
                        total += np.exp(
                            position_weights[kernel]
                            - position_normalizers[source_position]
                            + after_rate[destination_position, source_rate]
                            - maximum
                        )
                beta_current[source_position, source_rate] = maximum + np.log(total)
        beta_normalizer = _logsumexp_matrix(beta_current)
        if not np.isfinite(beta_normalizer) or beta_normalizer <= LATTICE_UNREACHABLE:
            raise ValueError("backward HMM lattice is unreachable")
        beta_current -= beta_normalizer
        beta_next = beta_current
        summary = _posterior_summary(
            alpha[step - 1],
            beta_next,
            offset_grid,
            rates,
            position_edge_cells,
            rate_edge_cells,
        )
        (
            mean_offset[step - 1],
            std_offset[step - 1],
            mean_rate[step - 1],
            std_rate[step - 1],
            position_edge[step - 1],
            rate_edge[step - 1],
        ) = summary

    return (
        mean_offset,
        std_offset,
        mean_rate,
        std_rate,
        position_edge,
        rate_edge,
        log_likelihood,
    )


def forward_backward(
    emissions: SparseEmissions,
    anchor_md: np.ndarray,
    center_u: np.ndarray,
    offset_grid: np.ndarray,
    rates: np.ndarray,
    *,
    initial_md: float,
    initial_center_u: float,
    initial_offset: float,
    initial_rate: float,
    config: HMMConfig,
    position_edge_cells: int = POSITION_EDGE_CELLS,
    rate_edge_cells: int = RATE_EDGE_CELLS,
) -> ForwardBackwardResult:
    """Run normalized exact forward-backward on block-anchor HMM states."""
    anchor_md = np.asarray(anchor_md, dtype=np.float64)
    center_u = np.asarray(center_u, dtype=np.float64)
    offset_grid = np.asarray(offset_grid, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    if emissions.values.ndim != 3:
        raise ValueError("emission values must have shape (blocks, offsets, rates)")
    expected_shape = (len(anchor_md), len(offset_grid), len(rates))
    if emissions.values.shape != expected_shape:
        raise ValueError(f"emission shape {emissions.values.shape} != {expected_shape}")
    if len(emissions.rows) != len(anchor_md) or len(emissions.observed_counts) != len(anchor_md):
        raise ValueError("emission metadata must contain one entry per anchor")
    if len(anchor_md) == 0 or len(offset_grid) < 3 or len(rates) < 3:
        raise ValueError("forward-backward requires blocks and at least three states per axis")
    if len(center_u) != len(anchor_md):
        raise ValueError("center_u must contain one value per anchor")
    if not (
        np.isfinite(anchor_md).all()
        and np.isfinite(center_u).all()
        and np.isfinite(offset_grid).all()
        and np.isfinite(rates).all()
        and np.isfinite(emissions.values).all()
        and np.isfinite([initial_md, initial_center_u, initial_offset, initial_rate]).all()
    ):
        raise ValueError("forward-backward inputs must be finite")
    if anchor_md[0] <= initial_md or not np.all(np.diff(anchor_md) > 0):
        raise ValueError("anchor MD values must be strictly increasing")
    offset_differences = np.diff(offset_grid)
    if not np.all(offset_differences > 0) or not np.allclose(
        offset_differences,
        offset_differences[0],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("offset grid must be strictly increasing and regular")
    if not np.all(np.diff(rates) > 0):
        raise ValueError("rate states must be strictly increasing")
    config_values = np.asarray(
        [
            config.momentum,
            config.rate_noise,
            config.position_noise,
            config.start_sigma,
            config.initial_rate_sigma,
        ],
        dtype=float,
    )
    if (
        not np.isfinite(config_values).all()
        or not 0.0 < config.momentum <= 1.0
        or (config_values[1:] <= 0.0).any()
    ):
        raise ValueError("HMM dynamics parameters must be finite and positive")
    if not 0 < position_edge_cells * 2 < len(offset_grid):
        raise ValueError("position edge cells must leave an interior state")
    if not 0 < rate_edge_cells * 2 < len(rates):
        raise ValueError("rate edge cells must leave an interior state")
    values = _forward_backward_core(
        np.asarray(emissions.values, dtype=np.float32),
        anchor_md,
        center_u,
        offset_grid,
        rates,
        float(initial_md),
        float(initial_center_u),
        float(initial_offset),
        float(initial_rate),
        float(config.momentum),
        float(config.rate_noise),
        float(config.position_noise),
        float(config.start_sigma),
        float(config.initial_rate_sigma),
        int(position_edge_cells),
        int(rate_edge_cells),
    )
    result = ForwardBackwardResult(*values)
    moment_arrays = (
        result.mean_offset,
        result.std_offset,
        result.mean_rate,
        result.std_rate,
        result.position_edge_mass,
        result.rate_edge_mass,
    )
    if not np.isfinite(result.log_likelihood) or not all(
        np.isfinite(array).all() for array in moment_arrays
    ):
        raise ValueError("HMM posterior moments or likelihood are non-finite")
    if (result.std_offset < 0).any() or (result.std_rate < 0).any():
        raise ValueError("HMM posterior standard deviations must be non-negative")
    for edge_mass in (result.position_edge_mass, result.rate_edge_mass):
        if (edge_mass < 0).any() or (edge_mass > 1).any():
            raise ValueError("HMM posterior edge mass must lie in [0, 1]")
    return result


def needs_corridor_expansion(
    result: ForwardBackwardResult, threshold: float = EDGE_MASS_THRESHOLD
) -> bool:
    """Use target-free position posterior mass to decide whether support is too narrow."""
    return bool(np.max(result.position_edge_mass, initial=0.0) > threshold)


def _evaluation_arrays(horizontal: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    allowed = horizontal[["MD", "Z", "GR", "TVT_input"]]
    input_tvt = pd.to_numeric(allowed["TVT_input"], errors="coerce")
    evaluation = input_tvt.isna().to_numpy()
    observed = ~evaluation
    if not evaluation.any() or not observed.any():
        raise ValueError("Horizontal well requires a visible prefix and missing suffix")
    if np.flatnonzero(evaluation)[0] <= np.flatnonzero(observed)[-1]:
        raise ValueError("Horizontal well must contain one contiguous missing suffix")
    md = pd.to_numeric(allowed.loc[evaluation, "MD"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(allowed.loc[evaluation, "Z"], errors="coerce").to_numpy(float)
    if not np.isfinite(md).all() or not np.isfinite(z).all() or not np.all(np.diff(md) > 0):
        raise ValueError("Suffix MD must increase strictly and suffix MD/Z must be finite")
    return evaluation, md, z


def _candidate_state_grids(initial_rate: float, half_width: float) -> tuple[np.ndarray, np.ndarray]:
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    offset_grid = np.arange(
        -half_width, half_width + 0.5 * OFFSET_STEP, OFFSET_STEP, dtype=np.float64
    )
    span = max(RATE_SPAN, abs(float(initial_rate)) + 0.04)
    rates = np.linspace(-span, span, RATE_STATES, dtype=np.float64)
    return offset_grid, rates


def _interpolate_anchor_posterior(
    md: np.ndarray,
    z: np.ndarray,
    anchor_rows: np.ndarray,
    center_tvt: np.ndarray,
    result: ForwardBackwardResult,
    *,
    initial_md: float,
    initial_u: float,
    config: HMMConfig,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_md = md[anchor_rows]
    anchor_center_u = center_tvt[anchor_rows] + z[anchor_rows]
    anchor_mean_u = anchor_center_u + result.mean_offset
    interpolation_md = np.concatenate(([initial_md], anchor_md))
    interpolation_u = np.concatenate(([initial_u], anchor_mean_u))
    mean_u = np.interp(md, interpolation_md, interpolation_u)
    std_u = np.interp(
        md,
        interpolation_md,
        np.concatenate(([0.0], result.std_offset)),
    )
    after_last = md > anchor_md[-1]
    if after_last.any():
        distance = md[after_last] - anchor_md[-1]
        mean_u[after_last] = anchor_mean_u[-1] + result.mean_rate[-1] * distance
        std_u[after_last] = np.sqrt(
            result.std_offset[-1] ** 2
            + result.std_rate[-1] ** 2 * distance**2
            + config.position_noise**2 * distance
        )
    return mean_u - z, std_u


def _prepared_smoother(
    horizontal: pd.DataFrame,
    center_tvt: np.ndarray,
    emissions: SparseEmissions,
    offset_grid: np.ndarray,
    rates: np.ndarray,
    initial_rate: float,
    config: HMMConfig,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray, ForwardBackwardResult, dict]:
    evaluation, md, z = _evaluation_arrays(horizontal)
    input_tvt = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
    last = horizontal.loc[~evaluation].iloc[-1]
    initial_md = float(pd.to_numeric(pd.Series([last["MD"]]), errors="coerce").iloc[0])
    initial_z = float(pd.to_numeric(pd.Series([last["Z"]]), errors="coerce").iloc[0])
    initial_tvt = float(input_tvt.loc[~evaluation].iloc[-1])
    initial_u = initial_tvt + initial_z
    anchor_rows = emissions.rows.astype(np.int64)
    center_u_all = center_tvt + z
    result = forward_backward(
        emissions,
        md[anchor_rows],
        center_u_all[anchor_rows],
        offset_grid,
        rates,
        initial_md=initial_md,
        initial_center_u=initial_u,
        initial_offset=0.0,
        initial_rate=initial_rate,
        config=config,
    )
    prediction, posterior_std = _interpolate_anchor_posterior(
        md,
        z,
        anchor_rows,
        center_tvt,
        result,
        initial_md=initial_md,
        initial_u=initial_u,
        config=config,
    )
    diagnostics = {
        "candidate": config.name,
        "half_width": half_width,
        "max_position_edge_mass": float(np.max(result.position_edge_mass, initial=0.0)),
        "max_rate_edge_mass": float(np.max(result.rate_edge_mass, initial=0.0)),
        "mean_posterior_std": float(np.mean(posterior_std)),
        "max_posterior_std": float(np.max(posterior_std, initial=0.0)),
        "log_likelihood": result.log_likelihood,
    }
    return prediction, posterior_std, result, diagnostics


def run_smoother(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    parent_tvt: np.ndarray,
    config: HMMConfig,
    *,
    half_width: float = INITIAL_HALF_WIDTH,
    block_size: int = BLOCK_SIZE,
) -> tuple[np.ndarray, np.ndarray, ForwardBackwardResult, dict]:
    """Run one target-free HMM candidate at one fixed corridor width."""
    evaluation, _, _ = _evaluation_arrays(horizontal)
    parent_tvt = np.asarray(parent_tvt, dtype=float)
    if parent_tvt.shape == (len(horizontal),):
        parent_tvt = parent_tvt[evaluation]
    if parent_tvt.shape != (int(evaluation.sum()),) or not np.isfinite(parent_tvt).all():
        raise ValueError("parent_tvt must contain finite predictions for every suffix row")
    reference_tvt, reference_gr = _prepare_typewell(typewell)
    _, _, _, initial_rate, _ = _prefix_calibration(
        horizontal[["MD", "Z", "GR", "TVT_input"]], reference_tvt, reference_gr
    )
    offset_grid, rates = _candidate_state_grids(initial_rate, half_width)
    emissions, emission_diagnostics = build_registration_emissions(
        horizontal,
        typewell,
        parent_tvt,
        offset_grid,
        rates,
        block_size=block_size,
    )
    prediction, posterior_std, result, diagnostics = _prepared_smoother(
        horizontal,
        parent_tvt,
        emissions,
        offset_grid,
        rates,
        initial_rate,
        config,
        half_width,
    )
    diagnostics.update(emission_diagnostics)
    return prediction, posterior_std, result, diagnostics


_PARENT_MODULE = None


def _parent_source_path() -> Path:
    return Path(__file__).resolve().with_name("rogii_state_space.py")


def _load_parent_module():
    global _PARENT_MODULE
    if _PARENT_MODULE is not None:
        return _PARENT_MODULE
    path = _parent_source_path()
    spec = importlib.util.spec_from_file_location("_rogii_v006_parent", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load v006 parent module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _PARENT_MODULE = module
    return module


def _parent_particle_path(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    particles: int = 500,
    seeds: int = 32,
) -> tuple[np.ndarray, dict]:
    parent = _load_parent_module()
    predictions, diagnostics = parent.particle_candidates(
        horizontal[["MD", "Z", "GR", "TVT_input"]],
        typewell[["TVT", "GR"]],
        particles=particles,
        seeds=seeds,
        scales=(12.0,),
        holds=(0.2,),
    )
    values = predictions[PARENT_CANDIDATE].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("v006 parent produced non-finite predictions")
    return values, diagnostics


def hmm_candidates(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    parent_tvt: np.ndarray | None = None,
    configs: Iterable[HMMConfig] = HMM_CANDIDATES,
    *,
    initial_half_width: float = INITIAL_HALF_WIDTH,
    expanded_half_width: float = EXPANDED_HALF_WIDTH,
    edge_threshold: float = EDGE_MASS_THRESHOLD,
    block_size: int = BLOCK_SIZE,
    parent_particles: int = 500,
    parent_seeds: int = 32,
) -> tuple[pd.DataFrame, dict]:
    """Generate both pure HMM candidates with one target-free 64 -> 128 audit."""
    configs = tuple(configs)
    if not configs:
        raise ValueError("At least one HMM config is required")
    names = [config.name for config in configs]
    if len(set(names)) != len(names):
        raise ValueError("HMM config names must be unique")
    evaluation, _, _ = _evaluation_arrays(horizontal)
    evaluation_indices = np.flatnonzero(evaluation)
    parent_diagnostics = None
    if parent_tvt is None:
        parent_tvt, parent_diagnostics = _parent_particle_path(
            horizontal,
            typewell,
            particles=parent_particles,
            seeds=parent_seeds,
        )
    parent_tvt = np.asarray(parent_tvt, dtype=float)
    if parent_tvt.shape == (len(horizontal),):
        parent_tvt = parent_tvt[evaluation]
    if parent_tvt.shape != (len(evaluation_indices),) or not np.isfinite(parent_tvt).all():
        raise ValueError("parent_tvt must align with the natural suffix")

    reference_tvt, reference_gr = _prepare_typewell(typewell)
    _, _, _, initial_rate, _ = _prefix_calibration(
        horizontal[["MD", "Z", "GR", "TVT_input"]], reference_tvt, reference_gr
    )

    def run_width(half_width):
        offset_grid, rates = _candidate_state_grids(initial_rate, half_width)
        emissions, emission_diagnostics = build_registration_emissions(
            horizontal,
            typewell,
            parent_tvt,
            offset_grid,
            rates,
            block_size=block_size,
        )
        outputs = {}
        for config in configs:
            outputs[config.name] = _prepared_smoother(
                horizontal,
                parent_tvt,
                emissions,
                offset_grid,
                rates,
                initial_rate,
                config,
                half_width,
            )
        return outputs, emission_diagnostics

    outputs, emission_diagnostics = run_width(initial_half_width)
    expanded = any(needs_corridor_expansion(item[2], edge_threshold) for item in outputs.values())
    final_width = initial_half_width
    if expanded:
        outputs, emission_diagnostics = run_width(expanded_half_width)
        final_width = expanded_half_width

    frame = pd.DataFrame({"_row_index": evaluation_indices})
    candidate_diagnostics = {}
    unresolved_position = False
    unresolved_rate = False
    for config in configs:
        prediction, posterior_std, result, diagnostics = outputs[config.name]
        if not np.isfinite(prediction).all() or not np.isfinite(posterior_std).all():
            raise ValueError(f"Candidate {config.name} produced non-finite posterior moments")
        frame[config.name] = prediction.astype(np.float32)
        frame[f"{config.name}__std"] = posterior_std.astype(np.float32)
        position_failed = needs_corridor_expansion(result, edge_threshold)
        rate_failed = bool(np.max(result.rate_edge_mass, initial=0.0) > edge_threshold)
        diagnostics["unresolved_position_boundary"] = position_failed
        diagnostics["unresolved_rate_boundary"] = rate_failed
        candidate_diagnostics[config.name] = diagnostics
        unresolved_position |= position_failed
        unresolved_rate |= rate_failed

    diagnostics = {
        **emission_diagnostics,
        "expanded_corridor": expanded,
        "final_half_width": final_width,
        "any_candidate_unresolved_position_boundary": unresolved_position,
        "any_candidate_unresolved_rate_boundary": unresolved_rate,
        "parent_candidate": PARENT_CANDIDATE,
        "parent_diagnostics": parent_diagnostics,
        "candidates": candidate_diagnostics,
    }
    return frame, diagnostics


def choose_nested_candidates(summary: pd.DataFrame) -> tuple[list[dict], str]:
    """Select a candidate for each outer fold using only the other folds."""
    required = {"well", "fold", "candidate", "squared_error", "rows"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Candidate summary missing columns: {missing}")
    if summary.empty:
        raise ValueError("Candidate summary is empty")
    if not np.isfinite(summary["squared_error"].to_numpy(dtype=float)).all():
        raise ValueError("Candidate squared errors must be finite")
    rows = summary["rows"].to_numpy(dtype=float)
    if not np.isfinite(rows).all() or (rows <= 0).any():
        raise ValueError("Candidate row counts must be finite and positive")
    expected_candidates = frozenset(str(value) for value in summary["candidate"].unique())
    if not expected_candidates:
        raise ValueError("Candidate summary has no candidates")
    for (well, fold), group in summary.groupby(["well", "fold"], sort=False):
        candidates = [str(value) for value in group["candidate"]]
        if len(candidates) != len(set(candidates)) or frozenset(candidates) != expected_candidates:
            raise ValueError(f"Incomplete candidate grid for well={well}, fold={fold}")
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
        if validation.empty or int(validation["rows"].sum()) <= 0:
            raise ValueError(f"Selected candidate has no validation rows for fold {fold}")
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


def atomic_publish_parquet(partial_path: Path, final_path: Path) -> None:
    """Validate a closed parquet then atomically publish it without overwriting."""
    if pq is None:
        raise RuntimeError("pyarrow is required to validate parquet artifacts")
    partial_path = Path(partial_path)
    final_path = Path(final_path)
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite {final_path}")
    if not partial_path.is_file():
        raise FileNotFoundError(partial_path)
    parquet = pq.ParquetFile(partial_path)
    if parquet.metadata is None or parquet.metadata.num_rows < 1:
        raise ValueError(f"Partial parquet has no rows: {partial_path}")
    os.replace(partial_path, final_path)


def scored_row_counts(train_root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(train_root.glob(f"*{HORIZONTAL_SUFFIX}")):
        frame = pd.read_csv(path, usecols=["TVT_input"])
        records.append(
            {"well": well_id_from_path(path), "scored_rows": int(frame["TVT_input"].isna().sum())}
        )
    if not records:
        raise FileNotFoundError(f"No horizontal-well CSV files found under {train_root}")
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
    path = workspace / "reports" / "canonical_outer_folds_v001.csv"
    expected = assign_balanced_folds(
        scored_row_counts(workspace / "data" / "raw" / "train"), n_splits
    )
    if not path.exists():
        raise FileNotFoundError(f"Canonical fold manifest is missing: {path}")
    current = pd.read_csv(path)
    pd.testing.assert_frame_equal(current, expected, check_dtype=False)
    return path, expected


def next_feature_path(workspace: Path) -> Path:
    root = workspace / "data" / "features"
    root.mkdir(parents=True, exist_ok=True)
    versions = [int(path.stem[1:]) for path in root.glob("v[0-9][0-9][0-9].parquet")]
    return root / f"v{max(versions, default=0) + 1:03d}.parquet"


class ParentOOFReader:
    """Stream and verify the immutable v006 parent path one well at a time."""

    def __init__(self, workspace: Path):
        if pq is None:
            raise RuntimeError("pyarrow is required to read the v006 parent feature")
        self.feature_path = workspace / "data" / "features" / "v003.parquet"
        self.oof_path = workspace / "models" / PARENT_VERSION / "oof_preds.npy"
        self.parquet = pq.ParquetFile(self.feature_path)
        self.oof = np.load(self.oof_path, mmap_mode="r")
        self.cursor = 0

        scores = json.loads(
            (workspace / "models" / PARENT_VERSION / "cv_scores.json").read_text(encoding="utf-8")
        )
        selected = {item["selected_candidate"] for item in scores["fold_selections"]}
        if selected != {PARENT_CANDIDATE} or scores.get("final_candidate") != PARENT_CANDIDATE:
            raise ValueError("v006 parent selections no longer match the locked corridor parent")

    def read_well(self, row_group: int, well: str, evaluation_indices: np.ndarray) -> np.ndarray:
        table = self.parquet.read_row_group(
            row_group,
            columns=["_well_id", "_row_index", PARENT_CANDIDATE],
        )
        frame = table.to_pandas()
        if frame["_well_id"].nunique() != 1 or str(frame["_well_id"].iloc[0]) != well:
            raise ValueError(f"v006 row group {row_group} does not match well {well}")
        row_indices = frame["_row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(row_indices, np.asarray(evaluation_indices, dtype=np.int64)):
            raise ValueError(f"v006 row indices do not match raw suffix for well {well}")
        values = frame[PARENT_CANDIDATE].to_numpy(dtype=float)
        stop = self.cursor + len(values)
        if stop > len(self.oof) or not np.allclose(
            values, np.asarray(self.oof[self.cursor : stop], dtype=float), rtol=0.0, atol=1e-6
        ):
            raise ValueError(f"v006 feature and OOF predictions disagree for well {well}")
        self.cursor = stop
        return values

    def finish(self, expected_groups: int) -> None:
        if self.parquet.num_row_groups != expected_groups:
            raise ValueError(
                f"v006 parent has {self.parquet.num_row_groups} row groups, expected {expected_groups}"
            )
        if self.cursor != len(self.oof):
            raise ValueError(f"Consumed {self.cursor} of {len(self.oof)} v006 parent predictions")


def build_training_candidates(
    workspace: Path,
    manifest: pd.DataFrame,
    feature_path: Path,
    *,
    configs: Iterable[HMMConfig] = HMM_CANDIDATES,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Stream both HMM candidates and atomically publish the immutable feature parquet."""
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required to stream HMM candidate features")
    configs = tuple(configs)
    if tuple(config.name for config in configs) != tuple(config.name for config in HMM_CANDIDATES):
        raise ValueError("Full v007 training requires the exact predeclared candidate grid")
    feature_path = Path(feature_path)
    partial_path = Path(f"{feature_path}.partial")
    for path in (feature_path, partial_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    paths = sorted((workspace / "data" / "raw" / "train").glob(f"*{HORIZONTAL_SUFFIX}"))
    if not paths:
        raise FileNotFoundError("No ROGII training wells found")
    fold_map = manifest.set_index("well")["fold"].to_dict()
    if set(fold_map) != {well_id_from_path(path) for path in paths}:
        raise ValueError("Canonical manifest wells do not match training files")
    parent_reader = ParentOOFReader(workspace)
    summary_records: list[dict] = []
    diagnostics: list[dict] = []
    failures: list[dict] = []
    writer = None
    try:
        for index, horizontal_path in enumerate(paths):
            well = well_id_from_path(horizontal_path)
            horizontal = pd.read_csv(horizontal_path)
            evaluation = pd.to_numeric(horizontal["TVT_input"], errors="coerce").isna().to_numpy()
            evaluation_indices = np.flatnonzero(evaluation)
            target = pd.to_numeric(horizontal.loc[evaluation, "TVT"], errors="coerce").to_numpy(
                dtype=float
            )
            if not np.isfinite(target).all():
                raise ValueError(f"Training target is non-finite for well {well}")
            parent_tvt = parent_reader.read_well(index, well, evaluation_indices)
            inference_horizontal = horizontal[["MD", "Z", "GR", "TVT_input"]]
            typewell = pd.read_csv(
                horizontal_path.with_name(f"{well}{TYPEWELL_SUFFIX}"),
                usecols=lambda column: column in {"TVT", "GR"},
            )
            try:
                candidates, well_diagnostics = hmm_candidates(
                    inference_horizontal,
                    typewell,
                    parent_tvt,
                    configs,
                )
            except Exception as exc:
                last_tvt = float(
                    pd.to_numeric(horizontal.loc[~evaluation, "TVT_input"], errors="coerce").iloc[
                        -1
                    ]
                )
                candidates = pd.DataFrame({"_row_index": evaluation_indices})
                for config in configs:
                    candidates[config.name] = np.full(len(target), last_tvt, dtype=np.float32)
                    candidates[f"{config.name}__std"] = np.full(
                        len(target), EXPANDED_HALF_WIDTH, dtype=np.float32
                    )
                well_diagnostics = {
                    "candidates": {
                        config.name: {
                            "unresolved_position_boundary": True,
                            "unresolved_rate_boundary": True,
                        }
                        for config in configs
                    }
                }
                failures.append({"well": well, "error": f"{type(exc).__name__}: {exc}"})

            frame = pd.DataFrame(
                {
                    "_id": [f"{well}_{row}" for row in evaluation_indices],
                    "_well_id": well,
                    "_row_index": evaluation_indices,
                    "_target": target,
                    "fold": int(fold_map[well]),
                    "parent_v006": parent_tvt.astype(np.float32),
                }
            )
            for config in configs:
                values = candidates[config.name].to_numpy(dtype=np.float32)
                std_values = candidates[f"{config.name}__std"].to_numpy(dtype=np.float32)
                if len(values) != len(frame) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid candidate {config.name} for well {well}")
                frame[config.name] = values
                frame[f"{config.name}__std"] = std_values
                error = values.astype(float) - target
                squared_error = float(np.dot(error, error))
                candidate_diagnostics = well_diagnostics["candidates"][config.name]
                summary_records.append(
                    {
                        "well": well,
                        "fold": int(fold_map[well]),
                        "candidate": config.name,
                        "squared_error": squared_error,
                        "rows": len(frame),
                        "rmse": float(np.sqrt(squared_error / len(frame))),
                        "unresolved_position_boundary": bool(
                            candidate_diagnostics["unresolved_position_boundary"]
                        ),
                        "unresolved_rate_boundary": bool(
                            candidate_diagnostics["unresolved_rate_boundary"]
                        ),
                    }
                )
            diagnostics.append({"well": well, **well_diagnostics})
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(partial_path, table.schema, compression="zstd")
            writer.write_table(table)
            if (index + 1) % 10 == 0 or index + 1 == len(paths):
                print(
                    f"hmm progress: {index + 1}/{len(paths)} wells ({time.strftime('%H:%M:%S')})",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
    parent_reader.finish(len(paths))
    atomic_publish_parquet(partial_path, feature_path)
    return pd.DataFrame(summary_records), diagnostics, failures


def materialize_oof(
    feature_path: Path, selections: list[dict], configs: Iterable[HMMConfig] = HMM_CANDIDATES
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    configs = tuple(configs)
    names = [config.name for config in configs]
    columns = ["_id", "_well_id", "_row_index", "_target", "fold"]
    frame = pd.read_parquet(
        feature_path,
        columns=[*columns, *names, *(f"{name}__std" for name in names)],
    )
    prediction = np.full(len(frame), np.nan, dtype=float)
    posterior_std = np.full(len(frame), np.nan, dtype=float)
    for selection in selections:
        fold = int(selection["fold"])
        candidate = str(selection["selected_candidate"])
        mask = frame["fold"].to_numpy(dtype=int) == fold
        prediction[mask] = frame.loc[mask, candidate].to_numpy(dtype=float)
        posterior_std[mask] = frame.loc[mask, f"{candidate}__std"].to_numpy(dtype=float)
    if not np.isfinite(prediction).all() or not np.isfinite(posterior_std).all():
        raise ValueError("Nested HMM OOF moments contain non-finite values")
    return frame[columns], prediction, posterior_std


def baseline_scores_on_manifest(workspace: Path, manifest: pd.DataFrame) -> dict:
    rows = pd.read_parquet(
        workspace / "models" / PARENT_VERSION / "oof_rows.parquet",
        columns=["_well_id", "_target"],
    )
    prediction = np.load(workspace / "models" / PARENT_VERSION / "oof_preds.npy")
    if len(rows) != len(prediction):
        raise ValueError("v006 OOF rows and predictions differ in length")
    fold_values = rows["_well_id"].map(manifest.set_index("well")["fold"]).to_numpy(dtype=int)
    target = rows["_target"].to_numpy(dtype=float)
    return {
        "overall": rmse(target, prediction),
        "fold_scores": [
            rmse(target[fold_values == fold], prediction[fold_values == fold])
            for fold in sorted(manifest["fold"].unique())
        ],
    }


def candidate_decision(
    overall: float,
    folds_better: int,
    *,
    training_failures: int = 0,
    selected_boundary_wells: int = 0,
    final_boundary_wells: int = 0,
    test_failures: int = 0,
) -> str:
    """Apply the predeclared v007 promotion gate without side effects."""
    if not np.isfinite(overall) or folds_better < 0:
        raise ValueError("Gate inputs must be finite and non-negative")
    counts = (
        training_failures,
        selected_boundary_wells,
        final_boundary_wells,
        test_failures,
    )
    if any(int(value) != value or value < 0 for value in counts):
        raise ValueError("Gate failure counts must be non-negative integers")
    if overall <= CANDIDATE_GATE and folds_better >= 4:
        if training_failures:
            return "candidate_blocked_training_failures"
        if selected_boundary_wells or final_boundary_wells:
            return "candidate_blocked_boundaries"
        if test_failures:
            return "candidate_blocked_test_failures"
        return "candidate_ready"
    if overall < PROMISING_GATE and folds_better >= 3:
        return "promising_continue"
    return "exhausted"


def _peak_rss_gib() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    divisor = 1024.0**2 if sys.platform != "darwin" else 1024.0**3
    return usage / divisor


def _smoke_inventory(train_root: Path) -> dict:
    paths = sorted(train_root.glob(f"*{HORIZONTAL_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(f"No horizontal wells found under {train_root}")
    counts = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["TVT_input"])
        counts.append((int(frame["TVT_input"].isna().sum()), path))
    counts.sort(key=lambda item: (item[0], item[1].name))
    if len(counts) < SMOKE_WELLS:
        raise ValueError(f"v007 smoke requires at least {SMOKE_WELLS} training wells")
    positions = np.linspace(0, len(counts) - 1, SMOKE_WELLS, dtype=int)
    selected = [counts[int(position)][1] for position in positions]
    return {
        "paths": paths,
        "selected": selected,
        "selection_positions": [int(position) for position in positions],
        "total_suffix_rows": int(sum(count for count, _ in counts)),
        "data_fingerprint": data_fingerprint(train_root),
    }


def _resolve_smoke_output(workspace: Path, output_path: Path | None) -> Path:
    if output_path is None:
        return workspace / "reports" / "v007_runtime_smoke.json"
    output_path = Path(output_path)
    return output_path if output_path.is_absolute() else workspace / output_path


def _atomic_write_json(path: Path, payload: dict) -> None:
    partial_path = Path(f"{path}.partial")
    for candidate in (path, partial_path):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite {candidate}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    verified = json.loads(partial_path.read_text(encoding="utf-8"))
    if verified != payload:
        raise ValueError("Smoke JSON verification changed the payload")
    os.replace(partial_path, path)


def run_smoke(
    workspace: Path,
    *,
    wells: int = SMOKE_WELLS,
    output_path: Path | None = None,
    parent_particles: int = 500,
    parent_seeds: int = 32,
) -> dict:
    """Run the actual compiled inference path without reading the suffix target."""
    if wells != SMOKE_WELLS:
        raise ValueError(f"v007 smoke requires exactly {SMOKE_WELLS} requested quantile wells")
    if not NUMBA_AVAILABLE:
        raise RuntimeError("The v007 runtime smoke requires numba")
    workspace = Path(workspace).resolve()
    output_path = _resolve_smoke_output(workspace, output_path)
    partial_output_path = Path(f"{output_path}.partial")
    for candidate in (output_path, partial_output_path):
        if candidate.exists():
            raise FileExistsError(f"Refusing to overwrite {candidate}")
    source_path = Path(__file__).resolve()
    parent_source_path = _parent_source_path()
    source_sha256_start = sha256_file(source_path)
    parent_source_sha256_start = sha256_file(parent_source_path)
    train_root = workspace / "data" / "raw" / "train"
    inventory = _smoke_inventory(train_root)
    paths = inventory["paths"]
    selected = inventory["selected"]

    started = time.perf_counter()
    records = []
    failures = []
    for path in selected:
        well_started = time.perf_counter()
        well = well_id_from_path(path)
        horizontal = pd.read_csv(path, usecols=["MD", "Z", "GR", "TVT_input"])
        typewell = pd.read_csv(
            path.with_name(f"{well}{TYPEWELL_SUFFIX}"),
            usecols=lambda column: column in {"TVT", "GR"},
        )
        try:
            candidates, diagnostics = hmm_candidates(
                horizontal,
                typewell,
                configs=HMM_CANDIDATES,
                parent_particles=parent_particles,
                parent_seeds=parent_seeds,
            )
            records.append(
                {
                    "well": well,
                    "suffix_rows": len(candidates),
                    "runtime_seconds": round(time.perf_counter() - well_started, 3),
                    "expanded_corridor": diagnostics["expanded_corridor"],
                    "final_half_width": diagnostics["final_half_width"],
                    "any_candidate_unresolved_position_boundary": diagnostics[
                        "any_candidate_unresolved_position_boundary"
                    ],
                    "any_candidate_unresolved_rate_boundary": diagnostics[
                        "any_candidate_unresolved_rate_boundary"
                    ],
                }
            )
        except Exception as exc:
            failures.append({"well": well, "error": f"{type(exc).__name__}: {exc}"})
    runtime_seconds = time.perf_counter() - started
    projected_hours = runtime_seconds / len(selected) * len(paths) / 3600.0
    peak_rss_gib = _peak_rss_gib()
    exact_parent = parent_particles == 500 and parent_seeds == 32
    eligible = bool(
        not failures
        and exact_parent
        and projected_hours <= MAX_PROJECTED_HOURS
        and peak_rss_gib <= MAX_PEAK_RSS_GIB
    )
    final_inventory = _smoke_inventory(train_root)
    immutable_inventory_keys = (
        "selection_positions",
        "total_suffix_rows",
        "data_fingerprint",
    )
    if any(final_inventory[key] != inventory[key] for key in immutable_inventory_keys) or [
        path.name for path in final_inventory["selected"]
    ] != [path.name for path in selected]:
        raise RuntimeError("Raw training data changed during the runtime smoke")
    source_sha256_end = sha256_file(source_path)
    parent_source_sha256_end = sha256_file(parent_source_path)
    if (
        source_sha256_end != source_sha256_start
        or parent_source_sha256_end != parent_source_sha256_start
    ):
        raise RuntimeError("Inference source changed during the runtime smoke")
    result = {
        "version": "v007",
        "intended_use": "target-free runtime and memory gate only; never select candidates",
        "target_read": False,
        "source_sha256": source_sha256_start,
        "source_sha256_end": source_sha256_end,
        "parent_source_sha256": parent_source_sha256_start,
        "parent_source_sha256_end": parent_source_sha256_end,
        "parent_version": PARENT_VERSION,
        "parent_candidate": PARENT_CANDIDATE,
        "parent_particles": parent_particles,
        "parent_seeds": parent_seeds,
        "candidate_grid": [asdict(config) for config in HMM_CANDIDATES],
        "fixed_settings": _fixed_settings(),
        "data_fingerprint": inventory["data_fingerprint"],
        "data_fingerprint_algorithm": DATA_FINGERPRINT_ALGORITHM,
        "total_suffix_rows": inventory["total_suffix_rows"],
        "required_smoke_wells": SMOKE_WELLS,
        "smoke_well_count": len(selected),
        "selection_method": "suffix_row_count_quantiles",
        "selection_positions": inventory["selection_positions"],
        "selected_wells": [well_id_from_path(path) for path in selected],
        "total_training_wells": len(paths),
        "runtime_seconds": round(runtime_seconds, 3),
        "projected_two_candidate_hours": projected_hours,
        "peak_rss_gib": peak_rss_gib,
        "max_projected_hours": MAX_PROJECTED_HOURS,
        "max_peak_rss_gib": MAX_PEAK_RSS_GIB,
        "eligible_for_full_cv": eligible,
        "records": records,
        "failures": failures,
    }
    _atomic_write_json(output_path, result)
    return result


def _require_runtime_smoke(workspace: Path) -> tuple[Path, dict]:
    path = workspace / "reports" / "v007_runtime_smoke.json"
    if not path.exists():
        raise FileNotFoundError("Run the target-free v007 runtime smoke before full CV")
    result = json.loads(path.read_text(encoding="utf-8"))
    partial_path = Path(f"{path}.partial")
    if partial_path.exists():
        raise ValueError("Runtime smoke has an unexpected partial companion")
    inventory = _smoke_inventory(workspace / "data" / "raw" / "train")
    expected_source = sha256_file(Path(__file__).resolve())
    expected_parent_source = sha256_file(_parent_source_path())
    if (
        result.get("source_sha256") != expected_source
        or result.get("source_sha256_end") != expected_source
        or result.get("parent_source_sha256") != expected_parent_source
        or result.get("parent_source_sha256_end") != expected_parent_source
    ):
        raise ValueError("Runtime smoke source hash does not match the current v007 implementation")
    if result.get("target_read") is not False:
        raise ValueError("Runtime smoke did not preserve the target-free contract")
    if result.get("parent_particles") != 500 or result.get("parent_seeds") != 32:
        raise ValueError("Runtime smoke did not use the locked v006 parent settings")
    if result.get("candidate_grid") != [asdict(config) for config in HMM_CANDIDATES]:
        raise ValueError("Runtime smoke candidate grid differs from the predeclared grid")
    if result.get("fixed_settings") != _fixed_settings():
        raise ValueError("Runtime smoke fixed settings differ from the current implementation")
    if result.get("data_fingerprint") != inventory["data_fingerprint"]:
        raise ValueError("Runtime smoke raw data fingerprint differs from current training data")
    if result.get("data_fingerprint_algorithm") != DATA_FINGERPRINT_ALGORITHM:
        raise ValueError("Runtime smoke data fingerprint algorithm is unexpected")
    if result.get("total_suffix_rows") != inventory["total_suffix_rows"]:
        raise ValueError("Runtime smoke suffix-row total differs from current training data")
    expected_wells = [well_id_from_path(path) for path in inventory["selected"]]
    if (
        result.get("required_smoke_wells") != SMOKE_WELLS
        or result.get("smoke_well_count") != SMOKE_WELLS
        or result.get("selection_method") != "suffix_row_count_quantiles"
        or result.get("selection_positions") != inventory["selection_positions"]
        or result.get("selected_wells") != expected_wells
    ):
        raise ValueError("Runtime smoke did not use the required six quantile wells")
    if result.get("total_training_wells") != len(inventory["paths"]):
        raise ValueError("Runtime smoke training-well total differs from current data")
    if not result.get("eligible_for_full_cv"):
        raise RuntimeError("Runtime smoke did not pass the full-CV resource gate")
    if (
        float(result["projected_two_candidate_hours"]) > MAX_PROJECTED_HOURS
        or float(result["peak_rss_gib"]) > MAX_PEAK_RSS_GIB
    ):
        raise RuntimeError("Runtime smoke exceeds the predeclared resource limits")
    return path, result


def _boundary_wells(
    summary: pd.DataFrame, selections: list[dict], final_candidate: str
) -> tuple[set[str], set[str]]:
    selected_wells: set[str] = set()
    for selection in selections:
        mask = (summary["fold"] == int(selection["fold"])) & (
            summary["candidate"] == str(selection["selected_candidate"])
        )
        rows = summary.loc[mask]
        boundary = rows["unresolved_position_boundary"] | rows["unresolved_rate_boundary"]
        selected_wells.update(rows.loc[boundary, "well"].astype(str))
    final_rows = summary.loc[summary["candidate"] == final_candidate]
    final_boundary = (
        final_rows["unresolved_position_boundary"] | final_rows["unresolved_rate_boundary"]
    )
    return selected_wells, set(final_rows.loc[final_boundary, "well"].astype(str))


def predict_visible_test(
    workspace: Path, final_candidate: str
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    configs = {config.name: config for config in HMM_CANDIDATES}
    if final_candidate not in configs:
        raise ValueError(f"Unknown final HMM candidate: {final_candidate}")
    frames = []
    failures = []
    diagnostics = []
    test_root = workspace / "data" / "raw" / "test"
    for path in sorted(test_root.glob(f"*{HORIZONTAL_SUFFIX}")):
        well = well_id_from_path(path)
        horizontal = pd.read_csv(path, usecols=["MD", "Z", "GR", "TVT_input"])
        evaluation = pd.to_numeric(horizontal["TVT_input"], errors="coerce").isna().to_numpy()
        evaluation_indices = np.flatnonzero(evaluation)
        try:
            candidates, well_diagnostics = hmm_candidates(
                horizontal,
                pd.read_csv(
                    path.with_name(f"{well}{TYPEWELL_SUFFIX}"),
                    usecols=lambda column: column in {"TVT", "GR"},
                ),
                configs=HMM_CANDIDATES,
            )
            candidate_diagnostics = well_diagnostics["candidates"][final_candidate]
            if (
                candidate_diagnostics["unresolved_position_boundary"]
                or candidate_diagnostics["unresolved_rate_boundary"]
            ):
                raise RuntimeError("unresolved posterior boundary")
            values = candidates[final_candidate].to_numpy(dtype=float)
            frames.append(
                pd.DataFrame(
                    {
                        "_id": [f"{well}_{row}" for row in evaluation_indices],
                        "prediction": values,
                    }
                )
            )
            diagnostics.append({"well": well, **well_diagnostics})
        except Exception as exc:
            failures.append({"well": well, "error": f"{type(exc).__name__}: {exc}"})
    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["_id", "prediction"])
    )
    return result, failures, diagnostics


def _align_test_predictions(workspace: Path, predictions: pd.DataFrame) -> np.ndarray:
    sample = pd.read_csv(workspace / "data" / "raw" / "sample_submission.csv")
    if list(sample.columns) != ["id", "tvt"] or sample["id"].duplicated().any():
        raise ValueError("Unexpected sample submission contract")
    if predictions["_id"].duplicated().any():
        raise ValueError("Visible-test prediction IDs are duplicated")
    values = sample["id"].map(predictions.set_index("_id")["prediction"])
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Visible-test predictions do not cover the sample IDs")
    return values.to_numpy(dtype=float)


def _write_summary(
    path: Path,
    *,
    overall: float,
    fold_scores: list[float],
    final_candidate: str,
    decision: str,
    folds_better: int,
    failures: int,
    selected_boundaries: int,
    final_boundaries: int,
) -> None:
    lines = [
        "# ROGII v007 Window-Registration HMM Smoother",
        "",
        f"Status: `{decision}`.",
        "",
        "## Grouped CV",
        "",
        f"- Nested pooled OOF RMSE: `{overall:.9f}`",
        f"- Final all-training candidate: `{final_candidate}`",
        f"- Canonical folds better than v006: `{folds_better} / 5`",
        f"- Fold RMSE: `{', '.join(f'{value:.6f}' for value in fold_scores)}`",
        f"- Training fallback wells: `{failures}`",
        f"- Selected/final unresolved boundary wells: `{selected_boundaries} / {final_boundaries}`",
        "",
        "The candidate grid and gates were fixed before target-scored execution in",
        "`reports/v007_kaggle_skill_audit.md`. This script never submits to Kaggle.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def train(workspace: Path, *, version: str = "v007", folds: int = 5) -> Path:
    """Run locked nested CV and persist complete immutable v007 artifacts."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Full v007 HMM training requires numba")
    workspace = Path(workspace).resolve()
    model_dir = workspace / "models" / version
    if model_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {model_dir}")
    model_dir.mkdir(parents=True)
    run_path = model_dir / "run.json"
    started = time.time()
    run = {
        "version": version,
        "status": "running",
        "parent_run": PARENT_VERSION,
        "template": "rogii-latent-surface-window-registration-hmm-smoother",
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
        smoke_path, smoke = _require_runtime_smoke(workspace)
        manifest_path, manifest = ensure_fold_manifest(workspace, folds)
        feature_path = next_feature_path(workspace)
        project_root = workspace.parents[1]
        source_path = Path(__file__).resolve()
        parent_source_path = _parent_source_path()
        source_sha256_start = sha256_file(source_path)
        parent_source_sha256_start = sha256_file(parent_source_path)
        training_data_fingerprint = str(smoke["data_fingerprint"])
        run.update(
            {
                "command": shlex.join([sys.executable, *sys.argv]),
                "reproduction_command": (
                    "uv run --with-requirements "
                    "workspaces/rogii-wellbore-geology-prediction/requirements-beam.txt python "
                    "workspaces/rogii-wellbore-geology-prediction/scripts/"
                    f"rogii_hmm_smoother.py --workspace {workspace} train --version {version} "
                    f"--folds {folds}"
                ),
                "git_commit": git_commit(project_root),
                "git_dirty_paths": git_dirty_paths(project_root),
                "source_sha256": source_sha256_start,
                "parent_source": str(parent_source_path.relative_to(project_root)),
                "parent_source_sha256": parent_source_sha256_start,
                "external_method_reference": REFERENCE_URL,
                "external_method_sha256": REFERENCE_SHA256,
                "params": {
                    "candidates": [asdict(config) for config in HMM_CANDIDATES],
                    "fixed_settings": _fixed_settings(),
                },
                "random_seed": None,
                "cv_splitter": "canonical row-balanced GroupKFold by well",
                "fold_manifest": str(manifest_path.relative_to(workspace)),
                "fold_manifest_sha256": sha256_file(manifest_path),
                "metric": "rmse",
                "metric_direction": "minimize",
                "target_column": "TVT",
                "id_column": "id",
                "feature_path": str(feature_path.relative_to(workspace)),
                "data_fingerprint": training_data_fingerprint,
                "data_fingerprint_algorithm": DATA_FINGERPRINT_ALGORITHM,
                "runtime_smoke": str(smoke_path.relative_to(workspace)),
                "runtime_smoke_sha256": sha256_file(smoke_path),
                "runtime_projection_hours": smoke["projected_two_candidate_hours"],
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                },
            }
        )
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

        summary, diagnostics, failures = build_training_candidates(
            workspace, manifest, feature_path
        )
        report_paths = {
            "by_well": workspace / "reports" / f"{version}_hmm_candidates_by_well.csv",
            "scan": workspace / "reports" / f"{version}_hmm_candidate_scan.csv",
            "diagnostics": workspace / "reports" / f"{version}_hmm_diagnostics.json",
            "summary": workspace / "reports" / f"{version}_hmm_smoother_summary.md",
        }
        for path in report_paths.values():
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
        summary.to_csv(report_paths["by_well"], index=False)
        scan = summary.groupby("candidate", as_index=False).agg(
            squared_error=("squared_error", "sum"), rows=("rows", "sum")
        )
        scan["pooled_rmse"] = np.sqrt(scan["squared_error"] / scan["rows"])
        scan.sort_values(["pooled_rmse", "candidate"]).to_csv(report_paths["scan"], index=False)
        report_paths["diagnostics"].write_text(
            json.dumps({"wells": diagnostics, "failures": failures}, indent=2),
            encoding="utf-8",
        )

        selections, final_candidate = choose_nested_candidates(summary)
        oof_rows, oof_prediction, oof_std = materialize_oof(feature_path, selections)
        target = oof_rows["_target"].to_numpy(dtype=float)
        fold_values = oof_rows["fold"].to_numpy(dtype=int)
        fold_scores = [
            rmse(target[fold_values == fold], oof_prediction[fold_values == fold])
            for fold in sorted(manifest["fold"].unique())
        ]
        overall = rmse(target, oof_prediction)
        v006 = baseline_scores_on_manifest(workspace, manifest)
        folds_better = int(
            sum(current < baseline for current, baseline in zip(fold_scores, v006["fold_scores"]))
        )
        selected_boundary_wells, final_boundary_wells = _boundary_wells(
            summary, selections, final_candidate
        )
        decision = candidate_decision(
            overall,
            folds_better,
            training_failures=len(failures),
            selected_boundary_wells=len(selected_boundary_wells),
            final_boundary_wells=len(final_boundary_wells),
        )

        test_failures = []
        test_diagnostics = []
        test_prediction = None
        test_data_fingerprint = None
        if decision == "candidate_ready":
            test_root = workspace / "data" / "raw" / "test"
            test_data_fingerprint = data_fingerprint(test_root)
            test_frame, test_failures, test_diagnostics = predict_visible_test(
                workspace, final_candidate
            )
            if data_fingerprint(test_root) != test_data_fingerprint:
                raise RuntimeError("Raw visible-test data changed during v007 inference")
            decision = candidate_decision(
                overall,
                folds_better,
                training_failures=len(failures),
                selected_boundary_wells=len(selected_boundary_wells),
                final_boundary_wells=len(final_boundary_wells),
                test_failures=len(test_failures),
            )
            if decision == "candidate_ready":
                test_prediction = _align_test_predictions(workspace, test_frame)

        if data_fingerprint(workspace / "data" / "raw" / "train") != training_data_fingerprint:
            raise RuntimeError("Raw training data changed during the v007 run")
        if (
            sha256_file(source_path) != source_sha256_start
            or sha256_file(parent_source_path) != parent_source_sha256_start
        ):
            raise RuntimeError("Inference source changed during the v007 run")

        np.save(model_dir / "oof_preds.npy", oof_prediction)
        np.save(model_dir / "oof_std.npy", oof_std)
        oof_rows.to_parquet(model_dir / "oof_rows.parquet", index=False)
        if test_prediction is not None:
            np.save(model_dir / "test_preds.npy", test_prediction)
        (model_dir / "feature_list.txt").write_text(
            "\n".join(INPUT_FEATURES) + "\n", encoding="utf-8"
        )
        pd.DataFrame(columns=["feature", "importance"]).to_csv(
            model_dir / "importance.csv", index=False
        )
        model = {
            "model": "latent_surface_window_registration_hmm_smoother",
            "parent_version": PARENT_VERSION,
            "parent_candidate": PARENT_CANDIDATE,
            "final_candidate": final_candidate,
            "fold_selections": selections,
            "configs": [asdict(config) for config in HMM_CANDIDATES],
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
            "selection_protocol": "outer-fold training wells select two fixed pure HMM candidates",
            "fold_selections": selections,
            "final_candidate": final_candidate,
            "fallback_wells": len(failures),
            "selected_boundary_wells": sorted(selected_boundary_wells),
            "final_boundary_wells": sorted(final_boundary_wells),
            "visible_test_failures": test_failures,
            "canonical_v006": v006,
            "folds_better_than_v006": folds_better,
            "candidate_gate": CANDIDATE_GATE,
            "promising_gate": PROMISING_GATE,
            "decision": decision,
        }
        (model_dir / "cv_scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
        if test_diagnostics:
            (model_dir / "test_diagnostics.json").write_text(
                json.dumps(test_diagnostics, indent=2), encoding="utf-8"
            )
        _write_summary(
            report_paths["summary"],
            overall=overall,
            fold_scores=fold_scores,
            final_candidate=final_candidate,
            decision=decision,
            folds_better=folds_better,
            failures=len(failures),
            selected_boundaries=len(selected_boundary_wells),
            final_boundaries=len(final_boundary_wells),
        )
        run.update(
            {
                "status": "completed",
                "runtime_seconds": round(time.time() - started, 3),
                "decision": decision,
                "cv_score": overall,
                "folds_better_than_v006": folds_better,
                "final_candidate": final_candidate,
                "fallback_wells": len(failures),
                "selected_boundary_wells": len(selected_boundary_wells),
                "final_boundary_wells": len(final_boundary_wells),
                "test_fallback_wells": len(test_failures),
                "test_predictions_saved": test_prediction is not None,
                "test_data_fingerprint": test_data_fingerprint,
                "submission_path": None,
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
                "submitted": False,
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
    smoke.add_argument("--wells", type=int, choices=(SMOKE_WELLS,), default=SMOKE_WELLS)
    smoke.add_argument("--output", type=Path)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--version", default="v007")
    train_parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.resolve()
    try:
        if args.command == "smoke":
            print(
                json.dumps(
                    run_smoke(workspace, wells=args.wells, output_path=args.output), indent=2
                )
            )
        else:
            print(train(workspace, version=args.version, folds=args.folds))
    except KeyboardInterrupt as exc:
        print(f"KeyboardInterrupt: {exc}", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
