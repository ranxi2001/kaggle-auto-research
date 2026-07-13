#!/usr/bin/env python3
"""Compare versioned OOF candidates using pooled RMSE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(target - prediction))))


def load_oof(model_dir: Path, label: str) -> pd.DataFrame:
    rows = pd.read_parquet(model_dir / "oof_rows.parquet")
    predictions = np.load(model_dir / "oof_preds.npy")
    if len(rows) != len(predictions):
        raise ValueError(f"{label} OOF row/prediction length mismatch")
    if rows["_id"].duplicated().any():
        raise ValueError(f"{label} OOF IDs are not unique")
    return rows[["_id", "_well_id", "_target"]].assign(**{label: predictions})


def analyze(workspace: Path, left_version: str, right_version: str) -> dict:
    reports = workspace / "reports"
    left = load_oof(workspace / "models" / left_version, left_version)
    right = load_oof(workspace / "models" / right_version, right_version)
    merged = left.merge(
        right[["_id", "_target", right_version]],
        on="_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_right"),
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("OOF candidate ID sets do not match")
    if not np.allclose(merged["_target"], merged["_target_right"]):
        raise ValueError("OOF candidate targets do not match")

    target = merged["_target"].to_numpy(dtype=float)
    left_prediction = merged[left_version].to_numpy(dtype=float)
    right_prediction = merged[right_version].to_numpy(dtype=float)
    blend_records = []
    for right_weight in np.linspace(0.0, 1.0, 41):
        prediction = (1.0 - right_weight) * left_prediction + right_weight * right_prediction
        blend_records.append(
            {"right_weight": right_weight, "pooled_rmse": rmse(target, prediction)}
        )
    blend_scan = pd.DataFrame(blend_records).sort_values("pooled_rmse")

    error_frame = pd.DataFrame(
        {
            "well": merged["_well_id"],
            "left_squared_error": np.square(left_prediction - target),
            "right_squared_error": np.square(right_prediction - target),
        }
    )
    by_well = (
        error_frame.groupby("well", as_index=False)
        .agg(
            rows=("left_squared_error", "size"),
            left_squared_error=("left_squared_error", "sum"),
            right_squared_error=("right_squared_error", "sum"),
        )
    )
    by_well["left_rmse"] = np.sqrt(by_well["left_squared_error"] / by_well["rows"])
    by_well["right_rmse"] = np.sqrt(by_well["right_squared_error"] / by_well["rows"])
    by_well["rmse_delta"] = by_well["right_rmse"] - by_well["left_rmse"]
    by_well = by_well.sort_values("rmse_delta")

    stem = f"{left_version}_{right_version}"
    blend_path = reports / f"blend_scan_{stem}.csv"
    by_well_path = reports / f"candidate_compare_{stem}.csv"
    summary_path = reports / f"candidate_summary_{stem}.json"
    for path in (blend_path, by_well_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    blend_scan.to_csv(blend_path, index=False)
    by_well.to_csv(by_well_path, index=False)

    best = blend_scan.iloc[0]
    summary = {
        "metric": "rmse",
        "direction": "minimize",
        "rows": len(merged),
        "wells": int(merged["_well_id"].nunique()),
        "left_version": left_version,
        "left_rmse": rmse(target, left_prediction),
        "right_version": right_version,
        "right_rmse": rmse(target, right_prediction),
        "best_right_weight": float(best["right_weight"]),
        "best_blend_rmse": float(best["pooled_rmse"]),
        "right_better_wells": int((by_well["rmse_delta"] < 0).sum()),
        "right_worse_wells": int((by_well["rmse_delta"] > 0).sum()),
        "median_well_rmse_delta": float(by_well["rmse_delta"].median()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_version")
    parser.add_argument("right_version")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    summary = analyze(args.workspace.resolve(), args.left_version, args.right_version)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
