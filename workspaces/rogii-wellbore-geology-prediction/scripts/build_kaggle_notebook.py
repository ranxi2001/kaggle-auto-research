#!/usr/bin/env python3
"""Generate self-contained ROGII Kaggle code notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def build(workspace: Path, username: str) -> tuple[Path, Path]:
    baseline_path = workspace / "scripts" / "rogii_baseline.py"
    source = baseline_path.read_text(encoding="utf-8")
    marker = '\nif __name__ == "__main__":'
    if marker not in source:
        raise ValueError("Could not locate rogii_baseline.py main guard")
    library_source = source.split(marker, maxsplit=1)[0]
    # Numba cannot create a disk cache for functions defined in a Jupyter code cell.
    library_source = library_source.replace("@njit(cache=True)", "@njit(cache=False)")

    driver = """
from pathlib import Path
import os


def find_competition_data():
    for sample_path in Path("/kaggle/input").rglob("sample_submission.csv"):
        candidate = sample_path.parent
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError("ROGII competition data was not mounted")


data_root = find_competition_data()
workspace = Path("/kaggle/temp/kar_rogii")
raw_root = workspace / "data" / "raw"
raw_root.mkdir(parents=True, exist_ok=True)
for name in ("train", "test"):
    destination = raw_root / name
    if not destination.exists():
        destination.symlink_to(data_root / name, target_is_directory=True)
sample_destination = raw_root / "sample_submission.csv"
if not sample_destination.exists():
    sample_destination.symlink_to(data_root / "sample_submission.csv")

print(f"competition data: {data_root}")
print(f"train wells: {len(list((raw_root / 'train').glob('*__horizontal_well.csv')))}")
print(f"hidden test wells: {len(list((raw_root / 'test').glob('*__horizontal_well.csv')))}")

residual_model_dir = train_residual(
    workspace,
    n_splits=5,
    n_estimators=600,
    learning_rate=0.03,
)
predict_residual(workspace, residual_model_dir)
residual_prediction = np.load(residual_model_dir / "test_preds.npy")

sample = pd.read_csv(raw_root / "sample_submission.csv")
if list(sample.columns) != ["id", "tvt"] or sample["id"].duplicated().any():
    raise RuntimeError("Unexpected sample submission contract")
beam_by_id = sample_fallback_predictions(raw_root / "test", sample)
beam_failures = 0
test_paths = sorted((raw_root / "test").glob(f"*{HORIZONTAL_SUFFIX}"))
for horizontal_path in test_paths:
    try:
        rows, failed = _beam_rows_for_well(
            horizontal_path,
            "very_loose",
            0.7,
            require_target=False,
        )
        well_predictions = pd.Series(
            rows["prediction"].to_numpy(float), index=rows["_id"].astype(str)
        )
        shared_ids = beam_by_id.index.intersection(well_predictions.index)
        beam_by_id.loc[shared_ids] = well_predictions.loc[shared_ids]
        beam_failures += int(failed)
    except Exception as exc:
        beam_failures += 1
        print(f"beam fallback for {horizontal_path.name}: {type(exc).__name__}: {exc}")
beam_prediction = beam_by_id.reindex(sample["id"].astype(str)).to_numpy(float)
if len(beam_prediction) != len(residual_prediction):
    raise RuntimeError("Beam and residual prediction lengths differ")
final_prediction = 0.175 * beam_prediction + 0.825 * residual_prediction
if not np.isfinite(final_prediction).all():
    raise RuntimeError("Final predictions contain NaN or infinity")

submission = sample[["id"]].copy()
submission["tvt"] = final_prediction
output_path = Path("/kaggle/working/submission.csv")
submission.to_csv(output_path, index=False)
reloaded = pd.read_csv(output_path)
if not reloaded["id"].equals(sample["id"]):
    raise RuntimeError("Final submission ID order changed")
if list(reloaded.columns) != ["id", "tvt"] or len(reloaded) != len(sample):
    raise RuntimeError("Final submission shape or columns are invalid")

print(json.dumps({
    "output": str(output_path),
    "rows": len(submission),
    "test_wells": len(test_paths),
    "beam_fallback_wells": beam_failures,
    "residual_cv": json.loads((residual_model_dir / "cv_scores.json").read_text()),
}, indent=2))
""".strip()

    notebook = {
        "cells": [
            markdown_cell(
                "# ROGII v004: Grouped Residual + GR Beam Ensemble\n\n"
                "This notebook trains only on the mounted competition data. It uses a "
                "well-grouped residual model and target-independent GR/typewell sequence "
                "alignment, then writes one sample-aligned `submission.csv`. Public-score "
                "artifacts and train/test well-ID target lookups are intentionally excluded."
            ),
            code_cell(library_source),
            code_cell(driver),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_dir = workspace / "notebooks" / "v004"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = notebook_dir / "rogii-v004.ipynb"
    metadata_path = notebook_dir / "kernel-metadata.json"
    for path in (notebook_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    metadata = {
        "id": f"{username}/kar-rogii-v004-grouped-residual-beam-ensemble",
        "title": "KAR ROGII v004 Grouped Residual Beam Ensemble",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return notebook_path, metadata_path


def build_state_space(
    workspace: Path,
    username: str,
    *,
    version: str = "v006",
    candidate: str = "pf_scale_12_hold_0p2",
    replace_existing: bool = False,
) -> tuple[Path, Path]:
    state_space_path = workspace / "scripts" / "rogii_state_space.py"
    source = state_space_path.read_text(encoding="utf-8")
    marker = '\nif __name__ == "__main__":'
    if marker not in source:
        raise ValueError("Could not locate rogii_state_space.py main guard")
    library_source = source.split(marker, maxsplit=1)[0]
    library_source = library_source.replace(
        "@njit(cache=True, nogil=True)", "@njit(cache=False, nogil=True)"
    )

    driver = f'''
import os


def find_competition_data(input_root):
    for sample_path in input_root.rglob("sample_submission.csv"):
        candidate_root = sample_path.parent
        if (candidate_root / "train").is_dir() and (candidate_root / "test").is_dir():
            return candidate_root
    raise FileNotFoundError("ROGII competition data was not mounted")


if not NUMBA_AVAILABLE:
    raise RuntimeError("The state-space notebook requires Numba")

started = time.time()
input_root = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
working_root = Path(os.environ.get("KAGGLE_WORKING_ROOT", "/kaggle/working"))
runtime_workspace = Path(os.environ.get("KAR_RUNTIME_WORKSPACE", "/kaggle/temp/kar_rogii_v006"))
data_root = find_competition_data(input_root)
raw_root = runtime_workspace / "data" / "raw"
raw_root.mkdir(parents=True, exist_ok=True)
for name in ("train", "test"):
    destination = raw_root / name
    if not destination.exists():
        destination.symlink_to(data_root / name, target_is_directory=True)
sample_destination = raw_root / "sample_submission.csv"
if not sample_destination.exists():
    sample_destination.symlink_to(data_root / "sample_submission.csv")

test_paths = sorted((raw_root / "test").glob(f"*{{HORIZONTAL_SUFFIX}}"))
print(f"competition data: {{data_root}}")
print(f"hidden test wells: {{len(test_paths)}}")

test_frame, failures = predict_visible_test(
    runtime_workspace,
    "{candidate}",
    particles=500,
    seeds=32,
    scales=(3.0, 5.0, 8.0, 12.0),
    holds=(0.0, 0.2, 0.5),
)
if failures:
    raise RuntimeError(f"Particle inference failed for test wells: {{failures}}")

sample = pd.read_csv(raw_root / "sample_submission.csv")
if list(sample.columns) != ["id", "tvt"] or sample["id"].duplicated().any():
    raise RuntimeError("Unexpected sample submission contract")
prediction_map = test_frame.set_index("_id")["prediction"]
submission = sample[["id"]].copy()
submission["tvt"] = submission["id"].map(prediction_map)
if submission["tvt"].isna().any() or not np.isfinite(submission["tvt"]).all():
    raise RuntimeError("Final predictions contain missing values or infinity")

working_root.mkdir(parents=True, exist_ok=True)
output_path = working_root / "submission.csv"
submission.to_csv(output_path, index=False)
reloaded = pd.read_csv(output_path)
if not reloaded["id"].equals(sample["id"]):
    raise RuntimeError("Final submission ID order changed")
if list(reloaded.columns) != ["id", "tvt"] or len(reloaded) != len(sample):
    raise RuntimeError("Final submission shape or columns are invalid")

summary = {{
    "version": "{version}",
    "candidate": "{candidate}",
    "particles": 500,
    "seeds": 32,
    "output": str(output_path),
    "rows": len(submission),
    "test_wells": len(test_paths),
    "fallback_wells": 0,
    "runtime_seconds": round(time.time() - started, 3),
}}
(working_root / "run_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
'''.strip()

    notebook = {
        "cells": [
            markdown_cell(
                "# ROGII v006: Latent-Surface Particle Filter\n\n"
                "This notebook performs deterministic target-free inference from each runtime "
                "horizontal well, its visible `TVT_input` prefix, trajectory, GR log, and paired "
                "typewell. The fixed candidate was selected by five-fold nested grouped CV. "
                "Missing GR rows skip measurement updates, and any well-level inference failure "
                "stops the notebook before `submission.csv` is written."
            ),
            code_cell(library_source),
            code_cell(driver),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_dir = workspace / "notebooks" / version
    notebook_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = notebook_dir / f"rogii-{version}.ipynb"
    metadata_path = notebook_dir / "kernel-metadata.json"
    for path in (notebook_path, metadata_path):
        if path.exists() and not replace_existing:
            raise FileExistsError(f"Refusing to overwrite {path}")
    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    metadata = {
        "id": f"{username}/kar-rogii-{version}-latent-surface-particle-filter",
        "title": f"KAR ROGII {version} Latent Surface Particle Filter",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return notebook_path, metadata_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--recipe", choices=("v004", "v006"), default="v004")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    if args.recipe == "v006":
        notebook_path, metadata_path = build_state_space(args.workspace.resolve(), args.username)
    else:
        notebook_path, metadata_path = build(args.workspace.resolve(), args.username)
    print(notebook_path)
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
