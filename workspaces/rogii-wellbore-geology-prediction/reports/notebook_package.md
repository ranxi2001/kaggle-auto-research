# ROGII v004 Notebook Package

Generated on 2026-07-13 for private Kaggle kernel `ranxi169/kar-rogii-v004`.

## Package Contract

- Notebook: `notebooks/v004/rogii-v004.ipynb`
- Kernel metadata: `notebooks/v004/kernel-metadata.json`
- Competition source: `rogii-wellbore-geology-prediction`
- Internet: disabled
- Accelerator: CPU
- Output: `/kaggle/working/submission.csv`
- Submission state: not pushed, not submitted

The notebook dynamically discovers the competition data mount, builds all temporary artifacts under
`/kaggle/temp/kar_rogii`, trains five grouped residual models, computes GR beam predictions for every
runtime test well, blends them at 17.5%/82.5%, and maps by ID to the runtime sample order.

## Verification

- Both code cells compile.
- The embedded library executes with an IPython-style filename and a live Numba dispatcher.
- A fresh temporary workspace completed residual training and sample-aligned prediction without any
  pre-existing `models/`, `features/`, or `submissions/` directory.
- Contract tests cover dynamic globbing, internet-off metadata, output path, exclusion of public
  artifacts and hardcoded well IDs, fresh submission-directory creation, and pooled-score metadata.
- The official Kaggle Docker image repository includes import/training tests for both Numba and
  LightGBM; no network installation is required by the notebook.

## Checksums

| Artifact | SHA-256 |
|---|---|
| `rogii-v004.ipynb` | `b86b55441eb7bea20acd73c2ca9b304ba8563c27341e9135190fe9a6adef2f60` |
| `kernel-metadata.json` | `834f6b0ef6ef4636485b97ea88fa1666297ef1c6d0e50c819f0b0536c5834b89` |
| Local visible-test v004 CSV | `2a76e66fecc151f95e37b85930a447a0489ec1462cbe7d6925a7a2a7d4302dc4` |

The visible-test CSV checksum is only a local format reference. The hidden rerun generates a new
`submission.csv` inside Kaggle and must be submitted only after explicit user approval.
