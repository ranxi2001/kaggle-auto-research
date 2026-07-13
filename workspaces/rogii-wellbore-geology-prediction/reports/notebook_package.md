# ROGII v004 Notebook Package

Generated on 2026-07-13 for private Kaggle kernel
`ranxi169/kar-rogii-v004-grouped-residual-beam-ensemble`.

## Package Contract

- Notebook: `notebooks/v004/rogii-v004.ipynb`
- Kernel metadata: `notebooks/v004/kernel-metadata.json`
- Competition source: `rogii-wellbore-geology-prediction`
- Internet: disabled
- Accelerator: CPU
- Output: `/kaggle/working/submission.csv`
- Submission state: kernel version 1 and submission ref 54653094 complete

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

## Remote Run

Kaggle kernel version 1 completed in about 11.4 minutes with 773 training wells, five grouped folds,
and zero beam fallback wells. Its pooled CV RMSE was 14.903702. The public-output file contained
14,151 sample-aligned finite predictions and passed the local dry-run validator.

The code submission completed as ref 54653094 with public LB RMSE 14.683 and rank 3628 of 4829.

## Checksums

| Artifact | SHA-256 |
|---|---|
| `rogii-v004.ipynb` | `b86b55441eb7bea20acd73c2ca9b304ba8563c27341e9135190fe9a6adef2f60` |
| `kernel-metadata.json` | `4808f0e00ea6b1a28bdeee900259920435584752feab2d74bfb37991d9b24608` |
| Local visible-test v004 CSV | `2a76e66fecc151f95e37b85930a447a0489ec1462cbe7d6925a7a2a7d4302dc4` |
| Kaggle kernel v1 output CSV | `a8f4570fbb1d56f1da81d2e2e0f096c520711151a347674581bf229563b6ab6c` |

The local visible-test CSV is only a format reference. Kaggle generated the kernel v1 output and
used that version as provenance for the completed code submission.
