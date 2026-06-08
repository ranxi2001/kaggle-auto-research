# DRW Crypto Experiment Log

## Current Best

- Metric: Pearson
- Local OOF score: `0.1497459525`
- Submission:
  `sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv`
- Validation: dry-run valid
- Kaggle submission status: not submitted

## Strong Components

- `v021`: closed-form Ridge, `top_k=180`, `alpha=3000`, TimeSeriesSplit Pearson `0.143261`.
- `v019`: closed-form Ridge, `top_k=80`, `alpha=10000`, TimeSeriesSplit Pearson `0.133615`.
- `v017`: closed-form Ridge, `top_k=140`, non-shuffled KFold Pearson `0.125693`.
- `v010`: cleaned LightGBM, useful as a low-correlation blend member.
- Rank-normalized simplex blending is currently slightly better than raw blending.

## Rejected / Low-Value Directions

- Microstructure derived features from `bid_qty`, `ask_qty`, `buy_qty`, `sell_qty`, and `volume`
  hurt closed-form Ridge in the current implementation.
  - Version: `v027`
  - Best alpha: `8000`
  - TimeSeriesSplit Pearson: `0.005263`
  - Outcome: do not include in the best ensemble.
- Ridge stacking over model OOF predictions underperformed the direct simplex blend.
  - Best observed stacker: rank-normalized KFold positive Ridge, Pearson about `0.1284`
  - Outcome: keep optimized simplex blending.
- Continuing to sweep nearby Ridge `top_k` values produced only tiny ensemble gains.
  - Best neighborhood model added after `v021`: `v023`
  - Incremental ensemble gain: about `0.000012`

## Notes

- Public notebooks use ordinary `scipy.stats.pearsonr`; no local train/test weight column exists.
- The workspace metric has been corrected to `pearson`; `weighted_pearson` remains only as a compatibility alias.
- Current best has not been leaderboard-calibrated. A real Kaggle submission is the next high-value validation step.
