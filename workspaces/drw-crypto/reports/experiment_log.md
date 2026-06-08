# DRW Crypto Experiment Log

## Current Best

- Metric: Pearson
- Local OOF score: `0.1497459525`
- Submission:
  `sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv`
- Validation: dry-run valid; real Kaggle submit complete
- Kaggle submission status: submitted once, `COMPLETE`
- Submission audit:
  - Shape: `538150 x 2`
  - Columns: `ID,prediction`
  - Missing values: none
  - Local budget before submission: `2/2`
  - Kaggle submission history before submission: none
  - Kaggle ref: `53486449`
  - Public LB: `0.08199`
  - Private score shown by Kaggle: `0.08268`
  - Local/LB gap: local OOF `0.149746` overestimates public LB by about `0.06776`
- Public leaderboard reference at audit time:
  - Rank 1 score: `0.13959`
  - Rank 5 score: `0.11116`

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
- Current best has been leaderboard-calibrated once. Public LB is materially below local OOF, so the next high-value work is CV/LB calibration rather than blindly increasing local OOF.
- Next candidate direction: build a holdout or fold-weighting scheme that better predicts the public LB, then compare smaller Ridge-only and ensemble submissions before using the final remaining daily budget.

## Leaderboard Calibration - 2026-06-09

- Submitted file: `sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv`
- Kaggle ref: `53486449`
- Submission description: `Manual submit ... CV=0.149746`
- Status: `COMPLETE`
- Public LB: `0.08199`
- Private score currently shown by Kaggle: `0.08268`
- Local OOF Pearson: `0.1497459525`
- Interpretation: local common-mask OOF is optimistic for leaderboard ranking. The model is valid and competitive enough to be useful, but future iteration should prioritize split calibration, robustness, and LB-aligned validation over local OOF maximization.
- Budget after submission: `1/2` remaining for the day.

## LB-Calibrated Candidate - Conservative Ensemble

- Generated file: `sub_ensemble_lbstable_ranknorm_v017_v020_v019_v021_v023_v026_v010.csv`
- Validation: dry-run valid
- Models: `v017,v020,v019,v021,v023,v026,v010`
- Objective: rank-normalized ensemble optimized on later common OOF rows with an L2-style weight penalty.
- Weights:
  - `v017`: `0.407184`
  - `v023`: `0.207389`
  - `v010`: `0.199775`
  - `v020`: `0.085177`
  - `v026`: `0.070705`
  - `v019`: `0.029769`
  - `v021`: `0.000000`
- Diagnostics:
  - full common OOF: `0.143862`
  - tail 30%: `0.141487`
  - tail 20%: `0.114601`
  - tail 10%: `0.104973`
  - ts fold 4: `0.166380`
  - ts fold 5: `0.110915`
- Interpretation: this is a valid conservative second candidate, but evidence is mixed. It improves some late fold diagnostics while not improving the most recent 10% tail versus the already submitted ensemble. Do not spend the last daily submission on it unless we decide that ts_fold4/5 stability is the better LB proxy than tail_10%.

## Calibrated Candidate Sweep - 2026-06-09

- Attempted stable-feature Ridge search using fold-specific target correlations. Two full-data scripts timed out before producing artifacts; this should be reimplemented as a sampled/vectorized CLI tool before retrying.
- Generated low-cost calibrated candidates from existing OOF/test artifacts instead. Report: `reports/calibrated_candidate_summary.csv`.
- Dry-run valid candidates:
  - `sub_calibrated_v017_single.csv`
  - `sub_calibrated_stable_020_017_023_010.csv`
- Conservative ranking by `0.35*ts_fold5 + 0.25*tail10 + 0.20*tail20 + 0.20*full`:
  - `v017_single`: composite `0.116791`, full `0.133303`, tail20 `0.120015`, tail10 `0.104541`, ts_fold5 `0.114263`
  - `stable_020_017_023_010`: composite `0.115122`, full `0.142775`, tail20 `0.111257`, tail10 `0.106182`, ts_fold5 `0.107915`
- Interpretation: `v017_single` is the safest second-candidate proxy because it has full OOF coverage and the best conservative late composite, even though its full OOF is below the already submitted ensemble. This is a possible use of the final daily submission only if we accept that the first LB result rewards robustness more than full OOF.
