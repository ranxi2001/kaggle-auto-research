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

## Small Top-K Strong-Regularized Ridge - 2026-06-09

- Trained three additional TimeSeries Ridge candidates:
  - `v028`: `top_k=40`, best alpha `30000`, OOF Pearson `0.135792`, fold5 `0.090686`
  - `v029`: `top_k=60`, best alpha `10000`, OOF Pearson `0.134066`, fold5 `0.083143`
  - `v030`: `top_k=100`, best alpha `30000`, OOF Pearson `0.133269`, fold5 `0.089768`
- Single-model conclusion: none of `v028-v030` beats `v017_single` as a conservative standalone candidate.
- Ensemble test: `sub_calibrated_ensemble_conservative_v017_v020_v023_v028_v029_v030_v010.csv` dry-run valid.
- Ensemble weights:
  - `v017`: `0.326560`
  - `v028`: `0.206291`
  - `v010`: `0.180569`
  - `v023`: `0.153317`
  - `v029`: `0.133264`
  - `v020`: `0.000000`
  - `v030`: `0.000000`
- Conservative diagnostics for this ensemble:
  - full: `0.144911`
  - tail20: `0.117256`
  - tail10: `0.113992`
  - ts_fold5: `0.114678`
  - composite: `0.121069`
- Interpretation: this is now a stronger second-submit candidate than `v017_single` under the current LB-gap proxy, because it improves full, tail10, ts_fold5, and composite while staying more conservative than the first submitted full-OOF optimized blend. Real LB is still uncertain, so using the last daily submission remains a deliberate exploration decision.

## KFold Ridge Neighbor Search - 2026-06-09

- Trained KFold/no-shuffle Ridge variants around `v017`:
  - `v031`: `top_k=100`, best alpha `30000`, OOF Pearson `0.121516`, fold5 `0.112676`
  - `v032`: `top_k=200`, best alpha `30000`, OOF Pearson `0.125318`, fold5 `0.101884`
- Single-model diagnostics after rank normalization show `v032` is stronger than `v017` on the current conservative composite, mainly from better tail20/tail10/ts_fold5.
- New current second-submit candidate: `sub_calibrated_ensemble_conservative_kfold_v017_v023_v028_v029_v031_v032_v010.csv`
- Dry-run: valid
- Weights:
  - `v032`: `0.249782`
  - `v017`: `0.170308`
  - `v028`: `0.158697`
  - `v010`: `0.157688`
  - `v023`: `0.105296`
  - `v029`: `0.100625`
  - `v031`: `0.057604`
- Conservative diagnostics:
  - full: `0.141599`
  - tail20: `0.120974`
  - tail10: `0.115559`
  - ts_fold5: `0.117269`
  - composite: `0.122449`
- Interpretation: this supersedes the previous small-Ridge conservative ensemble (`composite 0.121069`) as the best second LB probe. It deliberately sacrifices some full OOF versus the first submitted ensemble to improve recency-weighted diagnostics after the first LB gap.

## Submission Delta Risk Check - 2026-06-09

- Compared the first submitted file against the current KFold conservative candidate. Reports:
  - `reports/submission_delta_analysis.json`
  - `reports/candidate_correlation_matrix.csv`
- Current candidate: `sub_calibrated_ensemble_conservative_kfold_v017_v023_v028_v029_v031_v032_v010.csv`
- Difference versus first submitted ensemble:
  - Pearson: `0.952681`
  - Spearman: `0.954108`
  - mean absolute rank delta: `0.062890`
  - p90 absolute rank delta: `0.146736`
  - top 1% overlap: `0.725887`
  - bottom 1% overlap: `0.635384`
- Interpretation: the current candidate is not a totally new signal; it is a conservative rank adjustment of the first submission. That is appropriate after the first submission showed OOF optimism, but it limits upside.
- Single-model alternatives (`v017_single`, `v032_model`) differ more from the first submission, but carry higher score risk. The current KFold conservative ensemble remains the recommended second probe if using the last daily submission.

## First-LB / KFold Candidate Mix Scan - 2026-06-09

- Scanned linear mixes between the first submitted full-OOF optimized ensemble and the current KFold conservative candidate. Report: `reports/mix_firstlb_kfold_scan.csv`.
- Objective: same conservative recency-weighted proxy (`0.35*ts_fold5 + 0.25*tail10 + 0.20*tail20 + 0.20*full`).
- Best mix: `kfold_weight=1.0`, first-submission weight `0.0`.
- Result: adding the first submitted blend monotonically worsens the conservative proxy, despite increasing full OOF.
- Dry-run valid equivalent file: `sub_calibrated_mix_firstlb_kfold_100.csv`.
- Interpretation: keep the current KFold conservative ensemble as the second-submit recommendation; do not average it back toward the first submitted blend.
