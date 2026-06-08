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

## Manual Diverse Conservative Candidate - 2026-06-09

- Tried a larger random diversity-constrained search, but the all-row scoring loop timed out before writing artifacts. This should be optimized before reuse.
- Ran a manual diversity sweep instead. Report: `reports/manual_diverse_candidate_summary.csv`.
- New recommended second-submit candidate: `sub_calibrated_manual_tail10_boost.csv`
- Dry-run: valid
- Weights:
  - `v032`: `0.36`
  - `v028`: `0.22`
  - `v017`: `0.18`
  - `v010`: `0.14`
  - `v029`: `0.10`
- Conservative diagnostics:
  - full: `0.138370`
  - tail20: `0.122587`
  - tail10: `0.117011`
  - ts_fold5: `0.118790`
  - composite: `0.123021`
- Difference versus first submitted ensemble:
  - Spearman: `0.931755`
  - mean rank delta: `0.076877`
- Interpretation: this supersedes the KFold conservative ensemble as the best second-submit recommendation. It improves the conservative proxy (`0.123021` vs `0.122449`), improves tail10 and ts_fold5, and has more prediction diversity versus the first submission.

## Tail10 Boost Tuning Attempt - 2026-06-09

- Tried two local weight tuning methods around `sub_calibrated_manual_tail10_boost.csv`:
  - bounded 0.02 grid search
  - coordinate pairwise weight transfer search
- Both timed out because each evaluation recomputed full-row rank/correlation diagnostics. No new artifact was produced.
- Engineering note: future tuning should precompute segment-centered vectors and approximate rank deltas, or optimize on sampled rows before full validation.
- Decision: keep `sub_calibrated_manual_tail10_boost.csv` as the current best second-submit candidate. Its existing dry-run is valid and its conservative metrics remain the strongest completed evidence.

## Fast Tail10 Weight Tuning - 2026-06-09

- Reimplemented the local weight search with precomputed segment-centered OOF matrices and a reduced 5k search. Report: `reports/tail10_fast_tuning.csv`.
- New recommended second-submit candidate: `sub_calibrated_tail10_fast_tuned.csv`
- Dry-run: valid
- Weights:
  - `v032`: `0.459974`
  - `v028`: `0.194797`
  - `v010`: `0.171337`
  - `v017`: `0.120635`
  - `v029`: `0.053257`
- Conservative diagnostics:
  - full: `0.136446`
  - tail20: `0.123829`
  - tail10: `0.117130`
  - ts_fold5: `0.119877`
  - composite: `0.123294`
- Difference versus first submitted ensemble:
  - Spearman: `0.913948`
  - mean rank delta: `0.086738`
- Interpretation: this supersedes `sub_calibrated_manual_tail10_boost.csv` as the current best second-submit candidate. It improves composite (`0.123294` vs `0.123021`) and prediction diversity while keeping the same conservative objective.

## Expanded Tail10 Fast Tuning - 2026-06-09

- Expanded the fast segment search from five models to eight candidates by adding capped `v023`, `v031`, and `v018`. Report: `reports/tail10_expanded_tuning.csv`.
- New recommended second-submit candidate: `sub_calibrated_tail10_expanded_tuned.csv`
- Dry-run: valid
- Weights:
  - `v032`: `0.503893`
  - `v028`: `0.219887`
  - `v010`: `0.157249`
  - `v017`: `0.083438`
  - `v029`: `0.024288`
  - `v023`: `0.011245`
  - `v031`: `0.000000`
  - `v018`: `0.000000`
- Conservative diagnostics:
  - full: `0.136082`
  - tail20: `0.123994`
  - tail10: `0.117462`
  - ts_fold5: `0.119903`
  - composite: `0.123347`
- Difference versus first submitted ensemble:
  - Spearman: `0.910802`
  - mean rank delta: `0.088205`
- Interpretation: this supersedes `sub_calibrated_tail10_fast_tuned.csv` as the current best second-submit candidate. The gain is small but consistent on composite, tail10, and diversity versus the first submission.

## Refined Tail10 Core Search - 2026-06-09

- Ran a finer fast search around the `tail10_expanded_tuned` core models. Report: `reports/tail10_refined_tuning.csv`.
- New recommended second-submit candidate: `sub_calibrated_tail10_refined_tuned.csv`
- Dry-run: valid
- Weights:
  - `v032`: `0.536970`
  - `v028`: `0.223221`
  - `v010`: `0.158697`
  - `v017`: `0.047228`
  - `v029`: `0.020876`
  - `v023`: `0.013008`
- Conservative diagnostics:
  - full: `0.135723`
  - tail20: `0.124063`
  - tail10: `0.117680`
  - ts_fold5: `0.119955`
  - composite: `0.123361`
- Difference versus first submitted ensemble:
  - Spearman: `0.907469`
  - mean rank delta: `0.089917`
- Interpretation: this supersedes `sub_calibrated_tail10_expanded_tuned.csv` as the current best second-submit candidate. The gain is small but improves all target recency diagnostics and further increases prediction diversity versus the first submission.
## Current Submit Decision Snapshot

- See `reports/submit_decision_2026-06-09.md` for the current second-submit recommendation and exact command.

## Batch Tail10 Core Search - 2026-06-09

- Reworked the fast search into chunked vectorized scoring to avoid the previous all-at-once memory blowup. Report: `reports/tail10_batch_tuning.csv`.
- New numerically best second-submit candidate: `sub_calibrated_tail10_batch_tuned.csv`
- Dry-run: valid
- Weights:
  - `v032`: `0.543277`
  - `v028`: `0.228819`
  - `v010`: `0.159950`
  - `v017`: `0.032678`
  - `v023`: `0.021254`
  - `v029`: `0.014022`
- Conservative diagnostics:
  - full: `0.135859`
  - tail20: `0.123976`
  - tail10: `0.117740`
  - ts_fold5: `0.119889`
  - composite: `0.123363`
- Difference versus first submitted ensemble:
  - Spearman: `0.908233`
  - mean rank delta: `0.089543`
- Interpretation: this slightly supersedes `sub_calibrated_tail10_refined_tuned.csv` on the composite score. The gain is tiny (`~0.0000015`), so both candidates are effectively equivalent; use `sub_calibrated_tail10_batch_tuned.csv` only to follow the latest recorded optimum.

## CLI Reproduction - Tail Ensemble - 2026-06-09

- Ran the new reusable command:
  `kar drw-tail-ensemble drw-crypto --samples 12000 --seed 47 --output-tag tail_cli_full`
- Output: `sub_calibrated_tail_cli_full.csv`
- Dry-run: valid
- Result reproduced the batch tuned weights and scores:
  - composite: `0.123363`
  - full: `0.135859`
  - tail20: `0.123976`
  - tail10: `0.117740`
  - ts_fold5: `0.119889`
  - Spearman vs first submission: `0.908233`
- Decision: use `sub_calibrated_tail_cli_full.csv` as the locked second-submit file because it is directly reproducible from the CLI.

## Submission Audit - tail_cli_full - 2026-06-09

- Audited `sub_calibrated_tail_cli_full.csv` against `sample_submission.csv`.
- Rows: `538150`
- Columns: `ID,prediction`
- ID order match: `true`
- Missing predictions: `0`
- Duplicate IDs: `0`
- Prediction mean/std: `0.000000 / 0.496348`
- Prediction min/max: `-0.998667 / 0.998202`
- Result: schema and distribution checks passed.

## LB Sync - 2026-06-09

- Added `kar sync-lb <workspace>` and ran `kar sync-lb drw-crypto`.
- Wrote `reports/lb_sync.csv`.
- Synced first Kaggle submission into local history:
  - ref: `53486449`
  - public LB: `0.08199`
  - private score shown by Kaggle: `0.08268`
  - status: `SubmissionStatus.COMPLETE`

## Second Submission Result - tail_cli_full - 2026-06-09

- Submitted `sub_calibrated_tail_cli_full.csv` with `--force` because the normal CV threshold guard correctly blocked it against the higher full OOF first submission.
- Kaggle ref: `53487669`
- Public LB: `0.07184`
- Private score shown by Kaggle: `0.08128`
- Budget after submission: `2/2` used, `0` remaining.
- Synced with `kar sync-lb drw-crypto`; `reports/lb_sync.csv` and local submission history now contain both Kaggle refs.
- Interpretation: the recency-weighted composite (`0.123363`) did not improve public LB versus the first submission (`0.08199`). Keep the first submission as current Kaggle best and avoid re-submitting lower-confidence variants of this tail CLI candidate.

## Anchor Blend Candidate Search - 2026-06-09

- Added `kar drw-anchor-blend`, a DRW helper that uses the best real LB submission as an anchor and scans small rank blends against selected model groups.
- Rationale: the tail proxy submission stayed highly correlated with the first submission (`Spearman=0.908233`) but still reduced public LB from `0.08199` to `0.07184`; the next candidate should make a much smaller move from the known-good anchor.
- Generated two valid candidates:
  - `sub_anchor_blend_safe.csv`
    - group: `v032`
    - alpha: `0.21`
    - composite: `0.125443`
    - Spearman to best anchor: `0.991515`
    - Spearman to failed tail submission: `0.948163`
    - mean rank delta to anchor: `0.028320`
  - `sub_anchor_blend_conservative.csv`
    - group: `v016+v017+v031+v032`
    - alpha: `0.21`
    - composite: `0.125011`
    - Spearman to best anchor: `0.994191`
    - Spearman to failed tail submission: `0.938277`
    - mean rank delta to anchor: `0.023247`
- Both candidates passed schema checks against `sample_submission.csv`:
  - rows: `538150`
  - columns: `ID,prediction`
  - ID order match: `true`
  - missing predictions: `0`
  - duplicate IDs: `0`
- Decision: recommend `sub_anchor_blend_conservative.csv` for the next available submission because it is closer to the current public-LB best and further from the failed tail direction. Use `sub_anchor_blend_safe.csv` only if deliberately accepting higher movement toward the `v032` tail signal.

## Ridge v033 and Anchor Blend Check - 2026-06-09

- Ran a bounded Ridge experiment:
  `kar drw-ridge drw-crypto --top-k 210 --cv timeseries --folds 5 --alphas 3000,5000,7500,10000,30000`
- Output model: `models/v033`
- Output submission: `sub_076_v033.csv`
- Best alpha: `7500`
- OOF Pearson: `0.135032`
- Fold scores: `0.193575, 0.130195, 0.170884, 0.124920, 0.080229`
- Diagnostics versus submitted candidates:
  - tail20: `0.082329`
  - tail10: `0.140684`
  - tail5: `0.174613`
  - Spearman to best anchor: `0.905584`
  - Spearman to failed tail submission: `0.885442`
  - Spearman to conservative anchor blend: `0.918073`
- Ran an anchor-blend scan including `v033`:
  `kar drw-anchor-blend drw-crypto --groups with_v033:v021+v023+v033,safe033:v016+v017+v031+v032+v033,v033:v033 --alpha-grid 0.08,0.10,0.12,0.15,0.18,0.20,0.21 --min-spearman 0.994 --max-rank-delta 0.025 --output-tag anchor_blend_v033`
- Best valid v033 blend: `sub_anchor_blend_v033.csv`
  - group: `v016+v017+v031+v032+v033`
  - alpha: `0.21`
  - composite: `0.123785`
  - Spearman to best anchor: `0.995527`
  - Spearman to failed tail submission: `0.934960`
  - mean rank delta to anchor: `0.020516`
- Decision: do not replace `sub_anchor_blend_conservative.csv` as the next-submit candidate. The v033 blend is more conservative but has lower composite (`0.123785` vs `0.125011`) and does not add enough evidence to justify using the next Kaggle submission on it.

## Anchor Blend Metadata Weight Fix - 2026-06-09

- Fixed `kar drw-anchor-blend` to reconstruct the anchor OOF from the anchor submission's adjacent JSON metadata when available.
- Previous scans used equal-weight `--anchor-models` as the local OOF anchor even though the submitted rank ensemble had optimized weights:
  - `v010`: `0.178172`
  - `v017`: `0.162450`
  - `v019`: `0.143619`
  - `v021`: `0.453657`
  - `v023`: `0.062102`
- Re-generated the anchor blend reports and submissions:
  - `sub_anchor_blend_safe.csv`
    - group: `v032`
    - alpha: `0.21`
    - corrected composite: `0.129910`
    - Spearman to best anchor: `0.991515`
    - Spearman to failed tail submission: `0.948163`
    - mean rank delta to anchor: `0.028320`
  - `sub_anchor_blend_conservative.csv`
    - group: `v016+v017+v031+v032`
    - alpha: `0.21`
    - corrected composite: `0.129544`
    - full: `0.140456`
    - tail20 / ts_fold5: `0.115519`
    - tail10: `0.109920`
    - tail5: `0.151956`
    - Spearman to best anchor: `0.994191`
    - Spearman to failed tail submission: `0.938277`
    - mean rank delta to anchor: `0.023247`
  - `sub_anchor_blend_v033.csv`
    - group: `v016+v017+v031+v032+v033`
    - alpha: `0.21`
    - corrected composite: `0.128466`
    - Spearman to best anchor: `0.995527`
    - Spearman to failed tail submission: `0.934960`
    - mean rank delta to anchor: `0.020516`
- Decision remains unchanged: submit `sub_anchor_blend_conservative.csv` first after budget reset. The corrected score is stronger than previously recorded while preserving the more conservative move away from the public-best anchor.

## Anchor Blend Failed-Direction Risk Scan - 2026-06-09

- Ran a focused risk scan with higher alpha and lower-failed-direction groups:
  `kar drw-anchor-blend drw-crypto --groups conservative:v016+v017+v031+v032,low_failed:v021+v023+v017,balanced_no_v032:v016+v017+v021+v023,strong_core:v021+v023+v025,v021:v021,v023:v023 --alpha-grid 0.10,0.12,0.15,0.18,0.20,0.21,0.22,0.24 --min-spearman 0.993 --max-rank-delta 0.030 --output-tag anchor_blend_risk_scan`
- Output: `sub_anchor_blend_risk_scan.csv`
- Report: `reports/anchor_blend_risk_scan_anchor_blend_scan.csv`
- Best by raw composite was the same conservative model group at `alpha=0.22`:
  - composite: `0.129691`
  - Spearman to best anchor: `0.993600`
  - Spearman to failed tail submission: `0.939500`
  - mean rank delta to anchor: `0.024383`
- Current recommendation remains `sub_anchor_blend_conservative.csv` at `alpha=0.21`:
  - composite: `0.129544`
  - Spearman to best anchor: `0.994191`
  - Spearman to failed tail submission: `0.938277`
  - mean rank delta to anchor: `0.023247`
- Safer low-failed-direction alternatives were too weak:
  - `balanced_no_v032` at `alpha=0.24`: composite `0.126893`, Spearman to failed `0.913455`
  - `low_failed` at `alpha=0.24`: composite `0.125755`, Spearman to failed `0.905323`
- Decision: do not replace the next-submit candidate. The `alpha=0.22` gain is only `0.000147` composite while moving closer to the known failed submission and further from the best anchor.

## Risk-Aware Anchor Blend Selection - 2026-06-09

- Added risk-aware selection options to `kar drw-anchor-blend`:
  - `--selection-metric composite|utility`
  - `--failed-threshold`
  - `--risk-penalty`
- The command now writes `utility`, `failed_excess`, `anchor_shortfall`, and `rank_delta_excess` to scan reports.
- Ran:
  `kar drw-anchor-blend drw-crypto --groups conservative:v016+v017+v031+v032,low_failed:v021+v023+v017,balanced_no_v032:v016+v017+v021+v023,strong_core:v021+v023+v025,v021:v021,v023:v023 --alpha-grid 0.10,0.12,0.15,0.18,0.20,0.21,0.22,0.24 --min-spearman 0.993 --max-rank-delta 0.030 --selection-metric utility --failed-threshold 0.935 --risk-penalty 0.60 --output-tag anchor_blend_utility_scan`
- Output: `sub_anchor_blend_utility_scan.csv`
- Report: `reports/anchor_blend_utility_scan_anchor_blend_scan.csv`
- Selected candidate:
  - group: `v016+v017+v031+v032`
  - alpha: `0.18`
  - composite: `0.129080`
  - utility: `0.129080`
  - Spearman to best anchor: `0.995783`
  - Spearman to failed tail submission: `0.934484`
  - mean rank delta to anchor: `0.019854`
- Audit passed against sample submission:
  - rows: `538150`
  - columns: `ID,prediction`
  - ID order match: `true`
  - missing predictions: `0`
  - duplicate IDs: `0`
  - prediction mean/std: `0.000000 / 0.559251`
  - prediction min/max: `-0.999989 / 0.999991`
- Decision update: make `sub_anchor_blend_utility_scan.csv` the next-submit candidate. It gives up `0.000464` raw composite versus `sub_anchor_blend_conservative.csv` (`alpha=0.21`) but stays below the failed-direction threshold and closer to the current public-best anchor.

## Weighted Ridge Probe - 2026-06-09

- Tested a recency-weighted closed-form Ridge prototype on the v021-style feature set (`top_k=180` plus public-diverse features).
- Grid:
  - decay: `1.0, 0.995, 0.99, 0.985, 0.97`
  - alpha: `3000, 7500, 30000, 100000`
  - CV: `TimeSeriesSplit(n_splits=5)`
- Report: `reports/weighted_ridge_probe.csv`
- Best result remained the unweighted baseline:
  - decay: `1.0`
  - alpha: `3000`
  - full: `0.143261`
  - tail20: `0.086090`
  - tail10: `0.139815`
  - tail5: `0.176629`
  - composite: `0.128930`
- Best weighted run (`decay=0.995`, `alpha=3000`) was slightly worse:
  - full: `0.143249`
  - tail20: `0.086057`
  - tail10: `0.139800`
  - tail5: `0.176603`
  - composite: `0.128910`
- Decision: do not promote recency-weighted Ridge into a DRW CLI command yet. The tested time-decay weighting does not add useful signal and remains below the current next-submit utility candidate (`0.129080` composite).

## Anti-Failed Extrapolation Probe - 2026-06-09

- Tested direct public-feedback geometry using the two real submissions:
  - best anchor public LB: `0.08199`
  - failed tail public LB: `0.07184`
- Probe formula:
  `rank(first + beta * (first - failed_tail))`
- Report: `reports/anti_failed_extrapolation_probe.csv`
- Generated diagnostic fallback: `sub_anti_failed_rank_beta020.csv`
- Selected diagnostic beta:
  - mode: `rank`
  - beta: `0.20`
  - Spearman to best anchor: `0.996688`
  - Spearman to failed tail submission: `0.871446`
  - Spearman to utility candidate: `0.986599`
  - mean rank delta to anchor: `0.017765`
  - prediction mean/std: `0.000000 / 0.577350`
- Dry-run status: valid.
- Decision: keep `sub_anchor_blend_utility_scan.csv` as the first next submission. The anti-failed extrapolation has no OOF score and is directly public-feedback-derived, so it is only a diagnostic fallback if the utility candidate fails to improve LB.

## Submission Comparison CLI - 2026-06-09

- Added `kar drw-compare-submissions` to compare candidate CSVs before spending Kaggle submission budget.
- The command validates each file, reads adjacent JSON metadata when present, writes summary and pairwise reports, and prints a compact table.
- The comparison parser now normalizes local score sources in this order: `scores.utility`, `scores.composite`, `oof_pearson`, `mean_score`.
- Ran:
  `kar drw-compare-submissions drw-crypto --files sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv,sub_calibrated_tail_cli_full.csv,sub_anchor_blend_utility_scan.csv,sub_anchor_blend_conservative.csv,sub_anti_failed_rank_beta020.csv --output-tag next_submit_compare`
- Reports:
  - `reports/next_submit_compare_summary.csv`
  - `reports/next_submit_compare_pairs.csv`
- Key checks:
  - first submitted ensemble: valid, local score `0.149746` from `oof_pearson`, public LB `0.08199`.
  - `sub_anchor_blend_utility_scan.csv`: valid, composite `0.129080`, utility `0.129080`, Spearman to anchor `0.995783`, Spearman to failed `0.934484`.
  - `sub_anchor_blend_conservative.csv`: valid, composite `0.129544`, Spearman to anchor `0.994191`, Spearman to failed `0.938277`.
  - `sub_anti_failed_rank_beta020.csv`: valid, no OOF score, Spearman to anchor `0.996688`, Spearman to failed `0.871446`.
- Decision: keep the utility candidate as the next real submit, because it is the highest-scored candidate that stays below the configured failed-direction threshold and has proper metadata.

## Third Submission Result - 2026-06-10

- Submitted `sub_anchor_blend_utility_scan.csv` with a one-off manual budget override.
- Kaggle ref: `53488239`
- Public LB: `0.07695`
- Private score shown by Kaggle: `0.07766`
- Local utility/composite used for selection: `0.129080`
- Outcome:
  - Better than the second tail-calibrated submission (`0.07184` public).
  - Worse than the first public-best ensemble (`0.08199` public).
- Interpretation: the risk-aware anchor blend corrected part of the failed second submission's drift, but still gave up too much of the first submission's public-LB geometry. Local composite and utility remain only weakly aligned with the public leaderboard for this competition.
- Next direction: prioritize candidates with even higher Spearman similarity to the first submission, or test the already-generated anti-failed diagnostic fallback only as an explicit public-feedback probe.

## Post-Third Conservative Candidate Scan - 2026-06-10

- Generated smaller anchor moves after the third submission failed to beat the first public LB.
- Commands:
  - `kar drw-anchor-blend drw-crypto --groups conservative:v016+v017+v031+v032,balanced_no_v032:v016+v017+v021+v023,low_failed:v021+v023+v017,v017:v017,v021:v021,v023:v023 --alpha-grid 0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12 --min-spearman 0.997 --max-rank-delta 0.015 --selection-metric utility --failed-threshold 0.930 --risk-penalty 0.80 --output-tag anchor_blend_micro_scan`
  - `kar drw-anchor-blend drw-crypto --groups conservative:v016+v017+v031+v032,balanced_no_v032:v016+v017+v021+v023,low_failed:v021+v023+v017,v017:v017,v021:v021,v023:v023 --alpha-grid 0.04,0.05,0.06,0.08,0.10,0.12,0.14,0.16 --min-spearman 0.996 --max-rank-delta 0.020 --selection-metric utility --failed-threshold 0.925 --risk-penalty 1.20 --output-tag anchor_blend_low_failed_scan`
- Reports:
  - `reports/anchor_blend_micro_scan_anchor_blend_scan.csv`
  - `reports/anchor_blend_low_failed_scan_anchor_blend_scan.csv`
- Selected model-backed next candidate: `sub_anchor_blend_micro_scan.csv`
  - local utility/composite: `0.128042`
  - Spearman to first anchor: `0.998170`
  - Spearman to failed tail: `0.926368`
  - mean rank delta to anchor: `0.013142`
- Backup: `sub_anchor_blend_low_failed_scan.csv`
  - local utility/composite: `0.127900`
  - Spearman to first anchor: `0.997267`
  - Spearman to failed tail: `0.923697`
  - mean rank delta to anchor: `0.015701`
- Decision: if the next submission must remain model-backed, use `sub_anchor_blend_micro_scan.csv`. It is closer to the first public-best submission than the failed third submission while still reducing failed-tail similarity.

## Anti-Failed CLI And Candidate Family - 2026-06-10

- Added `kar drw-anti-failed` to turn a known failed submission into an explicit negative rank direction.
- Command:
  `kar drw-anti-failed drw-crypto --beta-grid 0.04,0.06,0.08,0.10,0.12,0.15 --output-tag anti_failed_rank_family`
- Report: `reports/anti_failed_rank_family.csv`
- Generated candidates:
  - `sub_anti_failed_rank_beta040.csv`
  - `sub_anti_failed_rank_beta060.csv`
  - `sub_anti_failed_rank_beta080.csv`
  - `sub_anti_failed_rank_beta100.csv`
  - `sub_anti_failed_rank_beta120.csv`
  - `sub_anti_failed_rank_beta150.csv`
- Best diagnostic tradeoff: `sub_anti_failed_rank_beta100.csv`
  - no OOF/local score by design
  - Spearman to first anchor: `0.999154`
  - Spearman to failed tail: `0.890386`
  - mean rank delta to anchor: `0.008920`
- Decision: do not mix this with model-backed local scores. Treat `beta100` as the next diagnostic public-feedback probe only if we choose to spend one submission testing whether moving away from the second failed direction improves LB.
