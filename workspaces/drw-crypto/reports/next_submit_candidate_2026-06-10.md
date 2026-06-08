# DRW Submission Result - 2026-06-10

## Current State

- Competition: `drw-crypto-market-prediction`
- Current Kaggle best: `sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv`
- Current best public LB: `0.08199`
- Failed second submission: `sub_calibrated_tail_cli_full.csv`
- Failed second public LB: `0.07184`
- Third submission: `sub_anchor_blend_utility_scan.csv`
- Third public LB: `0.07695`
- Third private score shown by Kaggle: `0.07766`
- Local budget after manual override submit: `3/2` used, `0` remaining

## Result

The utility-selected conservative anchor blend was submitted with a one-off manual budget override:

```powershell
sub_anchor_blend_utility_scan.csv
```

It improved over the second tail-calibrated submission (`0.07184` -> `0.07695`) but did not recover the first submission's public LB (`0.08199`). This confirms that anchoring and failed-direction penalties helped, but the blend still moved away from the strongest public leaderboard geometry.

## Next Recommendation

Do not keep pushing local-composite anchor blends unless they preserve more of the first submission. The next useful branch is either an even smaller anchor move or a diagnostic anti-failed extrapolation.

After the third submission result, the next offline candidates are:

| Candidate | Type | Local score | Spearman to best anchor | Spearman to failed tail | Rank delta to anchor | Use |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `sub_anchor_blend_micro_scan.csv` | model-based micro anchor blend | `0.128042` | `0.998170` | `0.926368` | `0.013142` | safest model-based next candidate |
| `sub_anchor_blend_low_failed_scan.csv` | model-based lower-failed blend | `0.127900` | `0.997267` | `0.923697` | `0.015701` | backup if accepting more local-score loss |
| `sub_anti_failed_rank_beta100.csv` | public-feedback anti-failed probe |  | `0.999154` | `0.890386` | `0.008920` | diagnostic candidate if testing failed-direction harm |
| `sub_anti_failed_rank_beta120.csv` | public-feedback anti-failed probe |  | `0.998787` | `0.886679` | `0.010695` | more aggressive diagnostic variant |

Recommendation for the next real submission is `sub_anchor_blend_micro_scan.csv` if staying within model-backed candidates. If deliberately using public LB feedback as the main signal, prefer `sub_anti_failed_rank_beta100.csv` over the earlier beta `0.20` because it is much closer to the first submission while still moving clearly away from the failed second submission.

After adding third-submission geometry scoring, `sub_anchor_blend_micro_scan.csv` remains the preferred model-backed next submit:

| Candidate | Geometry score | Local score | Anchor Spearman | Max failed Spearman | Rank delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sub_anchor_blend_micro_scan.csv` | `0.664044` | `0.128042` | `0.998170` | `0.999508` | `0.013142` |
| `sub_anchor_blend_low_failed_scan.csv` | `0.662822` | `0.127900` | `0.997267` | `0.999162` | `0.015701` |
| `sub_anti_failed_rank_w100_040.csv` | `0.650203` |  | `0.999030` | `0.991509` | `0.009562` |

The dual anti-failed family is useful as a diagnostic probe, but it has no local score and is still highly correlated with the third submission. Keep `sub_anchor_blend_micro_scan.csv` first unless explicitly choosing a public-feedback-only experiment.

Model geometry audit added one more exploratory candidate:

| Candidate | Role | Local composite | Anchor Spearman | Max failed Spearman | Rank delta | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `sub_anchor_blend_micro_scan.csv` | conservative next submit | `0.128042` | `0.998170` | `0.999508` | `0.013142` | submit first if optimizing for safety |
| `sub_low_failed_pool_random_best.csv` | exploratory random low-failed pool | `0.128656` | `0.929671` | `0.954688` | `0.079369` | backup exploration; valid but riskier |
| `sub_low_failed_pool_grid.csv` | exploratory low-failed pool | `0.128676` | `0.911003` | `0.939423` | `0.090004` | submit only if deliberately testing a larger geometry move |

`sub_low_failed_pool_random_best.csv` came from the random low-failed model-pool report and has the best pool utility (`0.132554`), but it is still much farther from the first public-best anchor than the queued micro blend. `sub_low_failed_pool_grid.csv` is a simpler equal-weight exploration branch. Treat both as exploration branches, not the default next submission.

Queue status: `sub_anchor_blend_micro_scan.csv` is reserved in `.state/submission_budget.json` with CV/utility `0.128042`. After the daily budget resets, run:

```powershell
.\kar.cmd submit drw-crypto --flush
```

Before flushing, verify with `.\kar.cmd submit drw-crypto --status` that the reserve queue contains only `sub_anchor_blend_micro_scan.csv`.

Tracked handoff plan: `reports/next_submit_plan.md`. A local ignored JSON mirror may also exist for automation.

## Third Submission Details

File:

```text
sub_anchor_blend_utility_scan.csv
```

Generation command:

```powershell
.\kar.cmd drw-anchor-blend drw-crypto --groups conservative:v016+v017+v031+v032,low_failed:v021+v023+v017,balanced_no_v032:v016+v017+v021+v023,strong_core:v021+v023+v025,v021:v021,v023:v023 --alpha-grid 0.10,0.12,0.15,0.18,0.20,0.21,0.22,0.24 --min-spearman 0.993 --max-rank-delta 0.030 --selection-metric utility --failed-threshold 0.935 --risk-penalty 0.60 --output-tag anchor_blend_utility_scan
```

Blend:

| Field | Value |
| --- | ---: |
| anchor | first submitted public-best ensemble |
| anchor OOF source | metadata-weighted rank ensemble |
| group | `v016+v017+v031+v032` |
| alpha | `0.18` |
| utility | `0.129080` |
| composite | `0.129080` |
| full | `0.140030` |
| tail20 / ts_fold5 | `0.114907` |
| tail10 | `0.109608` |
| tail5 | `0.151531` |
| Spearman to best anchor | `0.995783` |
| Spearman to failed tail submission | `0.934484` |
| mean rank delta to anchor | `0.019854` |

Audit:

| Check | Value |
| --- | --- |
| row count | `538150` |
| columns | `ID,prediction` |
| ID order matches sample | `true` |
| missing predictions | `0` |
| duplicate IDs | `0` |
| prediction mean | `0.000000` |
| prediction std | `0.559251` |
| prediction min/max | `-0.999989 / 0.999991` |

## Earlier Alternatives

`sub_anchor_blend_conservative.csv` has higher raw local composite (`0.129544`) but crosses the failed-direction threshold used by the utility selector:

| File | Composite | Spearman to best anchor | Spearman to failed tail | Rank delta to anchor |
| --- | ---: | ---: | ---: | ---: |
| `sub_anchor_blend_utility_scan.csv` | `0.129080` | `0.995783` | `0.934484` | `0.019854` |
| `sub_anchor_blend_conservative.csv` | `0.129544` | `0.994191` | `0.938277` | `0.023247` |
| `sub_anchor_blend_safe.csv` | `0.129910` | `0.991515` | `0.948163` | `0.028320` |

These candidates were evaluated before the third real submission. The utility candidate has already been submitted and scored `0.07695` public LB, so this section is retained for traceability rather than as a current recommendation.

## Rejected Follow-up

`sub_anchor_blend_v033.csv` was generated after adding Ridge `v033` (`top_k=210`, timeseries, alpha `7500`). It is valid but not recommended:

| File | Composite | Spearman to best anchor | Spearman to failed tail | Rank delta to anchor |
| --- | ---: | ---: | ---: | ---: |
| `sub_anchor_blend_utility_scan.csv` | `0.129080` | `0.995783` | `0.934484` | `0.019854` |
| `sub_anchor_blend_v033.csv` | `0.128466` | `0.995527` | `0.934960` | `0.020516` |

The v033 blend is slightly closer to the best anchor, but it gives up local composite. Keep it as a fallback only after the conservative candidate has been tested.

## Risk Scan

`sub_anchor_blend_risk_scan.csv` was generated to compare higher-alpha and lower-failed-direction alternatives. It is valid but not recommended over the current conservative candidate.

| Candidate | Composite | Spearman to best anchor | Spearman to failed tail | Rank delta to anchor | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| conservative `alpha=0.18` | `0.129080` | `0.995783` | `0.934484` | `0.019854` | already submitted |
| conservative `alpha=0.21` | `0.129544` | `0.994191` | `0.938277` | `0.023247` | higher composite, more failed-direction risk |
| conservative `alpha=0.22` | `0.129691` | `0.993600` | `0.939500` | `0.024383` | too little gain for more drift |
| balanced_no_v032 `alpha=0.24` | `0.126893` | `0.997147` | `0.913455` | `0.016538` | safer but too weak |
| low_failed `alpha=0.24` | `0.125755` | `0.998162` | `0.905323` | `0.013302` | safer but too weak |

The lower-failed-direction groups are useful diagnostics. Post-third-submission analysis now favors `sub_anchor_blend_micro_scan.csv` as the queued conservative candidate.

## Diagnostic Fallback

`sub_anti_failed_rank_beta020.csv` was generated from the two real LB submissions:

```text
rank(first + 0.20 * (first - failed_tail))
```

It is a diagnostic fallback, not the first submission choice, because it has no OOF score and directly uses public-LB feedback geometry.

| File | Spearman to best anchor | Spearman to failed tail | Spearman to utility candidate | Rank delta to anchor | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `sub_anchor_blend_utility_scan.csv` | `0.995783` | `0.934484` | `1.000000` | `0.019854` | already submitted |
| `sub_anti_failed_rank_beta020.csv` | `0.996688` | `0.871446` | `0.986599` | `0.017765` | diagnostic fallback only |

The anti-failed candidates can test whether the failed second submission direction is actively harmful, but they remain diagnostics because they have no OOF score.

## Pre-submit Comparison

Run this before the next real submission to verify the candidate set:

```powershell
.\kar.cmd drw-score-candidates drw-crypto --files sub_anchor_blend_micro_scan.csv,sub_low_failed_pool_random_best.csv,sub_low_failed_pool_grid.csv,sub_anti_failed_rank_beta100.csv --output-tag next_with_random_pool_score
```

Current comparison output:

| File | Valid | Local score | Source | Utility | Spearman to failed | Rank delta |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `sub_ensemble_ranknorm_...v026.csv` | `true` | `0.149746` | `oof_pearson` |  |  |  |
| `sub_calibrated_tail_cli_full.csv` | `true` | `0.123363` | `scores.composite` |  |  |  |
| `sub_anchor_blend_utility_scan.csv` | `true` | `0.129080` | `scores.utility` | `0.129080` | `0.934484` | `0.019854` |
| `sub_anchor_blend_conservative.csv` | `true` | `0.129544` | `scores.composite` |  | `0.938277` | `0.023247` |
| `sub_anti_failed_rank_beta020.csv` | `true` |  |  |  | `0.871446` | `0.017765` |

Pairwise checks:

| Pair | Spearman | Mean rank delta |
| --- | ---: | ---: |
| utility vs conservative | `0.999872` | `0.003400` |
| utility vs anti-failed | `0.986599` | `0.035465` |
| anchor vs utility | `0.995783` | `0.019854` |
| failed tail vs utility | `0.934484` | `0.075861` |

## After Submission

Run:

```powershell
.\kar.cmd leaderboard drw-crypto
.\kar.cmd sync-lb drw-crypto
```

Then update `reports/experiment_log.md` and `reports/lb_sync.csv`.
