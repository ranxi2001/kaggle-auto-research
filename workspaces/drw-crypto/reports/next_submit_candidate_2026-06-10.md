# DRW Next Submit Candidate - 2026-06-10

## Current State

- Competition: `drw-crypto-market-prediction`
- Current Kaggle best: `sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv`
- Current best public LB: `0.08199`
- Failed second submission: `sub_calibrated_tail_cli_full.csv`
- Failed second public LB: `0.07184`
- Local budget on 2026-06-09: `2/2` used, `0` remaining

## Recommendation

When the daily budget resets, submit the utility-selected conservative anchor blend first:

```powershell
.\kar.cmd submit drw-crypto --file submissions\sub_anchor_blend_utility_scan.csv --force
```

Use `--force` because the candidate is intentionally not optimizing the original full OOF score. It is anchored against the best real LB submission and selected by a utility score that penalizes similarity to the known failed submission.

## Candidate

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

## Alternative

`sub_anchor_blend_conservative.csv` has higher raw local composite (`0.129544`) but crosses the failed-direction threshold used by the utility selector:

| File | Composite | Spearman to best anchor | Spearman to failed tail | Rank delta to anchor |
| --- | ---: | ---: | ---: | ---: |
| `sub_anchor_blend_utility_scan.csv` | `0.129080` | `0.995783` | `0.934484` | `0.019854` |
| `sub_anchor_blend_conservative.csv` | `0.129544` | `0.994191` | `0.938277` | `0.023247` |
| `sub_anchor_blend_safe.csv` | `0.129910` | `0.991515` | `0.948163` | `0.028320` |

Prefer the utility candidate for the next submission because the first two real submissions showed that local composite alone overstates riskier moves. Use `sub_anchor_blend_conservative.csv` only if deliberately accepting a higher failed-direction similarity for more local score.

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
| conservative `alpha=0.18` | `0.129080` | `0.995783` | `0.934484` | `0.019854` | submit first |
| conservative `alpha=0.21` | `0.129544` | `0.994191` | `0.938277` | `0.023247` | higher composite, more failed-direction risk |
| conservative `alpha=0.22` | `0.129691` | `0.993600` | `0.939500` | `0.024383` | too little gain for more drift |
| balanced_no_v032 `alpha=0.24` | `0.126893` | `0.997147` | `0.913455` | `0.016538` | safer but too weak |
| low_failed `alpha=0.24` | `0.125755` | `0.998162` | `0.905323` | `0.013302` | safer but too weak |

The lower-failed-direction groups are useful diagnostics, but their local composite drop is too large for the next submission. Keep the existing conservative candidate.

## Diagnostic Fallback

`sub_anti_failed_rank_beta020.csv` was generated from the two real LB submissions:

```text
rank(first + 0.20 * (first - failed_tail))
```

It is a diagnostic fallback, not the first submission choice, because it has no OOF score and directly uses public-LB feedback geometry.

| File | Spearman to best anchor | Spearman to failed tail | Spearman to utility candidate | Rank delta to anchor | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `sub_anchor_blend_utility_scan.csv` | `0.995783` | `0.934484` | `1.000000` | `0.019854` | submit first |
| `sub_anti_failed_rank_beta020.csv` | `0.996688` | `0.871446` | `0.986599` | `0.017765` | diagnostic fallback only |

If the utility candidate does not improve LB, this anti-failed candidate can test whether the failed second submission direction is actively harmful. Do not submit it before the utility candidate.

## After Submission

Run:

```powershell
.\kar.cmd leaderboard drw-crypto
.\kar.cmd sync-lb drw-crypto
```

Then update `reports/experiment_log.md` and `reports/lb_sync.csv`.
