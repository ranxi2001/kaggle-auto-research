# DRW Next Submit Candidate - 2026-06-10

## Current State

- Competition: `drw-crypto-market-prediction`
- Current Kaggle best: `sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv`
- Current best public LB: `0.08199`
- Failed second submission: `sub_calibrated_tail_cli_full.csv`
- Failed second public LB: `0.07184`
- Local budget on 2026-06-09: `2/2` used, `0` remaining

## Recommendation

When the daily budget resets, submit the conservative anchor blend first:

```powershell
.\kar.cmd submit drw-crypto --file submissions\sub_anchor_blend_conservative.csv --force
```

Use `--force` because the candidate is intentionally not optimizing the original full OOF score. It is anchored against the best real LB submission and should be judged as an LB calibration experiment.

## Candidate

File:

```text
sub_anchor_blend_conservative.csv
```

Generation command:

```powershell
.\kar.cmd drw-anchor-blend drw-crypto --groups safe:v016+v017+v031+v032 --alpha-grid 0.15,0.18,0.19,0.20,0.21 --min-spearman 0.994 --max-rank-delta 0.025 --output-tag anchor_blend_conservative
```

Blend:

| Field | Value |
| --- | ---: |
| anchor | first submitted public-best ensemble |
| anchor OOF source | metadata-weighted rank ensemble |
| group | `v016+v017+v031+v032` |
| alpha | `0.21` |
| composite | `0.129544` |
| full | `0.140456` |
| tail20 / ts_fold5 | `0.115519` |
| tail10 | `0.109920` |
| tail5 | `0.151956` |
| Spearman to best anchor | `0.994191` |
| Spearman to failed tail submission | `0.938277` |
| mean rank delta to anchor | `0.023247` |

Audit:

| Check | Value |
| --- | --- |
| row count | `538150` |
| columns | `ID,prediction` |
| ID order matches sample | `true` |
| missing predictions | `0` |
| duplicate IDs | `0` |
| prediction mean | `0.000000` |
| prediction std | `0.556722` |
| prediction min/max | `-0.999989 / 0.999991` |

## Alternative

`sub_anchor_blend_safe.csv` has slightly higher local composite (`0.129910`) but moves more toward the failed tail direction:

| File | Composite | Spearman to best anchor | Spearman to failed tail | Rank delta to anchor |
| --- | ---: | ---: | ---: | ---: |
| `sub_anchor_blend_safe.csv` | `0.129910` | `0.991515` | `0.948163` | `0.028320` |
| `sub_anchor_blend_conservative.csv` | `0.129544` | `0.994191` | `0.938277` | `0.023247` |

Prefer the conservative candidate unless deliberately testing the stronger `v032` tail signal.

## Rejected Follow-up

`sub_anchor_blend_v033.csv` was generated after adding Ridge `v033` (`top_k=210`, timeseries, alpha `7500`). It is valid but not recommended:

| File | Composite | Spearman to best anchor | Spearman to failed tail | Rank delta to anchor |
| --- | ---: | ---: | ---: | ---: |
| `sub_anchor_blend_conservative.csv` | `0.129544` | `0.994191` | `0.938277` | `0.023247` |
| `sub_anchor_blend_v033.csv` | `0.128466` | `0.995527` | `0.934960` | `0.020516` |

The v033 blend is slightly closer to the best anchor, but it gives up local composite. Keep it as a fallback only after the conservative candidate has been tested.

## After Submission

Run:

```powershell
.\kar.cmd leaderboard drw-crypto
.\kar.cmd sync-lb drw-crypto
```

Then update `reports/experiment_log.md` and `reports/lb_sync.csv`.
