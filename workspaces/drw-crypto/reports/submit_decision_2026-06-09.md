# DRW Submit Decision - 2026-06-09

## Current Kaggle State

- Competition: `drw-crypto-market-prediction`
- Submitted today: `1/2`
- Remaining local budget: `1`
- Kaggle submissions:
  - `53486449`
  - file: `sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv`
  - public LB: `0.08199`
  - private score shown by Kaggle: `0.08268`

## Current Recommendation

Submit:

```powershell
.\kar.cmd submit drw-crypto --file submissions\sub_calibrated_tail10_batch_tuned.csv
```

Dry-run status: valid.

Do not submit older candidates unless deliberately running an ablation.

## Candidate Summary

Recommended file:

```text
sub_calibrated_tail10_batch_tuned.csv
```

Weights:

| Model | Weight |
| --- | ---: |
| `v032` | `0.543277` |
| `v028` | `0.228819` |
| `v010` | `0.159950` |
| `v017` | `0.032678` |
| `v023` | `0.021254` |
| `v029` | `0.014022` |

Diagnostics:

| Metric | Value |
| --- | ---: |
| full | `0.135859` |
| tail20 | `0.123976` |
| tail10 | `0.117740` |
| ts_fold5 | `0.119889` |
| conservative composite | `0.123363` |
| Spearman vs first submission | `0.908233` |
| mean rank delta vs first submission | `0.089543` |

## Why This Candidate

The first submission had strong local full OOF (`0.149746`) but only reached public LB `0.08199`, so the next submission should not maximize full OOF directly. The current candidate optimizes a recency-weighted proxy:

```text
0.35 * ts_fold5 + 0.25 * tail10 + 0.20 * tail20 + 0.20 * full
```

This candidate is the strongest completed run under that proxy and has more prediction diversity from the first submission than the earlier conservative ensembles.

## After Submission

Run:

```powershell
.\kar.cmd leaderboard drw-crypto
```

Then update:

- `reports/experiment_log.md`
- local submission history if `kar sync-lb` exists later
- README progress if the result materially changes the project status
