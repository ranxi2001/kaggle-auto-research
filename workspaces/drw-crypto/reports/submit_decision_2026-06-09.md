# DRW Submit Review - 2026-06-09

## Current Kaggle State

- Competition: `drw-crypto-market-prediction`
- Submitted today: `2/2`
- Remaining local budget: `0`
- Kaggle submissions:
  - `53487669`
  - file: `sub_calibrated_tail_cli_full.csv`
  - public LB: `0.07184`
  - private score shown by Kaggle: `0.08128`
  - `53486449`
  - file: `sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv`
  - public LB: `0.08199`
  - private score shown by Kaggle: `0.08268`

## Outcome

The second submission underperformed the first public LB result.

| File | Local score | Public LB | Private shown |
| --- | ---: | ---: | ---: |
| `sub_ensemble_ranknorm_v005_v010_v012_v015_v017_v018_v019_v020_v021_v022_v023_v024_v025_v026.csv` | `0.149746` | `0.08199` | `0.08268` |
| `sub_calibrated_tail_cli_full.csv` | `0.123363` | `0.07184` | `0.08128` |

Interpretation: the recency-weighted proxy reduced local overfit signals, but did not translate to public LB improvement. Keep the first submission as the current Kaggle best. Do not re-submit the tail CLI candidate.

## Candidate Summary

Submitted file:

```text
sub_calibrated_tail_cli_full.csv
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

Submission audit:

| Check | Value |
| --- | --- |
| row count | `538150` |
| columns | `ID,prediction` |
| ID order matches sample | `true` |
| missing predictions | `0` |
| duplicate IDs | `0` |
| prediction mean | `0.000000` |
| prediction std | `0.496348` |
| prediction min/max | `-0.998667 / 0.998202` |

## Why This Candidate

The first submission had strong local full OOF (`0.149746`) but only reached public LB `0.08199`, so the next submission should not maximize full OOF directly. The current candidate optimizes a recency-weighted proxy:

```text
0.35 * ts_fold5 + 0.25 * tail10 + 0.20 * tail20 + 0.20 * full
```

This candidate is the strongest completed run under that proxy and has more prediction diversity from the first submission than the earlier conservative ensembles.

The experiment shows this proxy is not sufficient as a submission selector on its own.

## Next Iteration Direction

- Treat `0.08199` public LB as the current external anchor.
- Use `kar sync-lb drw-crypto` after future submissions to keep `reports/lb_sync.csv` and `submissions/history.json` current.
- Build the next candidate around direct LB calibration, not only tail OOF proxies.
- Avoid submitting candidates that are mostly lower-confidence variants of `sub_calibrated_tail_cli_full.csv`.

## Reproduce The Candidate Search

The calibrated tail ensemble search is available as a CLI command:

```powershell
.\kar.cmd drw-tail-ensemble drw-crypto --samples 12000 --seed 47 --output-tag tail_cli
```

The command above reproduces the same weights as the locked recommendation when run as:

```powershell
.\kar.cmd drw-tail-ensemble drw-crypto --samples 12000 --seed 47 --output-tag tail_cli_full
```

Use the explicit submit command above for the current locked recommendation.
