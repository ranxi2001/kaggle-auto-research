# ROGII v006 Latent-Surface Particle Filter

Status: `submitted`. Submission ref `54693365` completed with public RMSE `10.693`.

## Method

v006 tracks latent formation surface `U = TVT + Z` and a slowly varying surface rate. Each suffix
row propagates 500 particles using real MD increments and Z trajectory, then applies a robust GR
likelihood against the paired typewell when the horizontal GR observation is finite. Missing GR
rows skip the measurement and evidence update. Thirty-two deterministic seeds are combined by
sequence evidence, with a nested-CV-selected 20% shrinkage toward the last visible `TVT_input`.

The fixed candidate grid crosses likelihood scales `(3, 5, 8, 12)` with last-known hold weights
`(0, 0.2, 0.5)`. A 12-well smoke test read targets only as a runtime gate; it did not alter this
grid or the particle dynamics. Every outer validation fold selected its candidate using only the
other four folds.

## Audit Correction

v005 was stopped at 230 of 773 wells after audit found that interpolated missing suffix GR values
were being counted as observed evidence. The run is recorded as failed and
`data/features/v002.parquet` is an invalid partial parquet. v006 preserves the original GR missing
mask and supersedes v005 without overwriting it.

Training suffix coverage:

- Observed GR rows: 2,583,152 (68.2653%)
- Missing GR rows: 1,200,837 (31.7347%)
- Training fallback wells: 0 of 773

## Grouped CV

The canonical manifest is `reports/canonical_outer_folds_v001.csv`, SHA-256
`25847e5c05fad652e79dfd2280c8615dec8c528f12acd88ab038533dc7c998fe`. It reproduces row-balanced
five-fold GroupKFold by well over 3,783,989 natural suffix rows.

| Fold | Selected candidate | v006 RMSE | Canonical v004 RMSE |
|---:|---|---:|---:|
| 1 | `pf_scale_12_hold_0p2` | 12.048074 | 14.135772 |
| 2 | `pf_scale_12_hold_0p2` | 11.384291 | 14.291809 |
| 3 | `pf_scale_12_hold_0p2` | 11.547903 | 13.510538 |
| 4 | `pf_scale_12_hold_0p2` | 13.352526 | 15.144821 |
| 5 | `pf_scale_12_hold_0p2` | 12.485944 | 16.460358 |

- Pooled OOF RMSE: `12.184563279387687`
- Fold mean / std: `12.163747603963726 / 0.7095354299225062`
- Absolute improvement over v004: `2.559187305` RMSE
- Relative improvement over v004: `17.357777%`
- Relative improvement over v001: `23.414985%`
- Canonical folds better than v004: `5 / 5`
- Gate: pooled RMSE `<= 12.5`, at least `4 / 5` folds better, and zero fallback wells

The globally selected candidate also leads the fixed scan at 12.184563; the next candidates are
`pf_scale_8_hold_0p2` at 12.248935 and `pf_scale_5_hold_0p2` at 12.299382.

## Artifacts

```text
models/v005/run.json                       # failed audit-invalidated attempt
models/v006/run.json
models/v006/cv_scores.json
models/v006/oof_preds.npy
models/v006/oof_rows.parquet
models/v006/test_preds.npy
models/v006/feature_list.txt
models/v006/importance.csv
models/v006/model.pkl
data/features/v003.parquet                 # 3,783,989 rows, 773 row groups
reports/v006_particle_candidate_scan.csv
reports/v006_particle_candidates_by_well.csv
reports/v006_particle_diagnostics.json
reports/v006_remote_kernel.json
submissions/sub_v006_particle_state_space.csv
submissions/sub_v006_particle_state_space.json
notebooks/v006/rogii-v006.ipynb
notebooks/v006/kernel-metadata.json
```

`run.json` records source SHA-256
`4fd0974cd45951d321986ced758aacb609a001b3d5e699137175d3cdd5b06f9f`, the data fingerprint,
canonical manifest hash, environment, exact temporary command, and a stable reproduction command.
The full CV runtime was 2,901.240 seconds.

## Offline Submission Validation

`kar submit rogii-wellbore-geology-prediction --file
submissions/sub_v006_particle_state_space.csv --dry-run` returned `Valid: Yes`. Before the remote
submission, the local candidate contained 14,151 finite predictions in exact sample ID order.

The generated private, internet-disabled notebook was executed locally with Kaggle-style data,
working, and runtime roots. It processed the three visible test wells in 12.603 seconds with zero
fallbacks and reproduced the local candidate CSV byte for byte.

| Artifact | SHA-256 |
|---|---|
| `notebooks/v006/rogii-v006.ipynb` | `9d4ef742ef53b18a9ae2e179d900afc8342b13af347b909fa050fcc225fe8cae` |
| `notebooks/v006/kernel-metadata.json` | `e718d542b01e525c9dc1afee519d03c6e80b32e2ccd218cc5bdb502f720d6d40` |
| Local v006 candidate CSV | `dec9ddc329ca97ebfee15a6d3118d1afed302f104b6eea6832fdf4a97c2504e3` |
| Notebook dry-run CSV | `dec9ddc329ca97ebfee15a6d3118d1afed302f104b6eea6832fdf4a97c2504e3` |

## Remote Kernel Build

Private kernel `ranxi169/kar-rogii-v006-latent-surface-particle-filter` version 1 reached
`KernelWorkerStatus.COMPLETE`. The visible runtime contained 3 test wells and produced 14,151 rows
in 26.599 seconds with zero fallback wells. Downloaded output SHA-256 is
`dec9ddc329ca97ebfee15a6d3118d1afed302f104b6eea6832fdf4a97c2504e3`, exactly matching both local
inference paths.

The remote log contains only notebook-format, debugger, and nbconvert dependency warnings. It has
no execution error. This build validates the private kernel packaging against the visible test
mount; hidden scoring was performed only by the subsequent competition submission.

Kaggle CLI 2.2.3 resolves `kernels status/output` to the slug's latest session even when a version
suffix is supplied. Version 1 was the only version and no later push occurred during retrieval, so
the downloaded output is unambiguous.

## Leaderboard Result

With explicit user authorization, kernel version 1 was submitted as ref `54693365`. It completed
with public RMSE `10.693`, rank `2,940 / 4,909`. Compared with v004 public RMSE `14.683`, this is an
absolute improvement of `3.990` and a relative improvement of `27.17%`. Public RMSE is `1.491563`
better than the strict v006 grouped OOF estimate.

The 2026-07-14 17:25 UTC leaderboard snapshot places the top-10% boundary at rank 491 and RMSE
`7.120`; v006 remains `3.573` RMSE behind it. The result validates the latent-surface direction but
supports moving next to a full HMM/forward-backward smoother with windowed GR registration.
