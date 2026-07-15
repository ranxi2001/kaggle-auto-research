# ROGII v010 Progress-Shaped HMM Plan

Snapshot: 2026-07-15. This protocol is fixed after v009 completed and before any five-stream v010
prediction is scored against suffix `TVT` targets.

## Statistical Status And Single Hypothesis

V009 scored `11.633993822`, improving all five folds over v006 but missing the `10.95` candidate
gate. Its post-hoc progress audit found that the constant HMM shrink is close to optimal after the
suffix midpoint but materially underweights the first two deciles. V010 tests exactly one derived
target-free basis and does not scan its shape.

For each well, suffix progress is
`q = (row_index - min_suffix_row) / (max_suffix_row - min_suffix_row)`. Every competition well has
multiple suffix rows; a non-positive denominator fails closed. Define the locked taper
`g(q) = max(0, 1 - 5*q)` and the new stream
`hmm_front20 = parent_v006 + g(q) * (hmm_wr21_slow - parent_v006)`. It equals the HMM correction at
the first suffix row, decays linearly, and equals the parent from 20% progress onward. The factor 5,
linear shape, endpoints, and progress definition are fixed from `reports/v009_postmortem.md`.

This is an adaptive target-aware diagnostic, not an independent estimate of the full discovery
process and not candidate-grade evidence. The factor and cutoff were chosen after inspecting target
residuals from all five v009 folds, so fold k's prior target information indirectly influenced the
basis later evaluated on fold k. Outer-fold weight fitting still prevents direct weight/donor
leakage, but it cannot undo this adaptive basis selection. V010 is therefore permanently forbidden
from `candidate_ready`, visible-test inference, or submission artifacts regardless of its score.
There is no taper scan, row/well gate, boundary neutralization, stream selection, regularization,
or post-score retry.

## Locked Five Streams

The exact stream order is:

1. `parent_v006`;
2. `hmm_wr21_slow`;
3. `hmm_front20`;
4. `sp_plane_ancc_k10`;
5. `sp_plane_best6_k10`.

V010 reads and validates the same immutable v004/v005 feature rows and v006-v008 provenance as
v009. It must reuse the globally unique `_id` one-to-one join, canonical v005 order, exact field
assertions, unanimous v007 slow selection, v004-to-v007 OOF value match, 186 inherited boundary
wells, v008 feature/model output hashes, canonical fold manifest, and raw-test fingerprint contract.
`models/v009/run.json`, `reports/v009_artifact_audit.json`, `reports/v009_postmortem.md`, the locked
v009 source, this plan, and the new v010 source are additional critical hashed inputs. Start/end
hashes must match, and the v009 audit must be `verified_no_blocker` with hashes matching the current
v009 artifacts.

## Outer-Fold Boundary

For outer fold `k`, meta fitting uses target rows only from folds `j != k`. Parent, slow HMM, and
front20 are fixed target-free row streams. ANCC and best6 meta streams use only the v008
`nested_outer_k__*` columns whose donor universes exclude both `k` and each query fold `j`.
Evaluation fold `k` uses the canonical formation columns whose donors exclude `k`. No fold-k target,
formation donor, fitted weight, or residual summary enters its v010 meta fit. The already disclosed
front20 basis shape remains indirectly informed by the earlier all-fold v009 postmortem, which is
why the resulting score is diagnostic rather than candidate-grade.

The training progress universe is exactly the canonical joined v005 row set keyed by unique `_id`;
no raw target mask or sample ordering may define it. Within each `_well_id`, `_row_index` must be a
finite exact int64 with no duplicates, at least two rows, and positive `max-min`. The computed `q`
must be finite, lie in `[0,1]`, and attain exact endpoints 0 and 1 for every well. The front20 formula
is recomputed and verified row-for-row in float64. Violations are counted in the integrity gate.
Progress is calculated independently within each well and does not pool distribution statistics
across wells or folds. Final all-OOF weights are saved for reproducibility only and never define the
primary nested score.

## Exact Five-Stream Simplex

All arithmetic is float64. With parent as the base and four delta streams, weights are non-negative,
sum to one, and have no intercept. The exact row-weighted SSE is solved by enumerating all 31
supports: 15 non-empty parent-active delta subsets, 15 parent-zero equality faces, and the parent
vertex. Solver, feasibility, direct-SSE, and tie tolerances are identical to the reviewed v009
protocol:

- `lstsq rcond=1e-12` only after a failed direct solve;
- original normal/KKT residual tolerance scaled by `1e-10`;
- raw feasibility tolerance `1e-10`, with no general clipping or renormalization;
- emitted weights must be non-negative and sum to one within `1e-12` before ranking;
- every feasible support is ranked by direct full-row SSE and must agree with the residual
  quadratic within `1e-8 * max(1, SSE)`;
- SSE ties use `64*eps(float64)` and then the tolerance-aware preference: maximum parent, minimum
  slow HMM, minimum front20, minimum ANCC, minimum best6.

Every fit records all 31 support outcomes and the selected face. Synthetic tests must cover exact
five-stream recovery, parent/parent-zero faces, duplicate-stream ties, singular systems, and outer
column isolation before the full run.

## Diagnostic Gate And Artifacts

`diagnostic_improvement_not_candidate` requires pooled nested OOF RMSE `<11.50`, at least three
folds better than v009, and all integrity checks. Every other outcome is `exhausted`. Neither
decision permits test inference or candidate promotion. The primary contract is exactly 3,783,989
rows, 773 wells, folds 1 through 5, finite five-stream inputs/predictions, six audits of 31 unique
supports, direct-SSE checks, and zero alignment/fold/progress/provenance violations.

V010 must not read raw test rows for inference, call HMM/formation test predictors, or create
`test_preds.npy`. The raw-test tree fingerprint is still recorded at start/end as a provenance
continuity check inherited from v009, not as authorization to use test data.

Before reading any target score, the run must create a new staging directory and atomically publish
a `status=running` `run.json`; completed or partial/staging output paths are never overwritten.
SIGTERM, KeyboardInterrupt, and every other `BaseException` atomically retain a `status=failed`
run with the error. A completed model directory is published only after critical input hashes and
the raw-test fingerprint match start/end, all report/model files are closed and hashed, and no
partial report companion remains.

The atomic `models/v010/` contract is `run.json`, `cv_scores.json`, `oof_preds.npy`,
`oof_rows.parquet`, `model.pkl`, `feature_list.txt`, and `importance.csv`, plus versioned summary,
fold, by-well, diagnostic, and artifact-audit reports. `test_preds.npy` is forbidden. V010 never
creates a submission CSV, notebook, Kaggle kernel, or API request, and this loop never submits to
Kaggle.
