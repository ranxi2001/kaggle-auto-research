# ROGII v007 HMM Postmortem

Status: `exhausted`. This analysis uses only the completed canonical OOF artifacts. It does not
promote a candidate or authorize test inference.

## Failure Shape

- v006 / v007 pooled RMSE: `12.184563 / 21.596877`; MAE: `7.687 / 13.301`.
- Slow HMM improved `274 / 773` wells and `41.82%` of rows, but worsened 499 wells. Median per-well
  RMSE change was `+2.800`; p25/p75/p90 were `-1.897 / +12.729 / +26.320`.
- The worst 10/50 wells contributed `25.4% / 64.2%` of added SSE. Even removing the worst 100 wells
  with target knowledge leaves HMM at `13.645` versus the matching v006 `11.976`.
- Slow had 181 rate-boundary and 9 position-boundary wells, 186 in their union. Excluding them does
  not rescue the pure candidate.

The HMM-parent shift has standard deviation `21.82`, while the correction v006 actually needs has
standard deviation `12.15`; their Pearson correlation is only `0.306`. HMM shift bias is `-1.369`
while the needed correction bias is `+0.850`.

## Drift Mechanism

The error grows with distance into the suffix. In the first suffix decile, shift standard deviation
is `5.09`, correction correlation is `0.571`, and the best diagnostic shrink is `0.485`. In the last
decile these become `28.50`, `0.281`, and `0.160`; added MSE versus v006 grows from `0.8` to `560.1`.

The posterior is overconfident under repeated GR motifs. Empirical 1/2-sigma coverage is only
`53.5% / 73.9%`, and some wells with RMSE above 60 have mean posterior std near 3-4 and negligible
edge mass. Missing-only registration blocks are not the cause: only `118 / 180,548` blocks contain
no finite GR. Calibration clipping is also not material: one slope hits its lower bound, none hit
the upper bound; 198 scales hit the lower bound and none the upper bound.

Independent dense-lattice checks reproduce the forward-backward posterior within `2.2e-8`. The
state signs, emission anchors, backward recursion, and artifact alignment are correct. The design
failure is that v006 supplies a moving coordinate corridor but no center attraction. The transition
mathematically cancels center motion, allowing a plausible but aliased GR mode to replace the parent
path and drift for the remainder of the well.

## Ruled-Out Gates

Target-free single-variable well gates all tie or lose under outer-fold evaluation. A multivariable
well-level Ridge gate scores `12.1899` and improves only two folds. Row-level progress and posterior
std gates score `12.1817 / 12.1824`, use only `6.4% / 0.5%` of rows, and each degrades one fold; gains
below 0.003 are not material.

The target-leaky well oracle (`8.852`) confirms complementarity but is unusable. A diagnostic
continuous shrink learned on the other four folds has alpha `0.149-0.178`, improves all five folds,
and scores `11.643301`. Because this hypothesis was discovered after v007 scoring, it is recorded
only as a future separately predeclared idea. V008 switches the core to fold-safe formation spatial
surfaces instead.
