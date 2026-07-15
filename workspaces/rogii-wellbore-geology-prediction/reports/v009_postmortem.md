# ROGII v009 Postmortem

Status: `promising_continue`. This is target-aware analysis performed only after the immutable v009
run completed. It motivates a future predeclared experiment but is not an independent validation
score for that future hypothesis.

## Verified Result

The cross-fitted four-stream simplex scored `11.633993822` versus v006 `12.184563279`, an absolute
improvement of `0.550569458` or `4.52%`. All five canonical folds improved. The final all-OOF
weights were `0.78994` parent, `0.15937` slow HMM, `0.01352` ANCC, and `0.03717` best6. Relative to
the earlier strict HMM-only shrink diagnostic (`11.643301125`), the two formation streams added only
`0.009307303` RMSE.

Independent artifact recomputation reproduced every pooled/fold score exactly and found zero
alignment, fold, provenance, non-finite, direct-SSE, partial-artifact, test, or submission issues.
The stable evidence is `reports/v009_artifact_audit.json`.

## Progress Shape

Suffix progress is computed within each well from its first to last natural suffix row. The table
shows the post-hoc coefficient obtained by regressing the remaining v009 residual on the fixed slow
HMM correction `hmm-parent` inside each decile.

| Suffix decile | v006 RMSE | v009 RMSE | Gain | HMM alpha from parent | Additional beta after v009 |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.3202 | 3.9009 | 0.4193 | 0.4848 | 0.3029 |
| 2 | 7.5365 | 6.9101 | 0.6264 | 0.2588 | 0.0864 |
| 3 | 9.2217 | 8.5003 | 0.7214 | 0.1869 | 0.0178 |
| 4 | 10.8779 | 9.8704 | 1.0075 | 0.1869 | 0.0201 |
| 5 | 12.2167 | 11.2060 | 1.0107 | 0.1813 | 0.0157 |
| 6 | 12.4492 | 11.7824 | 0.6668 | 0.1526 | -0.0128 |
| 7 | 13.3950 | 13.0760 | 0.3191 | 0.1439 | -0.0241 |
| 8 | 14.4375 | 14.0519 | 0.3856 | 0.1562 | -0.0160 |
| 9 | 15.4931 | 15.1356 | 0.3574 | 0.1623 | -0.0071 |
| 10 | 16.5417 | 16.0896 | 0.4521 | 0.1599 | -0.0109 |

The constant HMM weight is therefore close to the late-suffix optimum but materially underweights
the first two deciles. A fixed target-free front-load basis can represent the missing shape without
changing the HMM itself: for suffix progress `q` in `[0,1]`, define
`g(q)=max(0, 1-5q)` and `front=parent + g(q)*(hmm-parent)`. A non-negative simplex can add this basis
to the constant HMM stream, increasing correction near the boundary while converging exactly to the
parent after 20% progress.

## Boundary And Well Stability

V009 improved 442 of 773 wells and worsened 331; median per-well RMSE change was `-0.2832`, with
p10/p90 `-3.4967 / +3.5783`. The 186 inherited unresolved HMM-boundary wells improved from RMSE
`14.6094` to `13.4833`; their remaining HMM beta was still slightly positive (`0.0046`). The 587
non-boundary wells improved from `11.2581` to `10.9463`, with remaining beta `-0.0061`.

Boundary neutralization is therefore ruled out as the next global strategy. The sole recommended
v010 hypothesis is the fixed front-loaded HMM basis above, combined with the same two fold-safe
formation streams in a five-stream, fully nested simplex. Its score must be treated as another
hypothesis-confirming estimate because the basis shape was selected from this post-hoc analysis.
