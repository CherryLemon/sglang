# H100 V4.1 MoE tactic investigation

No MoE tactic override is included in this candidate. Configurations that
improved concentrated expert routing regressed substantially on uniform
routing; the alternative that passed the uniform check was effectively tied
with the original implementation.

The current backend is FlashInfer 0.6.18 CUTLASS SM90 W4A16: packed FP4/E8M0
expert weights, BF16 activations, EP8 with 48 local experts, hidden size 5120,
intermediate size 2304, top-k 6, and SwiGLU clamp 10. Attention TP2/DP4
gathers 480 or 768 rows for the two target workloads. The ordinary Python
`use_fused_finalize` option does not fuse finalization for this SM90
mixed-input implementation; the C++ path excludes it.

## Evidence and selection

All eight ranks were traced at both concurrency levels. FFN collective
arrival skew strongly tracks the rank's preceding grouped GEMMs: the last
arriver matches the largest FC1+FC2 duration in about 96%/97% of aligned
layer groups. Kernel residency includes waiting, and overlapping kernel
durations are not additive critical-path savings.

A separate diagnostic startup recorded actual global expert IDs and FP32
routing weights from five high-skew layers, eight stable verification steps,
and eight ranks at each row count. The 640 samples form 80 complete groups
with identical global routes across ranks. Clock calibration and independent
loads records place both captures inside actual full-occupancy plateaus.
The diagnostic hook and its captured kernels were removed before serving
performance validation.

The bounded replay selected 160 representative samples before timing and
held out the other 480. Five previously bitwise-equivalent candidates were
screened, followed by two finalists on held-out data: 1760 candidate
comparisons and 30 uniform-route comparisons. Activations and native packed
expert weights remained synthetic; routing tensors came from the model.
This measures the effect of real work distribution, not checkpoint output
quality. Every admitted eager and changed-input graph result was bitwise
equal to its baseline.

Startup caches were retained because their choices varied: the original
profiled startup used `[56,169]` at both buckets; the route-capture startup
used `[56,168]` at the 512 bucket for 480 rows, and `[56,169]` at 768.
Both baselines were compared for the affected shape and were bitwise equal.

| Finalist | Held-out layer maximum-rank comparison | Uniform guard |
|---|---|---|
| `[68,192]` | About 4.5% faster at 480 rows and 12.1% at 768 versus the original cache | Up to 44.8% slower; rejected |
| `[57,169]` | Essentially tied: about 0.01%/0.23% slower | Passed; no compelling improvement |

The first finalist was about 5.4% faster at 480 rows against the newer
cache. The second's roughly 0.95% improvement against that newer baseline
is a cache-selection difference, not a new algorithmic improvement.
These are serialized single-GPU replays of all rank assignments, summarized
by each layer/step's maximum rank. They are not simultaneous eight-GPU
wall times. Held-out steps from one repeated coding prompt also do not
establish generalization to independent agent contexts.

## Limits and next design requirement

The original FlashInfer output fails a strict independent PyTorch elementwise
reference check at a small number of cancellation cases. Those failures
remain recorded; a condition-number explanation does not replace the
acceptance rule. The tactic-only admission criterion here is unchanged
output bitwise relative to the original backend, not a claim that the
independent reference check passed.

FlashInfer currently selects tactic/profile integers on the host before
launch. Its device scheduler rebuilds the expert work map, but does not
switch tile/cluster/main-loop templates during graph replay. A route-aware
choice would need a new graph-safe conditional execution or adaptive kernel
design, with its overhead and numerical behavior verified. It is not an
existing serving flag and is not part of this candidate.
