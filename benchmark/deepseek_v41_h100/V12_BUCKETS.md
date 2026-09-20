# Cover neighboring DP graph buckets

The previous combined test exposed a padding cliff: one legal decode retry
changed a 128-request burst from 32/32/32/32 to 32/33/31/32. With graph buckets
32 then 40, every DP group used the larger padded shape and exited several
optimized shape guards. The measured evidence is in [V12_INTEGRATED.md](V12_INTEGRATED.md).

This follow-up admits verified neighboring shapes and adds explicit decode
graph buckets 34, 36, and 38 in the experiment launcher. It preserves retry,
model precision, context length, TP2/DP4/EP8, and DSpark block5. It does not
change any kernel's arithmetic or launch geometry.

```text
--cuda-graph-max-bs-decode 96
--cuda-graph-bs-decode 1 2 3 4 5 6 7 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 44 48 52 56 60 64 72 80 88 96
```

## Dispatch changes and component validation

- The six-query FP4 indexer now accepts B120 through B240 in steps of 12,
  covering even local request batches 20 through 40. Existing semantic
  target-verify, head-count, candidate-block, and request-identity guards
  remain. Other shapes retain the previous fallback.
- FP8 constexpr dispatch covers those batches' target rows `6 * bs`, draft
  rows `5 * bs`, and gathered shared-expert rows `4 * M`. All original dtype,
  layout, block32, complete-configuration, and SplitK guards remain. This is
  a finite shape list, not every M in an interval.
- The SM90 BF16 mHC split-H path extends from M65–192 to M65–240. M64 and below
  retain Triton; M241 and above retain the previous fallback.

The indexer extension passes 90 cases and 360 changed-input graph replays,
including 131072 and 614400 visible tokens, ratios 1/2, dense/source outputs,
full six-row padding groups, tile boundaries, different-request defense,
and partial-group fallback. Scores and candidate lengths match bitwise;
TopK comparisons on identical selector inputs agree. This component check
does not separately exercise the unchanged production selector pipeline.
Actual wrapper timing at the nine new shapes decreases 39.85–40.98% versus
their previous fallback; every compiled variant has zero spills and 8KiB
shared memory. These percentages are not end-to-end request improvements.

FP8 passes all 80 full shapes with 20 seeds and both BF16/FP32 outputs,
184 fallback cases, 3360 BF16 graph replays, and 400 additional FP32 changed
input graph replays. Quantization and SplitK reduction are included. Compiled
variants retain E4M3 WGMMA, 128 registers, 12KiB shared memory, and zero spills.
Full-call component speedups range from 1.0143x to 1.0922x. Small high-M output
projection gains were repeated without tuning, including M204 at
1.0160–1.0178x across three longer measurements.

mHC passes every added integer M193–240 with 30 seeds, graph replay, padding,
and cancellation/nonfinite stress. Component speedups are 1.1769–1.4293x.
No persistent tensor or workspace is added for a fixed M. Additional graph
and compiler-cache memory must still be checked in full-model startup.

Runtime commit `7a882e727d09ed7b896b44227b80b4e0b0b7481d` incorporates these
changes. The 29-file deployment bundle SHA256 is
`1e8639c179c3f6183fbc92f598a5dcdb18cf0db179e1cfa0cba898e24d923f7e`;
its rollback is the prior combined runtime `77f7fdc0`.

Combined unprofiled 80/128-request validation completed. The two formal
80-request bursts had minimum per-request generation rates of 103.004 and
102.816 tokens/s; both reached real running80. The two 128-request bursts
reached running128 with balanced 32/32/32/32 occupancy, but their minimum
rates were 76.104 and 75.240 tokens/s. This does not establish a 128-request
100 tokens/s result. The earlier imbalanced round remains part of the record.

All 624 requests including warmup passed API, cache-count, natural-EOS, and
explicit zero-retraction checks. The generated-code checker passed 618 of
624 distinct programs; six RollbackDSU semantic errors are retained separately
from runtime correctness. Seven known-answer boundary requests, including
cold600k, and all-eight-rank P graph capture passed.

A separate 128-request direct-PD correctness diagnostic held the controlled
32/33/31/32 distribution and completed every request successfully. All 128
generated programs passed. Capture, target graph logs, and verified replay
selection jointly support target graph34; this is not an Nsight kernel trace
or a normal-router performance measurement. The final checkpoint and the
requested 1/4/16/32/64/80 sweep are recorded in [V12.md](V12.md).
