# Exact FP4 decoding for the H100 V4.1 indexer

The SM90 indexer spends substantial time decoding FP4 keys before scoring.
The shared E2M1 helper now builds the exact FP32 encoding with integer
operations. An explicit `add.rn.f32(value, +0.0)` preserves the legacy
canonicalization of negative zero. The change preserves candidates, masks,
query groups, tiles, scales, BF16 rounding, dot accumulation, and TopK.
It adds no flag or workspace.

Component measurements use an otherwise idle H100 80GB HBM3, PyTorch
2.13.0+cu130, Triton 3.7.1, and runtime `18356102` as the baseline. Inputs
match the shapes observed in the 80/128 concurrent, 131072-token serving
traces, with synthetic keys, queries, random physical pages, and per-query
candidates. They are not a dump of model activations. Timings are medians
of three alternated baseline/candidate rounds, each with five groups of
20 CUDA-event measurements after warmup, without a profiler.

| Physical query rows / mode | Before, ms | After, ms | Time reduction |
|---|---:|---:|---:|
|120 / dense|1.947458|1.556322|20.08%|
|120 / source|4.100898|3.173043|22.63%|
|120 / compact|0.442187|0.343710|22.27%|
|192 / dense|3.117000|2.492154|20.05%|
|192 / source|6.504824|5.076197|21.96%|
|192 / compact|0.702354|0.547392|22.06%|

The target graph has three dense, one source, and five compact calls. The
weighted component sum at 192 rows falls from 19.368 to 15.290 ms. This
estimates an operator budget; it does not establish an end-to-end speedup
or the 100 tokens/s per-request target.

## Correctness and resource checks

All 16 E2M1 encodings and 256 E8M0 exponents pass storage-bitwise comparison
for decode and scaled BF16 conversion, including zero and nonfinite cases.
Twenty-five scoring cases cover 1/7/120/192 rows, visible lengths 131072
and 614400, ratios 1/2, dense/source/compact modes, empty and partial tiles,
random page layouts, interleaved requests, and independent query candidates.
Scores and candidate lengths are bitwise equal; the applicable TopK-512
index comparisons agree. Twenty graph replays change all input buffers and
also pass bitwise comparison. The same helper serves FP4-to-FP8 unpack;
the exhaustive helper checks cover its decode and scale domain.

Separate Nsight Compute measurements on dense 192 rows show warp
instructions falling from 1.9225 billion to 1.4999 billion, about 22%.
The baseline reaches 3.39% of peak HBM throughput and 7.64% L2 throughput;
instruction throughput, rather than saturated HBM bandwidth, dominates
the sampled SM throughput. The candidate retains 127 registers, 8 KiB
dynamic shared memory, and no spills. Hardware counters and profiled
durations are not substituted for the timings above.

Larger tiles were screened and rejected. A separate TileLang prototype
did not pass the score equality check and is not included. The original
Triton TTGIR feeds the first dot accumulator into the second dot, even
though the Python expression resembles two dots and an addition; future
rewrites must account for the compiled accumulation order.

Integrated multi-rank serving and long-context regression results are
reported separately once available. The component checks alone do not
guarantee autoregressive output or coding-task correctness.
