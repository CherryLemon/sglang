# H100 FP8 projections for concurrent V4.1 decoding

This note records the configuration expansion and a later fixed-shape
code-generation specialization. All measurements below are component results;
combined serving validation is separate.

Four existing block32 FP8 projection configurations now cover the larger
physical batches produced by TP2 / DP4 with DSPARK. Small-batch dispatch and
larger generic fallbacks are preserved. This changes launch configurations,
not the model weights, activation quantizer, arithmetic dtype or MoE backend.

Base runtime: `f5223ce7f132171ef644c9cd1aadc02a39602807`.
Integrated configuration commits: `950b5b0fb70e6ca2bcb0dda7a7e76e8b29f12942`
and `183561020dff4c56e34d258f88b4b6b0d9404c93`.

All new configurations use BK32, BN64, GROUP_M1, four warps, three stages,
and SWAP_AB. The changed physical M ranges are:

| Projection (N,K) | M | BM / SplitK |
|---|---:|---|
| wqkv_a (1792,5120) |81–384 / 385–768|64/4 / 128/2|
| wq_b (16384,1280) |45–384 / 385–768|64/1 / 128/1|
| wo_b (5120,4096) |45–96 / 97–320 / 321–384 / 385–768|32/4 / 64/2 / 64/1 / 128/1|
| shared gate/up (576,5120) |385–1536 / 1537–3072|64/4 / 128/2|

Adjacent JSON keys constrain nearest-key selection at each boundary. The
configuration files are in `python/sglang/kernels/ops/quantization/configs/`.
Deploy before process startup: the lookup has an in-process cache, and existing
CUDA graphs retain their captured kernels.

## Component measurements

The measurement uses one otherwise idle H100 80GB HBM3 with PyTorch2.13.0+cu130,
Triton3.7.1 and FlashInfer0.6.18. It calls the production BF16-to-UE8M0 group32
activation quantizer and FP8 linear operation with block32 E4M3 weights.
Timing includes quantization, GEMM, output allocation, partial workspace and
reduction. Each graph contains16 calls;25 eager warmups precede graph warmup
and nine alternated timing rounds.

The first expansion screened three fixed candidates on23 representative
shapes;128 changed verification points improved, with minimum speed ratio1.029.
The tail expansion screened four fixed candidates on12 representative shapes;
all67 changed verification points improved by at least1.2168. These are bounded
component searches, not endpoint ablations.

Representative tail results versus the previous configuration:

| M,N,K | Before, µs | After, µs | Speed ratio |
|---|---:|---:|---:|
|528,1792,5120|112.182|89.150|1.258|
|528,16384,1280|186.502|117.972|1.581|
|528,5120,4096|227.818|179.920|1.266|
|2112,576,5120|155.140|111.314|1.394|

Use the actual padded graph shape when applying these results. For example,
85 requests per DP group pad to88, producing target M528 and gathered shared
M2112, rather than unpadded M510/2040. These single-GPU, repeated-input,
hot-cache timings do not predict whole-model speedup or serving capacity.

## Numerical and graph validation

The two expansions were checked separately, retaining each original result:

| Check | First expansion | Tail expansion |
|---|---:|---:|
| Numerical shapes, including preserved/fallback boundaries |175|129|
| Main graph replays, including changed inputs |1400|1032|
| Logical/physical padding cases |43|28|
| Padding graph replays |516|336|
| Wide-scale stress cases |8|8|

Each replay agrees bitwise with its corresponding eager execution. Fixed
configuration padding checks preserve the valid quantized prefix and valid
output bitwise; comparisons across accumulation configurations use the original
numerical tolerances. FP32 baseline compatibility uses rtol1e-4 and
atol=reference RMS×3e-4; BF16 uses rtol0.008 with the same atol.
Independent dequantized-FP32 error must remain within the pre-existing v11
baseline-relative bounds: max error≤baseline×1.05+RMS×2e-5 and
RMSE≤baseline×1.05+RMS×5e-6. An initial additional strict reference check failed
the baseline as well and was stopped before timing; its evidence was retained.
The original compatibility criteria were not relaxed.

Generated Hopper code uses E4M3 WGMMA. No new FP16 MMA path was introduced.
The largest explicit SplitK partial remains13.5MiB per call. This is not the
memory usage of the complete model or its collection of CUDA graphs.

Complete serving validation and capacity measurements are separate from these
component results. In particular, graph/kernel checks do not guarantee identical
autoregressive text across batch shapes or correctness on arbitrary coding tasks.

## Fixed-shape integer arithmetic

The later specialization in `fp8_hopper_static.py` makes M/N/K and quantization
group sizes compile-time constants for 16 exact concurrent-decode shapes.
It preserves the current tile and SplitK configuration, each K32 E4M3 dot,
scale multiplication order, FP32 accumulation, activation quantizer, and
SplitK reduction. The dispatch also checks the complete expected configuration;
an altered configuration falls back to the original implementation.

| N,K | Physical M | SplitK |
|---|---|---:|
| 1792,5120 | 100,120,160,192 | 4 |
| 16384,1280 | 100,120,160,192 | 1 |
| 5120,4096 | 100,120,160,192 | 2 |
| 576,5120 | 400,480,640,768 | 4 |

The whole production call, including UE8M0 activation quantization and any
SplitK reduction, improves by 1.043–1.091× over runtime `18356102` in these
16 shapes. For example, M120/N16384/K1280 changes from 35.614 to 32.642 µs,
and M192/N5120/K4096 from 82.302 to 78.342 µs. Two additional candidates
with ordered loop unrolling were screened and rejected on performance.

Each shape passes 20 changed-input seeds for both BF16 and FP32 outputs with
storage-bitwise equality to the original kernel, in addition to the numerical
criteria above. Validation includes 672 graph replays with changed inputs,
scales and padding, and 65 neighboring or fallback shapes. These checks cover
the actual dispatch wrapper. The new kernel adds no global workspace.

Generated PTX retains E4M3 WGMMA. Static div/rem instructions drop from 27 to
two; 26 of the original instructions were in CTA setup and only one was in the
K loop. This is not a claim of 25 fewer instructions per K iteration. Both
versions allocate 128 registers per thread in hardware, use 12 KiB dynamic
shared memory, and have no spills in the sampled shapes.

Adding the isolated call-time differences across the covered target and draft
layers estimates 0.399 ms saved per cycle at 80 concurrent requests, and
0.558 ms at 128. Cache behavior, overlap and other operators can change the
integrated gain. These estimates do not establish a 100 tokens/s serving SLA.
