# H100 FP8 decode configuration coverage

The v10 TP2 × DP4 trace exposed a configuration gap in the shared expert:
24 local verification tokens become 96 gathered tokens, where its
`N=576, K=5120` configuration reverted to the generic kernel. The resulting
36-block launch accounted for 327.4 ms of rank0's profiled 16-request run.
The trace and startup warnings also identified six uncovered projection
shapes. These configurations use the existing Hopper transpose/Split-K
implementation; no activation or weight quantization changes are needed.

| Weight shape `(N, K)` | Tuned row range | Original configuration resumes at |
|---|---|---|
| `(576, 5120)` | Extend existing range through 384 | 385 |
| `(512, 5120)` | 0–96 | 97 |
| `(1280, 5120)` | 0–96 | 97 |
| `(1536, 5120)` | 0–96 | 97 |
| `(4096, 1280)` | 0–32 | 33 |
| `(5120, 288)` | 0–96 | 97 |
| `(5120, 15360)` | 0–96 | 97 |

The explicit adjacent boundary entries matter because configuration lookup
uses the nearest row count. They prevent a small-batch configuration from
being selected for large prefill batches. Existing smaller-row entries in
`N=576, K=5120` are retained.

New transpose entries use `BLOCK_SIZE_N=64`. Generated PTX retains
`wgmma.mma_async` on E4M3 operands, as in the original kernel. A 32-column
candidate lowered to FP16 `mma.sync` and produced different dot-rounding
behavior; it was excluded. Split-K still changes the FP32 reduction order,
so arbitrary inputs are not promised to be bitwise equal.

## Component validation

H100 80GB HBM3, PyTorch 2.13.0+cu130, Triton 3.7.1. Weights and activations
were E4M3 with nonuniform power-of-two FP32 block scales. Candidate selection
checked 308 cases against independently dequantized FP32 matrix products
before timing. The final configurations passed 308 boundary/graph shapes and
924 CUDA graph replays with changed activation payloads and scales.

The existing compatibility tolerances are unchanged: FP32 `rtol=1e-4` and
`atol=reference_rms*3e-4`; BF16 `rtol=0.008` and the same absolute allowance.
Checks include empty inputs, every decode graph tier, gathered shared-expert
tiers through 384, and both sides of each new upper dispatch boundary.

Representative unprofiled medians from an isolated H100 follow. Each timing
includes the complete operator, its output, Split-K workspace and reduction:
64 calls per CUDA graph, seven repetitions alternating v10 and candidate,
after warming both graphs. All 73 changed dispatch cases improved; the
smallest measured ratio was 1.14×. Active sampled clocks were 1830 MHz,
maximum sampled temperature was 42°C, with no thermal flags or volatile
uncorrected ECC errors.

| `(M, N, K)` | v10 µs | Candidate µs | Ratio |
|---|---:|---:|---:|
| `(96, 576, 5120)` | 53.302 | 12.701 | 4.20× |
| `(384, 576, 5120)` | 53.493 | 31.270 | 1.71× |
| `(24, 512, 5120)` | 51.882 | 7.841 | 6.62× |
| `(24, 1280, 5120)` | 52.255 | 11.519 | 4.54× |
| `(24, 1536, 5120)` | 52.161 | 11.569 | 4.51× |
| `(24, 4096, 1280)` | 13.853 | 8.039 | 1.72× |
| `(96, 5120, 288)` | 6.512 | 5.648 | 1.15× |
| `(24, 5120, 15360)` | 225.904 | 59.130 | 3.82× |

A separate Nsight Compute capture of `(96,576,5120)` showed 36 → 216 blocks,
4.63% → 19.23% SM throughput, and 6.25% → 10.14% achieved occupancy. Clocks
were left unchanged. The profile explains the parallelism improvement;
its replay durations are not the timing numbers above.

Run the extended numerical/graph checker with:

```bash
python test/manual/dsv41_h100/check_block32_configs.py \
  --configs python/sglang/kernels/ops/quantization/configs \
  --shapes 576:5120,1280:5120,512:5120,1536:5120,4096:1280,5120:288,5120:15360 \
  --rows 0,1,2,3,4,5,6,7,8,10,12,14,16,18,24,30,31,32,33,36,42,48,60,64,65,72,80,84,95,96,97,120,128,144,168,192,240,256,288,336,383,384,385,512 \
  --output check.json
```

## Integration scope

No flags are added. Ship all seven H100 JSON files before process startup;
configuration lookup is cached. Other GPU names and block geometries retain
their existing dispatch. The largest added per-call Split-K allocation is
15 MiB for `(96,5120,15360)`; `(384,576,5120)` needs 6.75 MiB. CUDA graph pool
reuse and serving capacity must be checked in the complete integration.

Routed CUTLASS MXFP4/W4A16 MoE, metadata, model capacity, DSPARK, and request
handling are unchanged. These are component results, not endpoint throughput
or model-quality claims. The complete v11 integration owns the final
1/4/16-request and 600k-context acceptance tests; no endpoint ablation was run.
