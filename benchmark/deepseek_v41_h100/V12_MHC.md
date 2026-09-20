# H100 mHC post with independent hidden tiles

The SM90 mHC post path now uses a TileLang token-by-hidden grid for contiguous
BF16 inputs with HC=4, H=5120, FP32 mixing coefficients, and 65–192 physical
token rows. Each CTA processes 1024 hidden elements. The original kernel
processed all five hidden tiles serially inside one CTA per token.

The existing small-batch Triton path, larger-batch TileLang fallback,
FlashInfer precedence, and other GPU architectures keep their dispatch.
The new path preserves the original four-channel FMA order, BF16 storage,
and PDL synchronization and trigger. It adds no global workspace.

The implementation is in
`python/sglang/kernels/ops/layernorm/mhc_post_split_h_tilelang.py`, with a
shape gate in `DeepseekV4DecoderLayer.hc_post`. This gate can also match
prefill rows; it is not restricted to decode by forward mode.

## Component results

Measurements use one idle H100 80GB HBM3, PyTorch 2.13.0+cu130,
Triton 3.7.1, and TileLang 0.1.12. The complete allocation wrapper is
captured in both CUDA graphs, with 24 calls per graph and nine alternated
timing rounds after warmup. Replays use the captured allocations and hot
data; these are component measurements, not serving latency.

| Physical rows | Original TileLang, µs | Existing Triton, µs | New TileLang, µs |
|---:|---:|---:|---:|
| 100 | 3.567 | 3.167 | 2.684 |
| 120 | 3.671 | 3.525 | 2.876 |
| 160 | 5.384 | 3.981 | 3.232 |
| 192 | 5.440 | 4.496 | 3.775 |

All 128 enabled integer row counts improve by 1.240–1.758× against the
original TileLang kernel. A separate bounded screen compared three TileLang
and four Triton configurations at six representative shapes; the selected
TileLang configuration won those measured cases. This does not establish a
general performance ordering between the languages.

## Correctness and resources

All 128 enabled shapes pass 30 changed-input seeds with bitwise BF16 storage
equality against the original implementation. Validation includes 5,183
graph replays, padded logical/physical row boundaries, finite-scale stress,
zero/cancellation/Inf/NaN cases, and preserved dispatch boundaries. The
independent FP32 FMA reference also agrees bitwise at the tested reference
points. The final formatted kernel is AST-identical to the GPU-tested source.

Generated code and separate Nsight Compute samples show 85→72 registers per
thread and 28→10 KiB dynamic shared memory per CTA. At 120 rows, the grid
changes from 120 CTAs to 600, and sampled active warps increase from 6.23%
to 25.35%. Nsight Compute replay durations are not used for the timing table.

## Serving impact remains to be measured

For balanced attention TP2/DP4 with DSPARK block size 5, 80 concurrent
requests produce target/draft row counts of 120/100; 128 requests produce
192/160. Counting 80 target and six draft post calls, the component timings
estimate only 0.069 ms and 0.146 ms saved per cycle, respectively. PDL wait
may inflate post duration in a serving trace and is not removable arithmetic
work attributable to this patch.

The patch has not yet passed integrated serving validation. It does not by
itself establish 100 tokens/s per request, nor does it change the model's
context limit. Combined endpoint measurements are reported separately.
