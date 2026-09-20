# DeepSeek V4.1 Flash: H100 PD optimization

The v9 checkpoint preserves a tested integration on top of SGLang
`da64c5cbb8cf6bfd39be19da43573fdfd484c43a`. It is the checkpoint before
further block-FP8 GEMM and compact-prefill optimization.

The latest whole-version optimization and natural-EOS coding measurements are
in [V11.md](V11.md). [V10.md](V10.md) documents the preceding compute/indexer
step; the sections below preserve the historical v9 checkpoint and configuration.

## Changes and provenance

- Opt-in heterogeneous PD: prefill attention TP8 and decode attention TP2 x DP4,
  global TP8 / EP8, with topology admission and C2 stride validation.
- DSpark attention-TP reductions and Engram idle-rank handling.
- Hopper indexer: invalid-tile skipping, direct request-to-token lookup,
  compact candidate scoring/finalization including TARGET_VERIFY, and safe
  initialization of invisible score tiles and invalid candidate indices.
- Hopper FP8 prefill scoring with the existing backend metadata interface.
- Semantic ports of [#39057](https://github.com/sgl-project/sglang/pull/39057)
  (`acd54da3c2259cdfe8d27d201c21664561695506`) and
  [#39086](https://github.com/sgl-project/sglang/pull/39086)
  (`91fe12f315c7a499efe5fdb05348bf955d779921`). The direct-lookup benefit of
  [#39979](https://github.com/sgl-project/sglang/pull/39979) overlaps the chosen
  #39057 implementation and is not applied twice.

At the v9 checkpoint, the runtime files correspond to bundle SHA256
`34fd2da133c817f94bbe241ace4e9e4f68d6849c7320bd41652e05c35e6b89be`;
one lab-machine comment is removed. File hashes are in `v9-manifest.json`.
This is an experimental fork checkpoint, not an upstream support claim.

## Validated configuration

Model: `deepseek-ai/DeepSeek-V4.1-Flash`, revision
`dba1be0a40aa45a94ad051997016db3960a90277`. Two servers, each 8 x H100 80GB.
Torch 2.13 + CUDA 13.0, Triton 3.7.1; image `lmsysorg/sglang:dev-dsv41`,
image configuration digest
`sha256:00b8215c42c516f8f4737ed2d38bd2794ef065484407ff8a637b6e251699d172`.

Both sides: context length 655360, token pool 1048576, max running 128,
mem fraction .80, EP8, MoE A2A backend `none`, Mooncake RDMA transfer,
DSpark block size 5, static verify, CUDA Graph max decode batch 16.
Prefill: attention TP8, chunk 4096. Decode: attention TP2 x DP4, DP LM head;
DSpark makes its effective chunk size 1024. Router is PD-aware and DP-aware.

Environment overrides used for the tested integration:

```sh
export SGLANG_OPT_DSV41_INDEXER_SKIP_INVALID_TILES=1
export SGLANG_OPT_DSV41_DEEPSELECT_CANDIDATE_TOPK=0
export SGLANG_DSPARK_ATTN_TP_REDUCE=1
export SGLANG_RAGGED_VERIFY_MODE=static
# Decode only; enables only the explicitly admitted PD topology pairs.
export SGLANG_DSV41_EXPERIMENTAL_DPA_PD=1
```

Set each server's host, ports, RDMA device list, GID and transfer settings for
its own environment. The last environment variable is decode-only; leave it
off on prefill. These values describe the tested setup, not universal tuning
recommendations.

## Validation already completed

- 52 GPU component checks: request-map lookup, score padding, zero lengths,
  compact candidate scoring/finalization, and fixed-buffer CUDA Graph replay.
- Functional single/16-request batches, mixed lengths, uneven DP activity,
  reasoning, tools, cancellation, and 16 known-answer coding assertions.
- DSpark output checks on all four DP ranks, serial and parallel.
- Exact cold 32768 / 131072 / 600000-token requests, cached_tokens=0,
  with correct values from the beginning, middle and end of the input.
- Model and parameter hashes checked on both nodes, including after profiling.

The existing GPU checks and their frozen v8 indexer reference are included:

```sh
PYTHONPATH=python python test/manual/dsv41_h100/check_v9_kernels.py
```

The reference comes from the preceding validated v8 integration of this
repository's FP4 indexer. It is a test fixture, not a serving implementation.
Controlled equality tests do not establish full model quality equivalence;
FP8 prefill changes query quantization and is not claimed bitwise equivalent
for arbitrary model inputs.

## Recorded whole-integration results

Same pair of servers and capacity, frozen approximately 32k inputs,
512 output tokens/request, temperature 0, ignore_eos=true; two versions
compared over ten warmed repetitions per concurrency. Pooled burst output
throughput is total output tokens / sum of batch wall times.

| Concurrency | v8 tok/s | v9 tok/s | Change |
|---|---:|---:|---:|
| 1 | 141.7 | 160.9 | +13.5% |
| 4 | 433.1 | 497.4 | +14.8% |
| 16 | 1287.3 | 1471.6 | +14.3% |

Cold 600k first-output-block latency: 324.488 s -> 113.093 s (one request
per version). Streaming blocks can contain multiple tokens, so this is not
an exact per-token ITL measurement. Single-request throughput had greater
run-to-run variance in v9. These results do not establish sustained-arrival
SLA capacity, benchmark-suite quality, or 16 concurrent 600k requests.

## Profiling findings that motivated v10

Fresh warmed 1/4/16-request Nsight Systems captures and a bounded late-600k
prefill capture identify:

- Decode block32 FP8 linear kernels: 39-48% of summed rank0 kernel time;
  shape-specific H100 configs were missing. Target small M6/M24 and draft
  shapes while preserving scale and CUDA Graph semantics.
- Late prefill FP8 index scoring: 55.8%; four candidate-consuming layers
  still score the full context before masking. Use compact candidate IDs and
  bounded reusable workspaces, preserving the FP8 rounding/selection contract.
- Grouped MoE GEMM: 23.5% at concurrency 16; revisit expert grouping and
  overlap after the first two improvements.

These are kernel-duration shares, not measured end-to-end speedups or a
hardware-counter diagnosis. Profiler timings were excluded from the throughput
comparison. Long prefill showed allocator retries but completed correctly;
reduce temporary allocation pressure without reducing required context capacity.
