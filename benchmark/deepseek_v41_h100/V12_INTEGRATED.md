# Combined 128k-input decode experiment

The combined candidate reaches the measured 100 generation tokens/s per-request
threshold at concurrency 80. It does not reach it at concurrency 128. A retry
in the second 128-request round also exposed a graph-padding performance cliff;
that round is retained below.

Runtime commit: `77f7fdc0fcdadcf5ce0e13f239137bf9b165151d`. The subsequent
documentation commits do not change its Python sources. Both nodes were
verified against a 29-file manifest. The candidate combines the validated mHC
post, FP8 constexpr, exact FP4 decoder, and six-query FP4 reuse changes. It has
no MoE tactic override, profiler, or diagnostic route-capture hook.

## Workload and results

P and D each use eight healthy H100 80GB GPUs. P uses TP8/EP8; D uses attention
TP2/DP4 and EP8, DSpark block5 with static verify. The model is the formal
DeepSeek-V4.1-Flash revision
`dba1be0a40aa45a94ad051997016db3960a90277`. The context limit remains 655360.

Each native streaming request submits the same 131072-token coding input. P
caches 130816 prefix tokens; D radix caching is disabled, so the D requests
have independent KV allocations. Temperature is zero, output is capped at
8192 tokens, and every request ends naturally. Each concurrency has one
excluded warmup burst and two measured bursts. No feature ablations are used.

Generation speed is `(output_tokens - 1) / (last_content - first_content)`;
the timestamps refer to streamed content chunks. It excludes TTFT and is not
a per-token GPU timestamp measurement. The threshold requires every measured
request to average at least 100 tokens/s.

| Concurrency / measured round | Min generation tokens/s | Median generation tokens/s | Requests per DP group | Threshold |
|---|---:|---:|---|---|
| 80 / 1 | 101.4656 | 106.1679 | 20, 20, 20, 20 | Pass |
| 80 / 2 | 103.1874 | 106.7536 | 20, 20, 20, 20 | Pass |
| 128 / 1 | 74.0858 | 77.4424 | 32, 32, 32, 32 | Fail |
| 128 / 2 | 59.4154 | 63.9241 | 32, 33, 31, 32 | Fail |

Request metadata, sampled loads, and decode logs agree on the actual peak
concurrency and DP distribution. All 624 requests including warmups pass the
API, prefix-cache, natural-EOS, and explicit zero-retraction checks. Seven
sequential known-answer context checks also pass, including 600000 cold input
tokens; all eight P ranks have the expected prefill graph capture evidence.

The 80-request result has limited margin over 100 and establishes only this
workload and these two rounds. It is not a general serving SLA. Separately,
614 of 621 unique generated Python modules pass the coding fixture's semantic
checks; seven occurrences contain model algorithm errors (six RollbackDSU,
one scheduling implementation), with no protocol errors. These semantic
results are distinct from kernel numerical validation and known-answer tests.

## Why the second 128-request round slowed

The router records one decode transport failure followed by successful retry.
Retry selects a worker again. This provides a concrete explanation for a
128-request burst reaching DP loads 32/33/31/32 instead of 32/32/32/32; it is
not evidence that retries should be disabled.

The captured settings have graph buckets 32 then 40. The graph runner takes
the maximum DP request count and rounds it up for every group. Source and
load evidence therefore imply a 40-request graph for the imbalanced round:
local target/draft rows become 240/200 rather than 192/160, while gathered
rows become 960/800 rather than 768/640. This adds 25% padded rows and also
exits the new group6 and FP8 shape guards. No per-request kernel trace was
collected in this unprofiled round, so individual contributions are not
separately attributed.

Weighted acceptance length barely changes, 5.121762 to 5.123636; output tokens
increase only 1.06%. Median client decode time per verification increases
from 66.14 to 80.38ms and all four DP groups slow. Full-occupancy samples have
zero waiting/transfer/retraction queues and at most 33.65% KV usage.

Across the two rounds, 47 hardware queries covering all eight GPUs record
1830MHz SM clocks throughout, inactive thermal-slowdown flags, zero volatile
uncorrected ECC, and a maximum temperature of 54C. These approximately
2.23-second samples cannot exclude shorter events. Runtime logs and selected
cache timestamps show no round-time recompilation or autotuning.

Next work addresses this cliff with intermediate graph buckets and validated
neighboring kernel shapes. It retains legal retry/failover and the anomalous
round as regression evidence. TileLang indexer research remains separate
until its full numerical and performance checks pass.

## Evidence identities

The private experiment archive contains raw responses, loads, settings,
hardware samples, source hashes, startup and runtime logs, and autotune cache
snapshots. Compressed snapshots were verified again after download:

- D: `953309a061179171545b76127272ad2c8d979650687e7e5d5fe0e8fbddccd1bb`
- P: `ff51961122ec4a889b2a9f58ab60d56f4c6a25197c4d7598ade932eae12d02f6`

The earlier controller's health-response parsing error is retained separately;
it occurred before any workload submission. It is not counted as a model or
performance failure.
