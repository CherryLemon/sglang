"""Whole-version metrics using actual completion tokens and retained rounds."""

import argparse
import collections
import json
import pathlib
import random
import statistics


def pct(a, p):
    a = sorted(a)
    f = (len(a) - 1) * p
    i = int(f)
    return a[i] + (a[min(i + 1, len(a) - 1)] - a[i]) * (f - i)


def read_arm(root):
    assert json.loads((root / "status.json").read_text())["phase"] == "passed"
    result = {}
    for part, first in [("legacy", 1), ("natural-coding", 2)]:
        result[part] = {}
        for c in [1, 4, 16]:
            docs = [
                json.loads((root / part / f"c{c}-r{i}.json").read_text())
                for i in range(first, first + 10)
            ]
            rows = [r for d in docs for r in d["requests"]]
            assert len(rows) == 10 * c and all(r["ok"] for r in rows)
            hashes = {d["input_sha256"] for d in docs}
            assert len(hashes) == 1
            if part == "legacy":
                assert all(r["usage"]["completion_tokens"] == 512 for r in rows)
            else:
                assert all(
                    r["finish_reasons"] == ["stop"] for r in rows
                ), "Natural coding must complete without truncation"
            rounds = [
                {
                    "tokens": sum(
                        r["usage"]["completion_tokens"] for r in d["requests"]
                    ),
                    "seconds": d["summary"]["batch_wall_seconds"],
                }
                for d in docs
            ]
            ttft = [r["first_generated_chunk_seconds"] for r in rows]
            duration = [r["duration_seconds"] for r in rows]
            decode = [r["estimated_decode_tokens_per_second"] for r in rows]
            result[part][str(c)] = {
                "input_sha256": next(iter(hashes)),
                "measured_rounds": 10,
                "requests": len(rows),
                "rounds": rounds,
                "pooled_output_tps": sum(x["tokens"] for x in rounds)
                / sum(x["seconds"] for x in rounds),
                "ttft_proxy_p50_s": statistics.median(ttft),
                "ttft_proxy_p95_s": pct(ttft, 0.95),
                "latency_p50_s": statistics.median(duration),
                "latency_p95_s": pct(duration, 0.95),
                "estimated_decode_tps_p50": statistics.median(decode),
                "prompt_token_counts": sorted(
                    {r["usage"]["prompt_tokens"] for r in rows}
                ),
                "output_token_counts": dict(
                    sorted(
                        collections.Counter(
                            r["usage"]["completion_tokens"] for r in rows
                        ).items()
                    )
                ),
                "output_hashes": (
                    dict(collections.Counter(r["content_sha256"] for r in rows))
                    if part == "natural-coding"
                    else None
                ),
                "finish_reasons": dict(
                    collections.Counter(v for r in rows for v in r["finish_reasons"])
                ),
            }
    return result


def rate(rows):
    return sum(x["tokens"] for x in rows) / sum(x["seconds"] for x in rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=pathlib.Path, required=True)
    p.add_argument("--candidate", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    a = p.parse_args()
    arms = {"baseline": read_arm(a.baseline), "candidate": read_arm(a.candidate)}
    changes = {}
    rng = random.Random(1141)
    for part in ["legacy", "natural-coding"]:
        changes[part] = {}
        for c in ["1", "4", "16"]:
            b, t = [arms[k][part][c] for k in ["baseline", "candidate"]]
            assert b["input_sha256"] == t["input_sha256"]
            draws = [
                100
                * (
                    rate(rng.choices(t["rounds"], k=10))
                    / rate(rng.choices(b["rounds"], k=10))
                    - 1
                )
                for _ in range(10000)
            ]
            changes[part][c] = {
                "throughput_gain_pct": 100
                * (t["pooled_output_tps"] / b["pooled_output_tps"] - 1),
                "bootstrap_round_resampling_95_interval_pct": [
                    pct(draws, 0.025),
                    pct(draws, 0.975),
                ],
                "ttft_proxy_p50_change_pct": 100
                * (t["ttft_proxy_p50_s"] / b["ttft_proxy_p50_s"] - 1),
                "latency_p50_change_pct": 100
                * (t["latency_p50_s"] / b["latency_p50_s"] - 1),
            }
    result = {
        **arms,
        "change": changes,
        "caveats": [
            "Whole-version comparison. No per-feature ablation or attribution.",
            "Natural coding repeats one weighted-interval task with a shared 33005-token reference, not a diverse multi-task benchmark.",
            "Code implementations vary; semantic checks are independent, tokens are actual reported completion counts, EOS is honored.",
            "Legacy ignore_eos fixed512 includes output after natural EOS and has continuation/acceptance instability in both versions.",
            "Two warmup rounds excluded for natural coding; one warmup excluded for legacy.",
            "Streaming first chunk is a TTFT proxy; estimated decode rate is not a per-token trace.",
            "Round resampling does not remove run-order or system-state confounders; one full run per arm.",
            "Burst concurrency does not establish steady-arrival SLA capacity or simultaneous 600k capacity.",
        ],
    }
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(changes, indent=2))


if __name__ == "__main__":
    main()
