#!/usr/bin/env python3
"""Measure a frozen JSONL workload against an OpenAI-compatible streaming API.

Each JSONL row is a complete chat-completions request body. This script does not
declare cache state or evaluate coding quality. API keys come from an environment
variable and are never included in the report. Token rates are estimates based on
server usage plus arrival times; streaming chunks are not counted as tokens.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
import threading
import time
import urllib.error
import urllib.request


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * percent / 100
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def run_request(args, body, barrier, index):
    body = dict(body)
    body["stream"] = True
    body["stream_options"] = {**body.get("stream_options", {}), "include_usage": True}
    if args.model:
        body["model"] = args.model
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(args.api_key_env)
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    # Internal endpoints must not accidentally inherit download proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    result = {"request_index": index, "ok": False}
    barrier.wait(timeout=60)
    started = time.monotonic()
    first_generated = first_visible = generation_finished = None
    fragments = {"content": [], "reasoning_content": [], "reasoning": []}
    characters = {"content": 0, "reasoning_content": 0, "reasoning": 0}
    finish_reasons = []
    usage = None
    stream_done = False
    try:
        with opener.open(request, timeout=args.timeout) as response:
            result["http_status"] = response.status
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    stream_done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise ValueError("API returned a streaming error")
                if event.get("usage"):
                    usage = event["usage"]
                now = time.monotonic()
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    generated = False
                    for key in characters:
                        value = delta.get(key)
                        if isinstance(value, str) and value:
                            characters[key] += len(value)
                            fragments[key].append(value)
                            generated = True
                            if key == "content" and first_visible is None:
                                first_visible = now
                    for call in delta.get("tool_calls", []):
                        function = call.get("function", {})
                        if function.get("name") or function.get("arguments"):
                            generated = True
                    if generated:
                        if first_generated is None:
                            first_generated = now
                    if choice.get("finish_reason") is not None:
                        finish_reasons.append(choice["finish_reason"])
                        generation_finished = now
        if not stream_done or not finish_reasons or first_generated is None:
            raise ValueError(
                "Stream incomplete or contains no generated content/tool call"
            )
        if not usage or not isinstance(usage.get("completion_tokens"), int):
            raise ValueError("Server did not report completion-token usage")
        if usage["completion_tokens"] <= 0:
            raise ValueError("Server reported zero completion tokens")
        result["ok"] = True
    except urllib.error.HTTPError as exc:
        result.update(http_status=exc.code, error=f"HTTP {exc.code}")
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    ended = time.monotonic()
    result.update(
        duration_seconds=ended - started,
        first_generated_chunk_seconds=(
            (first_generated - started) if first_generated else None
        ),
        first_visible_content_seconds=(
            (first_visible - started) if first_visible else None
        ),
        generation_finished_seconds=(
            (generation_finished - started) if generation_finished else None
        ),
        characters=characters,
        usage=usage,
        finish_reasons=finish_reasons,
        generated_text={k: "".join(v) for k, v in fragments.items()},
        content_sha256=hashlib.sha256(
            "".join(fragments["content"]).encode()
        ).hexdigest(),
    )
    # Approximate: a chunk can contain more than one token, and usage may include
    # reasoning tokens. Preserve timestamps/usage rather than count chunk events.
    if (
        usage
        and first_generated is not None
        and generation_finished is not None
        and generation_finished > first_generated
        and isinstance(usage.get("completion_tokens"), int)
    ):
        result["estimated_decode_tokens_per_second"] = (
            usage["completion_tokens"] - 1
        ) / (generation_finished - first_generated)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="URL ending in /v1")
    parser.add_argument("--requests-jsonl", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--label", default="unspecified_cache_state")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    rows = [
        json.loads(line)
        for line in pathlib.Path(args.requests_jsonl).read_text().splitlines()
        if line.strip()
    ]
    if len(rows) < args.concurrency:
        parser.error("provide at least one distinct request row per concurrent request")
    if any(not isinstance(row, dict) or not row.get("messages") for row in rows):
        parser.error("each row must be a chat request object with messages")
    if any(row.get("ignore_eos", False) for row in rows):
        parser.error("Natural coding workload must honor EOS")
    if any(row.get("n", 1) != 1 for row in rows):
        parser.error(
            "use one completion per request (n=1) for comparable latency metrics"
        )
    barrier = threading.Barrier(args.concurrency + 1)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(run_request, args, rows[i], barrier, i)
            for i in range(args.concurrency)
        ]
        started = time.monotonic()
        barrier.wait(timeout=60)
        results = [future.result() for future in futures]
        elapsed = time.monotonic() - started
    good = [result for result in results if result["ok"]]
    summary = {
        "label": args.label,
        "concurrency": args.concurrency,
        "succeeded": len(good),
        "failed": len(results) - len(good),
        "batch_wall_seconds": elapsed,
        "aggregate_completed_output_tokens_per_second": sum(
            result["usage"]["completion_tokens"] for result in good
        )
        / elapsed,
        "caveats": [
            "Single batch; repeat for stable distributions.",
            "First generated chunk is a TTFT proxy, not a per-token trace.",
            "Cache state is not inferred from the user-provided label.",
            "Decode rates use server token usage and stream arrival times.",
        ],
    }
    for metric in (
        "first_generated_chunk_seconds",
        "first_visible_content_seconds",
        "duration_seconds",
        "estimated_decode_tokens_per_second",
    ):
        values = [result[metric] for result in good if result.get(metric) is not None]
        summary[metric] = {"p50": percentile(values, 50), "p95": percentile(values, 95)}
    pathlib.Path(args.output).write_text(
        json.dumps(
            {"summary": summary, "requests": results}, indent=2, ensure_ascii=False
        )
        + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if len(good) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
