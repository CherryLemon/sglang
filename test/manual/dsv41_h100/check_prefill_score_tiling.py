"""Check H100 scoring tile changes against the frozen pre-v10 dense reference.

Run alongside check_compact_prefill.py: this stresses shapes that dispatch to
wide tiles, dense/compact CUDA graphs, and exact selected-score equality. The
reference is intentionally frozen so a shared implementation bug cannot make
both sides of the comparison pass.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from sglang.kernels.ops.attention.dsv4 import topk_transform_ragged_v2
from sglang.srt.layers.attention.dsv4.indexer import select_candidate_block_ids


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-module")
    parser.add_argument(
        "--reference-module",
        default=str(Path(__file__).with_name("reference_v9_fp8_indexer.py")),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.candidate_module:
        candidate = load("tiled_score_candidate", args.candidate_module)
    else:
        from sglang.kernels.ops.attention.dsv4 import sm90_fp4_indexer as candidate
    reference = load("frozen_dense_score", args.reference_module)
    torch.manual_seed(1102)
    results = []
    # 65 rows activate the wider tiles while exercising ragged row counts.
    shapes = [
        (65, width)
        for width in (4095, 4096, 4097, 8191, 8192, 8193, 16383, 16384, 16385)
    ]
    shapes += [(7, 131071), (7, 299009), (7, 600000), (7, 655360)]
    for rows, width in shapes:
        q = torch.randn(rows, 32, 128, device="cuda").to(torch.float8_e4m3fn)
        weights = torch.rand(rows, 32, device="cuda", dtype=torch.bfloat16)
        keys = torch.randn(width, 128, device="cuda").to(torch.float8_e4m3fn)
        lens = torch.linspace(0, width, rows, device="cuda").to(torch.int32)
        lens[:6] = torch.tensor([0, 1, 63, 127, 255, 257], device="cuda")
        for tied in (False, True) if width == 16385 else (False,):
            if tied:
                q.zero_()
            expected = reference.fp8_index_logits_prefill(q, weights, keys, lens)
            actual = candidate.fp8_index_logits_prefill(q, weights, keys, lens)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            blocks = select_candidate_block_ids(actual, lens[:, None], 2048, 8)
            expected_blocks = select_candidate_block_ids(
                expected, lens[:, None], 2048, 8
            )
            torch.testing.assert_close(blocks, expected_blocks, rtol=0, atol=0)
            logical = (
                blocks[:, :, None].long() * 8 + torch.arange(8, device="cuda")
            ).flatten(1)
            compact = candidate.fp8_index_logits_prefill(
                q, weights, keys, lens, candidate_blocks=blocks
            )
            expected_compact = expected.gather(
                1, logical.clamp_max(expected.shape[1] - 1)
            )
            expected_compact.masked_fill_(logical >= lens[:, None], -torch.inf)
            torch.testing.assert_close(compact, expected_compact, rtol=0, atol=0)
            # The fused ragged selector has unspecified ordering and tie IDs.
            # Its selected score multiset must stay exact, including zero ties.
            for scores, gold, score_lens in (
                (actual, expected, lens),
                (compact, expected_compact, torch.full_like(lens, compact.shape[1])),
            ):
                out = torch.empty(rows, 512, dtype=torch.int32, device="cuda")
                gold_out = torch.empty_like(out)
                offsets = torch.zeros_like(lens, dtype=torch.int32)
                topk_transform_ragged_v2(
                    scores, score_lens, out_offsets=offsets, out_indices=out
                )
                topk_transform_ragged_v2(
                    gold, score_lens, out_offsets=offsets, out_indices=gold_out
                )
                selected = scores.gather(1, out.clamp_min(0).long()).masked_fill(
                    out < 0, -torch.inf
                )
                gold_selected = gold.gather(
                    1, gold_out.clamp_min(0).long()
                ).masked_fill(gold_out < 0, -torch.inf)
                torch.testing.assert_close(
                    selected.sort(-1).values,
                    gold_selected.sort(-1).values,
                    rtol=0,
                    atol=0,
                )
            results.append({"rows": rows, "width": width, "tied": tied, "passed": True})
            print("PASS", results[-1], flush=True)

    # Graph input addresses are fixed. Update queries, keys, weights, lengths,
    # and block IDs in-place so replay cannot accidentally specialize on data.
    rows, width = 65, 16385
    q = torch.randn(rows, 32, 128, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.rand(rows, 32, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(width, 128, device="cuda").to(torch.float8_e4m3fn)
    lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    blocks = torch.arange(2048, device="cuda", dtype=torch.int32)[None, :].repeat(
        rows, 1
    )
    for _ in range(2):
        candidate.fp8_index_logits_prefill(q, weights, keys, lens)
        candidate.fp8_index_logits_prefill(
            q, weights, keys, lens, candidate_blocks=blocks
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dense = candidate.fp8_index_logits_prefill(q, weights, keys, lens)
        compact = candidate.fp8_index_logits_prefill(
            q, weights, keys, lens, candidate_blocks=blocks
        )
    lengths = [0, 1, 63, 127, 128, 129, 255, 256, 257, 16384, width, 0, width]
    for i, length in enumerate(lengths):
        lens.fill_(length)
        q.copy_(torch.randn_like(q, dtype=torch.float32).to(q.dtype))
        weights.uniform_(-1, 1)
        keys.copy_(torch.randn_like(keys, dtype=torch.float32).to(keys.dtype))
        blocks.copy_(
            torch.randint(
                0, (width + 7) // 8 + 1, blocks.shape, device="cuda", dtype=blocks.dtype
            )
            .sort(-1)
            .values
        )
        graph.replay()
        expected = reference.fp8_index_logits_prefill(q, weights, keys, lens)
        torch.testing.assert_close(dense, expected, rtol=0, atol=0)
        logical = (
            blocks[:, :, None].long() * 8 + torch.arange(8, device="cuda")
        ).flatten(1)
        expected_compact = expected.gather(1, logical.clamp_max(expected.shape[1] - 1))
        expected_compact.masked_fill_(
            (logical >= length) | (logical >= width), -torch.inf
        )
        torch.testing.assert_close(compact, expected_compact, rtol=0, atol=0)
    torch.cuda.synchronize()
    Path(args.output).write_text(
        json.dumps(
            {
                "passed": True,
                "shapes": results,
                "graph_replays": len(lengths),
                "exact_selected_score_multisets": True,
            },
            indent=2,
        )
    )
    print("PREFILL_SCORE_TILING_PASS", flush=True)


if __name__ == "__main__":
    main()
