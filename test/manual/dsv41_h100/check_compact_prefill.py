"""Validate compact FP8 prefill against the unchanged dense-score/mask path."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch
from sglang.kernels.ops.attention.dsv4 import topk_transform_ragged_v2
from sglang.srt.layers.attention.dsv4.indexer import select_candidate_blocks


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-module")
    p.add_argument("--indexer-module")
    p.add_argument(
        "--reference-module",
        default=str(Path(__file__).with_name("reference_v9_fp8_indexer.py")),
    )
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.candidate_module:
        mod = load("compact_score_candidate", a.candidate_module)
    else:
        from sglang.kernels.ops.attention.dsv4 import sm90_fp4_indexer as mod
    if a.indexer_module:
        idxmod = load("compact_indexer_candidate", a.indexer_module)
    else:
        from sglang.srt.layers.attention.dsv4 import indexer as idxmod
    reference = (
        load("dense_score_reference", a.reference_module)
        if a.reference_module
        else None
    )
    torch.manual_seed(926)
    results = []
    for width in [
        0,
        1,
        7,
        8,
        9,
        63,
        64,
        65,
        511,
        512,
        513,
        16383,
        16384,
        16385,
        32768,
        131072,
        600000,
    ]:
        rows = 7
        q = torch.randn(rows, 32, 128, device="cuda").to(torch.float8_e4m3fn)
        w = torch.rand(rows, 32, device="cuda", dtype=torch.bfloat16)
        keys = torch.randn(width, 128, device="cuda").to(torch.float8_e4m3fn)
        lens = torch.tensor(
            [
                0,
                min(1, width),
                min(7, width),
                min(8, width),
                width // 2,
                max(0, width - 1),
                width,
            ],
            device="cuda",
            dtype=torch.int64,
        )
        dense = mod.fp8_index_logits_prefill(q, w, keys, lens)
        if reference is not None:
            torch.testing.assert_close(
                dense,
                reference.fp8_index_logits_prefill(q, w, keys, lens),
                rtol=0,
                atol=0,
            )
        blocks = idxmod.select_candidate_block_ids(dense, lens[:, None], 2048, 8)
        old_mask = select_candidate_blocks(dense, lens[:, None], 2048, 8)
        logical = (
            blocks[:, :, None].long() * 8 + torch.arange(8, device="cuda")
        ).reshape(rows, -1)
        valid = (logical >= 0) & (logical < width) & (logical < lens[:, None])
        compact = mod.fp8_index_logits_prefill(
            q, w, keys, lens, candidate_blocks=blocks
        )
        if width:
            expected = dense.gather(
                1, logical.clamp(0, dense.shape[1] - 1)
            ).masked_fill(~valid, -torch.inf)
            torch.testing.assert_close(compact, expected, rtol=0, atol=0)
            reproduced = torch.zeros_like(old_mask)
            safe = logical.clamp(0, dense.shape[1] - 1)
            # Scatter-add avoids an invalid sentinel clobbering a selected boundary bit.
            counts = torch.zeros_like(old_mask, dtype=torch.int32)
            counts.scatter_add_(
                1, safe, ((logical >= 0) & (logical < dense.shape[1])).int()
            )
            reproduced = counts > 0
            torch.testing.assert_close(reproduced, old_mask, rtol=0, atol=0)
            k = min(512, width)
            selected = torch.empty(rows, k, device="cuda", dtype=torch.int32)
            topk_transform_ragged_v2(
                compact,
                torch.full_like(lens, compact.shape[1], dtype=torch.int32),
                out_offsets=torch.zeros_like(lens, dtype=torch.int32),
                out_indices=selected,
            )
            sel = selected.long()
            ok = (sel >= 0) & (sel < compact.shape[1])
            vals = compact.gather(1, sel.clamp(0, compact.shape[1] - 1)).masked_fill(
                ~ok, -torch.inf
            )
            gold = dense.masked_fill(~old_mask, -torch.inf).topk(k, dim=-1).values
            torch.testing.assert_close(
                vals.sort(-1).values, gold.sort(-1).values, rtol=0, atol=0
            )
        else:
            assert compact.numel() == 0
        results.append(
            {"width": width, "passed": True, "compact_columns": compact.shape[1]}
        )
        print("PASS", width, flush=True)
    # Fixed graph addresses; vary both visible length and the candidate-ID content.
    for _ in range(2):
        mod.fp8_index_logits_prefill(q, w, keys, lens, candidate_blocks=blocks)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mod.fp8_index_logits_prefill(q, w, keys, lens, candidate_blocks=blocks)
    for length in [600000, 0, 1, 63, 16385, 600000]:
        lens.fill_(length)
        blocks.copy_(
            torch.arange(blocks.shape[1], device="cuda", dtype=blocks.dtype)[
                None, :
            ].expand_as(blocks)
        )
        graph.replay()
        torch.cuda.synchronize()
        dense = mod.fp8_index_logits_prefill(q, w, keys, lens)
        logical = (
            blocks[:, :, None].long() * 8 + torch.arange(8, device="cuda")
        ).reshape(rows, -1)
        expected = dense.gather(1, logical).masked_fill(
            logical >= lens[:, None], -torch.inf
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    # Preserve the torch selector's NaN/-inf and forced-tail semantics.
    scores = torch.full((3, 65), -torch.inf, device="cuda")
    scores[0, :] = 0
    scores[1, 8] = torch.nan
    lens = torch.tensor([65, 64, 0], device="cuda")
    b = idxmod.select_candidate_block_ids(scores, lens[:, None], 4, 8)
    mask = select_candidate_blocks(scores, lens[:, None], 4, 8)
    loc = (b[:, :, None].long() * 8 + torch.arange(8, device="cuda")).reshape(3, -1)
    counts = torch.zeros_like(mask, dtype=torch.int32)
    counts.scatter_add_(1, loc.clamp(0, 64), ((loc >= 0) & (loc < 65)).int())
    torch.testing.assert_close(counts > 0, mask, rtol=0, atol=0)
    # Empty query batch is also legal.
    assert mod.fp8_index_logits_prefill(q[:0], w[:0], keys, lens[:0]).shape == (
        0,
        600000,
    )
    Path(a.output).write_text(
        json.dumps(
            {
                "passed": True,
                "shapes": results,
                "graph_replays": 6,
                "nan_tail_selector": True,
                "pinned_dense_reference": a.reference_module is not None,
            },
            indent=2,
        )
    )
    print("COMPACT_PREFILL_PASS", flush=True)


if __name__ == "__main__":
    main()
