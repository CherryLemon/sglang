"""Candidate-block scores and visibility masking for paged indexer logits."""

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_DEEPSELECT_INPUT_ALIGNMENT_BYTES = 1024


@triton.jit
def _maximum_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _candidate_scores_kernel(
    X,
    LENS,
    OUT,
    SCORES,
    WIDTH: tl.constexpr,
    NBLOCKS: tl.constexpr,
    STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    offsets = tl.arange(0, GROUP_PAD)
    cols = blocks[:, None] * GROUP + offsets[None, :]
    length = tl.load(LENS + row)
    in_bounds = (cols < WIDTH) & (offsets[None, :] < GROUP)
    values = tl.load(
        X + row * STRIDE + cols, in_bounds & (cols < length), other=-float("inf")
    ).to(tl.float32)
    tl.store(OUT + row * WIDTH + cols, values, in_bounds)
    scores = tl.reduce(values, axis=1, combine_fn=_maximum_with_nan)
    # `blocks < NBLOCKS` is required: when the score row is padded past the real block
    # count for the DeepSelect alignment, a row whose visible length reaches beyond the
    # captured width would otherwise stamp +inf onto a PADDING column. Top-K would then
    # select that out-of-range index, the publication kernel would reject it, and the row
    # would silently publish no candidates at all (a correctness loss, not a tie).
    scores = tl.where(
        (length > 0) & (blocks == (length - 1) // GROUP) & (blocks < NBLOCKS),
        float("inf"),
        scores,
    )
    tl.store(SCORES + row * SCORE_STRIDE + blocks, scores, blocks < SCORE_STRIDE)


@triton.jit
def _candidate_mask_kernel(
    X,
    LENS,
    KEEP,
    OUT,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    KEEP_STRIDE: tl.constexpr,
    KEEP_COL_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    visible = (cols < WIDTH) & (cols < tl.load(LENS + row))
    keep = tl.load(KEEP + row * KEEP_STRIDE + cols * KEEP_COL_STRIDE, visible, other=0)
    values = tl.load(X + row * STRIDE + cols, visible & keep, other=-float("inf")).to(
        tl.float32
    )
    tl.store(OUT + row * WIDTH + cols, values, cols < WIDTH)


@triton.jit
def _publish_candidate_mask_kernel(
    INDICES,
    VALUES,
    KEEP,
    WIDTH: tl.constexpr,
    NBLOCKS: tl.constexpr,
    GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    VALUE_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * TILE + tl.arange(0, TILE)
    selected = tl.load(INDICES + row * INDEX_STRIDE + i // GROUP, i < TOPK * GROUP, 0)
    score = tl.load(
        VALUES + row * VALUE_STRIDE + i // GROUP,
        i < TOPK * GROUP,
        -float("inf"),
    )
    # `selected` is int32 from the Top-K. A sentinel (INT_MAX), a negative wrap of an
    # oversized index, or any index outside [0, NBLOCKS) must never reach the store:
    # `selected * GROUP` can wrap a 32-bit index into a value that still satisfies
    # `cols < WIDTH` and would then write *before* this row's keep slice. Widen first,
    # then require the block to be in range.
    sel = selected.to(tl.int64)
    cols = sel * GROUP + (i % GROUP)
    in_range = (sel >= 0) & (sel < NBLOCKS) & (sel != 0x7FFFFFFF)
    # Top-K returns unique block indices: each output position has one writer.
    # A NaN score fails `score > -inf` and therefore never publishes.
    tl.store(
        KEEP + row * WIDTH + cols,
        score > -float("inf"),
        (i < TOPK * GROUP) & in_range & (cols < WIDTH),
    )


def candidate_block_logits(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    topk_blocks: int,
    block_size: int,
    published: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Select candidate blocks and publish their token-level visibility mask.

    A source masks the unread tail while reducing each block. A consumer masks
    visibility and the published candidates in one pass, without copying the
    capacity-sized logits before each masked_fill. With
    SGLANG_OPT_DSV41_DEEPSELECT_CANDIDATE_TOPK the block Top-K runs on the
    DeepSelect SM90 kernel, which can choose a different valid subset when finite
    block scores tie.
    """
    rows, width = logits.shape
    output = torch.empty((rows, width), dtype=torch.float32, device=logits.device)
    if published is not None:
        _candidate_mask_kernel[(rows, triton.cdiv(width, 4096))](
            logits,
            seq_lens,
            published,
            output,
            width,
            logits.stride(0),
            published.stride(0),
            published.stride(1),
            4096,
        )
        return output, None

    blocks = triton.cdiv(width, block_size)
    use_deepselect = envs.SGLANG_OPT_DSV41_DEEPSELECT_CANDIDATE_TOPK.get()
    if use_deepselect:
        if torch.cuda.get_device_capability(logits.device) != (9, 0):
            raise RuntimeError(
                "SGLANG_OPT_DSV41_DEEPSELECT_CANDIDATE_TOPK only supports SM90"
            )
        try:
            import deep_select
        except ImportError as exc:
            raise RuntimeError(
                "SGLANG_OPT_DSV41_DEEPSELECT_CANDIDATE_TOPK requires the "
                "deep_select package"
            ) from exc
    score_alignment = _DEEPSELECT_INPUT_ALIGNMENT_BYTES // torch.float32.itemsize
    score_stride = triton.cdiv(blocks, score_alignment) * score_alignment
    scores = torch.empty(
        (rows, score_stride if use_deepselect else blocks),
        dtype=torch.float32,
        device=logits.device,
    )
    group_pad = triton.next_power_of_2(block_size)
    tile = max(1, 1024 // group_pad)
    _candidate_scores_kernel[
        (rows, triton.cdiv(score_stride if use_deepselect else blocks, tile))
    ](
        logits,
        seq_lens,
        output,
        scores,
        width,
        blocks,
        logits.stride(0),
        scores.stride(0),
        block_size,
        group_pad,
        tile,
    )
    # Publication only needs membership; sorting the selected pairs is unused.
    selected = min(topk_blocks, blocks)
    if use_deepselect:
        # abort_when_nan_found=False is deliberate: the default (True) calls trap(),
        # which would take down the whole server CUDA context on one corrupt block
        # score. With False the kernel writes the documented idx_oob_fill_value
        # sentinel for that row; the publication kernel below rejects any index that
        # is not a real block, so a sentinel row publishes nothing instead of
        # wrapping into a writes-before-the-row mask.
        top_values, top_indices = deep_select.topk(
            scores,
            selected,
            indices_type=torch.int32,
            return_value=True,
            abort_when_nan_found=False,
        )
    else:
        top = scores.topk(selected, dim=-1, sorted=False)
        top_values, top_indices = top.values, top.indices
    keep = torch.zeros((rows, width), dtype=torch.bool, device=logits.device)
    _publish_candidate_mask_kernel[
        (rows, triton.cdiv(top_indices.shape[1] * block_size, 256))
    ](
        top_indices,
        top_values,
        keep,
        width,
        blocks,
        block_size,
        top_indices.shape[1],
        top_indices.stride(0),
        top_values.stride(0),
        256,
        num_warps=4,
    )
    return output, keep


@triton.jit
def _sort_candidate_blocks_kernel(
    BLOCKS,
    SCORES,
    LENS,
    SORTED,
    COUNTS,
    BLOCK_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    TOPK_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, TOPK_BLOCKS)
    blocks = tl.load(BLOCKS + row * BLOCK_STRIDE + offsets).to(tl.int32)
    safe_blocks = tl.maximum(blocks, 0)
    scores = tl.load(SCORES + row * SCORE_STRIDE + safe_blocks, mask=(blocks >= 0) & (blocks < NUM_BLOCKS), other=-float("inf"))
    valid = (blocks >= 0) & (blocks < NUM_BLOCKS) & (scores > -float("inf"))
    blocks = tl.where(valid, blocks, NUM_BLOCKS)
    blocks = tl.sort(blocks)
    tl.store(SORTED + row * TOPK_BLOCKS + offsets, blocks)

    length = tl.load(LENS + row)
    remaining = length - blocks * BLOCK_SIZE
    contribution = tl.minimum(tl.maximum(remaining, 0), BLOCK_SIZE)
    contribution = tl.where(blocks < NUM_BLOCKS, contribution, 0)
    tl.store(COUNTS + row, tl.sum(contribution, axis=0))


@triton.jit
def _finalize_candidate_topk_kernel(
    SELECTED,
    SCORES,
    LENS,
    REQ_TO_TOKEN,
    REQ,
    CANDIDATE_BLOCKS,
    PAGE_INDICES,
    RAW_INDICES,
    SELECTED_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    CANDIDATE_BLOCK_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    SOURCE_WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    USE_CANDIDATES: tl.constexpr,
    HAS_RAW: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, TOPK)
    selected = tl.load(SELECTED + row * SELECTED_STRIDE + offsets).to(tl.int64)
    length = tl.load(LENS + row).to(tl.int64)
    valid = (selected >= 0) & (selected < length) & (selected < SOURCE_WIDTH)
    score = tl.load(
        SCORES + row * SCORE_STRIDE + tl.maximum(selected, 0),
        mask=valid,
        other=-float("inf"),
    )
    valid &= score > -float("inf")
    selected = tl.where(valid, selected, SOURCE_WIDTH)
    selected = tl.sort(selected)
    valid = selected < length
    safe = tl.minimum(selected, SOURCE_WIDTH - 1)

    if USE_CANDIDATES:
        block_col = safe // CANDIDATE_BLOCK_SIZE
        within = safe % CANDIDATE_BLOCK_SIZE
        block = tl.load(
            CANDIDATE_BLOCKS + row * CANDIDATE_BLOCK_STRIDE + block_col,
            mask=valid,
            other=0,
        )
        logical = block * CANDIDATE_BLOCK_SIZE + within
        req = tl.load(REQ + row).to(tl.int64)
        slot = tl.load(
            REQ_TO_TOKEN + req * REQ_STRIDE + logical * RATIO,
            mask=valid,
            other=0,
        )
        slot = slot // RATIO
    else:
        logical = safe
        req = tl.load(REQ + row).to(tl.int64)
        slot = tl.load(
            REQ_TO_TOKEN + req * REQ_STRIDE + logical * RATIO,
            mask=valid,
            other=0,
        )
        slot = slot // RATIO

    tl.store(
        PAGE_INDICES + row * OUTPUT_STRIDE + offsets,
        tl.where(valid, slot, -1),
    )
    if HAS_RAW:
        tl.store(
            RAW_INDICES + row * OUTPUT_STRIDE + offsets,
            tl.where(valid, logical, -1),
        )


def candidate_block_state(
    block_scores: torch.Tensor,
    block_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    topk_blocks: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Publish sorted candidate blocks, valid position counts, and a shared plan."""
    from sglang.kernels.ops.attention.dsv4.topk import (
        plan_topk_v2,
        topk_transform_paged_v2,
    )

    rows, num_blocks = block_scores.shape
    assert triton.next_power_of_2(topk_blocks) == topk_blocks
    assert block_lens.shape == seq_lens.shape == (rows,)

    selected_blocks = torch.empty(
        (rows, topk_blocks), dtype=torch.int32, device=block_scores.device
    )
    topk_transform_paged_v2(
        block_scores,
        block_lens,
        None,
        selected_blocks,
        1,
        plan_topk_v2(block_lens),
    )

    sorted_blocks = torch.empty_like(selected_blocks)
    counts = torch.empty(rows, dtype=torch.int32, device=block_scores.device)
    _sort_candidate_blocks_kernel[(rows,)](
        selected_blocks,
        block_scores,
        seq_lens,
        sorted_blocks,
        counts,
        selected_blocks.stride(0),
        block_scores.stride(0),
        topk_blocks,
        block_size,
        num_blocks,
        num_warps=8,
    )
    return sorted_blocks, counts, plan_topk_v2(counts)


def finalize_candidate_topk(
    selected: torch.Tensor,
    scores: torch.Tensor,
    score_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    req: torch.Tensor,
    page_indices: torch.Tensor,
    raw_indices: torch.Tensor | None,
    *,
    ratio: int,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 1,
) -> None:
    """Finalize sorted sparse-attention slots without PyTorch elementwise launches."""
    use_candidates = candidate_blocks is not None
    assert triton.next_power_of_2(page_indices.shape[1]) == page_indices.shape[1]
    source_width = scores.shape[1]
    _finalize_candidate_topk_kernel[(selected.shape[0],)](
        selected,
        scores,
        score_lens,
        req_to_token,
        req,
        candidate_blocks if use_candidates else req_to_token,
        page_indices,
        raw_indices if raw_indices is not None else page_indices,
        selected.stride(0),
        scores.stride(0),
        req_to_token.stride(0),
        candidate_blocks.stride(0) if use_candidates else 0,
        page_indices.stride(0),
        page_indices.shape[1],
        source_width,
        ratio,
        candidate_block_size,
        use_candidates,
        raw_indices is not None,
        num_warps=8,
    )
