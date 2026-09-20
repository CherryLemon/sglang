"""DSpark draft metadata across CUDA graph replay and eager fallback."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.cuda_graph_buffer_registry import build_decode_registry
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def _graph_admission_runner(width):
    # Exercise the production admission predicate without constructing a model.
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.ragged_verify_mode = False
    runner.captured_req_width = width
    runner.require_mlp_tp_gather = True
    runner.require_mlp_sync = True
    runner.enable_pdmux = False
    runner.disable_padding = False
    runner.max_bs = 16
    runner.is_encoder_decoder = False
    runner.enable_two_batch_overlap = False
    runner.model_runner = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.DSPARK)
    return runner


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA graph replay")
class TestDsparkDraftGraphMetadata(unittest.TestCase):
    def test_replay_refreshes_counts_after_idle_and_eager_steps(self):
        # Block 5 and the extra-anchor variant; simulate both attention-TP
        # ranks, including replicated layouts that must not divide the count.
        for width in (5, 6):
            for tp_rank in (0, 1):
                for sequence_sharded in (False, True):
                    with self.subTest(
                        width=width, tp_rank=tp_rank, sequence_sharded=sequence_sharded
                    ):
                        self._check_replays(width, tp_rank, sequence_sharded)

    def _check_replays(self, width, tp_rank, sequence_sharded):
        device = torch.device("cuda")
        spec_info = SimpleNamespace(
            num_tokens_per_req=width,
            num_tokens_for_logprob_per_req=1,
        )
        proposer = DraftBlockProposer(
            draft_model=SimpleNamespace(sample_from_anchor=True),
            draft_model_runner=SimpleNamespace(
                device=device, decode_cuda_graph_runner=_graph_admission_runner(width)
            ),
            gamma=width,
            mask_token_id=0,
            draft_block_spec_info=spec_info,
            tp_sync=None,
            dp_moe_sync=True,
        )
        registry = build_decode_registry(
            device=device,
            max_bs=16,
            max_num_token=16 * width,
            seq_len_fill_value=1,
            cache_loc_dtype=torch.int64,
            enable_num_token_non_padded=True,
            require_gathered_buffer=True,
            require_mlp_tp_gather=True,
            dp_size=4,
            share_pool=False,
        )
        names = (
            "global_num_tokens_gpu",
            "global_num_tokens_for_logprob_gpu",
            "num_token_non_padded",
        )
        buffers = [registry.get_slot(name).buffer for name in names]
        addresses = [tensor.data_ptr() for tensor in buffers]
        observed = [torch.empty_like(tensor) for tensor in buffers]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for output, source in zip(observed, buffers):
                output.copy_(source)

        module = "sglang.srt.model_executor.forward_batch_info"
        with (
            patch(
                "sglang.srt.speculative.dspark_components.dspark_draft.enable_num_token_non_padded",
                return_value=True,
            ),
            patch(
                f"{module}.get_parallel",
                return_value=SimpleNamespace(attn_tp_size=2, attn_tp_rank=tp_rank),
            ),
            patch(
                f"{module}._attn_tp_sequence_sharded_predicate",
                return_value=sequence_sharded,
            ),
        ):
            for counts, admitted in [
                ([1, 3, 0, 2], True),
                ([0, 1, 4, 0], True),
                ([17, 2, 1, 0], True),  # Too large: eager fallback.
                ([2, 0, 1, 3], False),  # Scheduler disables replay.
                ([16, 1, 0, 4], True),
                ([1, 0, 0, 0], True),
            ]:
                bs = counts[0]
                empty = torch.empty(0, dtype=torch.int64, device=device)
                forward_batch = ForwardBatch(
                    forward_mode=ForwardMode.TARGET_VERIFY if bs else ForwardMode.IDLE,
                    batch_size=bs,
                    input_ids=torch.arange(bs * width, device=device),
                    req_pool_indices=empty,
                    seq_lens=empty,
                    out_cache_loc=empty,
                    seq_lens_sum=0,
                    spec_info=spec_info,
                )
                batch = SimpleNamespace(
                    global_num_tokens=counts,
                    global_num_tokens_for_logprob=[int(x > 0) for x in counts],
                    can_run_decode_cuda_graph=admitted,
                )
                can_run = proposer._prepare_forward_metadata(forward_batch, batch)
                expected_graph = admitted and max(counts) <= 16
                self.assertEqual(can_run, expected_graph)
                if not can_run:
                    self.assertEqual(
                        forward_batch.global_num_tokens_gpu.tolist(),
                        [x * width for x in counts],
                    )
                    self.assertEqual(
                        forward_batch.global_num_tokens_for_logprob_gpu.tolist(),
                        batch.global_num_tokens_for_logprob,
                    )
                    continue

                self.assertIsNone(forward_batch.global_num_tokens_gpu)
                self.assertIsNone(forward_batch.global_num_tokens_for_logprob_gpu)
                padded_bs = next(x for x in (1, 2, 4, 8, 16) if x >= max(counts))
                padded_tokens = padded_bs * width
                # Only count slots need refreshing in this metadata test. The
                # graph still reads the registry-owned persistent CUDA buffers.
                registry.fill_from(
                    SimpleNamespace(
                        num_token_non_padded=forward_batch.num_token_non_padded
                    ),
                    raw_bs=bs,
                    padded_bs=padded_bs,
                    raw_num_tokens=bs * width,
                    padded_num_tokens=padded_tokens,
                )
                graph.replay()
                for output in observed[:2]:
                    self.assertEqual(output.tolist(), [padded_tokens] * 4)
                per_rank = padded_tokens // 2
                expected_local = (
                    min(max(bs * width - tp_rank * per_rank, 0), per_rank)
                    if sequence_sharded
                    else bs * width
                )
                self.assertEqual(observed[2].item(), expected_local)
                self.assertEqual([tensor.data_ptr() for tensor in buffers], addresses)


if __name__ == "__main__":
    unittest.main()
