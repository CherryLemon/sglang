"""CPU contracts for context-bounded DeepSeek V4 prefill graph metadata."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.layers.attention.deepseek_v4_backend as backend_module
from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    BuildPageTablePositions,
)
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DSV4Metadata,
    DeepseekV4AttnBackend,
)
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSV4BoundedPrefillGraph(unittest.TestCase):
    def setUp(self):
        self.backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
        self.backend.MAX_SEQ_LEN_FOR_CAPTURE = 655360
        self.backend.low_ratios = ()
        self.bound = patch.object(
            backend_module, "_prefill_graph_max_seq_len", return_value=65536
        )
        self.bound.start()
        self.addCleanup(self.bound.stop)

    def test_admission_short_long_short_and_missing_cpu_evidence(self):
        for lengths, admitted in (
            ([33024], True),
            ([65536], True),
            ([65537], False),
            ([600000], False),
            ([655360], False),
            ([33024], True),
            ([512, 65537], False),
            ([512, 65536], True),
            ([], False),
        ):
            with self.subTest(lengths=lengths):
                batch = SimpleNamespace(seq_lens_cpu=torch.tensor(lengths))
                self.assertEqual(
                    self.backend.can_run_prefill_cuda_graph(batch), admitted
                )
        for lengths in (None, torch.empty(1, device="meta")):
            self.assertFalse(
                self.backend.can_run_prefill_cuda_graph(
                    SimpleNamespace(seq_lens_cpu=lengths)
                )
            )
        self.assertEqual(self.backend.MAX_SEQ_LEN_FOR_CAPTURE, 655360)

    def test_capture_and_replay_use_same_bounded_width(self):
        captured = DSV4Metadata(Mock(), None)
        live_metadata = DSV4Metadata(Mock(), None)
        self.backend._build_forward_metadata = Mock(
            side_effect=[captured, live_metadata]
        )
        capture_batch = SimpleNamespace(seq_lens_cpu=torch.tensor([256]))
        live_batch = SimpleNamespace(seq_lens_cpu=torch.tensor([65536]))
        static_batch = SimpleNamespace(seq_lens_cpu=torch.tensor([65536]))

        self.assertIs(
            self.backend.init_forward_metadata_for_breakable_cuda_graph_capture(
                capture_batch
            ),
            captured,
        )
        self.backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
            captured, live_batch, static_forward_batch=static_batch
        )
        calls = self.backend._build_forward_metadata.call_args_list
        self.assertEqual(
            [call.args[0] for call in calls], [capture_batch, static_batch]
        )
        for call in calls:
            self.assertEqual(call.kwargs["max_seq_len_override"], 65536)
            self.assertTrue(call.kwargs["use_prefill_cuda_graph"])
        captured.core_attn_metadata.refresh_for_breakable_cuda_graph_replay_.assert_called_once_with(
            live_metadata.core_attn_metadata
        )
        self.assertIs(self.backend.forward_metadata, captured)
        self.assertEqual(self.backend.MAX_SEQ_LEN_FOR_CAPTURE, 655360)

    def test_runner_admission_combines_context_and_token_bucket_bounds(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.model_runner = SimpleNamespace(attn_backend=self.backend)
        runner._is_full_backend = False
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = Backend.BREAKABLE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.FULL
        runner.capture_num_tokens = [256]
        runner.max_num_tokens = 256
        for context, tokens, admitted in (
            (32768, 256, True),
            (65536, 256, True),
            (65537, 256, False),
            (600000, 256, False),
            (32768, 256, True),
            (32768, 4096, False),
            (32768, 128, True),
            (32768, 127, False),
        ):
            with self.subTest(context=context, tokens=tokens):
                batch = SimpleNamespace(
                    batch_size=1,
                    input_embeds=None,
                    replace_embeds=None,
                    forward_mode=ForwardMode.EXTEND,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    global_num_tokens_cpu=None,
                    return_logprob=False,
                    input_ids=list(range(tokens)),
                    extend_prefix_lens_cpu=[context - tokens],
                    seq_lens_cpu=torch.tensor([context]),
                )
                self.assertEqual(runner.can_run_graph(batch), admitted)

    def test_direct_metadata_calls_refuse_out_of_bound_before_build(self):
        self.backend._build_forward_metadata = Mock()
        too_long = SimpleNamespace(seq_lens_cpu=torch.tensor([600000]))
        with self.assertRaisesRegex(ValueError, "capture exceeds"):
            self.backend.init_forward_metadata_for_breakable_cuda_graph_capture(
                too_long
            )
        with self.assertRaisesRegex(ValueError, "replay exceeds"):
            self.backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
                DSV4Metadata(Mock(), None), too_long
            )
        self.backend._build_forward_metadata.assert_not_called()

    def test_metadata_page_table_is_bounded_and_keeps_last_partial_page(self):
        # Exercise the real CPU metadata builder against full-capacity request
        # storage. A non-page-aligned bound still needs its last partial page.
        req_to_token = torch.arange(655360).reshape(1, -1)
        for bound in (65535, 65536):
            with self.subTest(bound=bound), patch.object(
                backend_module, "_prefill_graph_max_seq_len", return_value=bound
            ):
                width = self.backend._prefill_cuda_graph_metadata_max_seq_len()
                metadata = BuildPageTablePositions.execute(
                    req_to_token=req_to_token,
                    req_pool_indices_repeated=torch.tensor([0, 0]),
                    seq_lens_casual=torch.tensor([256, bound]),
                    max_seq_len=width,
                    page_size=256,
                    swa_window=128,
                )
                self.assertEqual(metadata.page_table.shape, (2, 256))
                torch.testing.assert_close(
                    metadata.page_table[-1], torch.arange(256, dtype=torch.int32)
                )
                self.assertEqual(req_to_token.shape[1], 655360)

    def test_unbounded_default_and_capacity_clamp(self):
        for bound, expected in ((None, 655360), (1048576, 655360)):
            with self.subTest(bound=bound), patch.object(
                backend_module, "_prefill_graph_max_seq_len", return_value=bound
            ):
                self.assertEqual(
                    self.backend._prefill_cuda_graph_metadata_max_seq_len(), expected
                )
        with patch.object(
            backend_module, "_prefill_graph_max_seq_len", return_value=0
        ), self.assertRaisesRegex(ValueError, "must be positive"):
            self.backend._prefill_cuda_graph_metadata_max_seq_len()

    def test_low_ratio_indexer_retains_partial_boundary_page(self):
        self.backend.page_size = 256
        self.backend.token_to_kv_pool = SimpleNamespace(
            get_index_k_page_size=lambda _ratio: 64
        )
        core = SimpleNamespace(
            page_table=torch.arange(2560, dtype=torch.int32).reshape(1, -1),
            seq_lens_casual=torch.tensor([65535]),
        )
        with patch.object(
            backend_module, "_prefill_graph_max_seq_len", return_value=65535
        ), patch.object(
            backend_module,
            "PagedIndexerMetadata",
            side_effect=lambda **kw: SimpleNamespace(**kw),
        ):
            for ratio in (1, 2):
                metadata = self.backend._low_ratio_prefill_indexer_metadata(core, ratio)
                pages = 256 * (4 // ratio)
                self.assertEqual(metadata.page_table.shape, (1, pages))
                self.assertEqual(metadata.page_table[0, -1].item(), pages - 1)


if __name__ == "__main__":
    unittest.main()
