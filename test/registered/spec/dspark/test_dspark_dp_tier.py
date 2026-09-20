import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_components.dspark_planner import (
    dp_global_verify_tier_num_tokens,
    local_verify_tier_num_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLocalVerifyTierNumTokens(CustomTestCase):
    def test_no_budget_returns_sentinel(self):
        self.assertEqual(
            local_verify_tier_num_tokens(
                bs=8,
                verify_token_budget=None,
                verify_num_draft_tokens=6,
                min_verify_len=1,
            ),
            -1,
        )

    def test_budget_adds_to_anchor_floor(self):
        self.assertEqual(
            local_verify_tier_num_tokens(
                bs=8,
                verify_token_budget=10,
                verify_num_draft_tokens=6,
                min_verify_len=1,
            ),
            18,
        )

    # Clamp/floor variants (verify-all clamp, min_verify_len floor, min=0) are
    # covered by the TestBusyIdleGraphKeyIdentity sweep bounds.


class TestDpGlobalVerifyTierNumTokens(CustomTestCase):
    def test_any_sentinel_pins_everyone(self):
        # The sweep never emits a -1 contribution, so this is the only guard
        # on "any rank without a budget pins everyone"; losing it forks graph
        # keys across DP ranks.
        self.assertIsNone(
            dp_global_verify_tier_num_tokens(global_tier_num_tokens=[100, -1, 50, 0])
        )


class TestDraftDpSyncMetadata(CustomTestCase):
    def _proposer(self, graph_runner=None, *, dp_moe_sync=True):
        proposer = DraftBlockProposer.__new__(DraftBlockProposer)
        proposer._dp_moe_sync = dp_moe_sync
        proposer._draft_block_spec_info = SimpleNamespace(
            num_tokens_per_req=6,
            num_tokens_for_logprob_per_req=1,
        )
        proposer.draft_model_runner = SimpleNamespace(
            device="cpu", decode_cuda_graph_runner=graph_runner
        )
        return proposer

    def _prepare(self, proposer, counts, *, local_bs=1, cuda=True, admitted=True):
        forward_batch = SimpleNamespace(
            input_ids=torch.arange(local_bs * 6),
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_gpu=None,
        )
        # A logprob width independent of the draft width must survive eager
        # fallback, even when graph replay synthesized padded token counts.
        batch = SimpleNamespace(
            global_num_tokens=counts,
            global_num_tokens_for_logprob=[int(x > 0) for x in counts],
            can_run_decode_cuda_graph=admitted,
        )
        module = "sglang.srt.speculative.dspark_components.dspark_draft"
        with (
            patch(f"{module}.enable_num_token_non_padded", return_value=True),
            patch(f"{module}.is_cuda", return_value=cuda),
        ):
            can_run_graph = proposer._prepare_forward_metadata(forward_batch, batch)
        return forward_batch, can_run_graph

    def test_eager_metadata_preserves_counts_and_logprob_width(self):
        forward_batch, can_run_graph = self._prepare(self._proposer(), [1, 3, 0, 2])
        self.assertFalse(can_run_graph)
        self.assertEqual(
            forward_batch.original_global_num_tokens_cpu,
            [1, 3, 0, 2],
        )
        self.assertEqual(forward_batch.global_num_tokens_cpu, [6, 18, 0, 12])
        self.assertEqual(forward_batch.global_num_tokens_gpu.tolist(), [6, 18, 0, 12])
        self.assertEqual(
            forward_batch.global_num_tokens_for_logprob_gpu.tolist(), [1, 1, 0, 1]
        )
        self.assertEqual(forward_batch.num_token_non_padded.item(), 6)
        self.assertEqual(forward_batch.num_token_non_padded.dtype, torch.int32)
        self.assertEqual(forward_batch.num_token_non_padded_cpu, 6)
        self.assertTrue(forward_batch.can_run_decode_cuda_graph)

    def test_graph_admission_reads_host_counts_before_skipping_uploads(self):
        admissions = []

        def admit(forward_batch):
            admissions.append(forward_batch.original_global_num_tokens_cpu)
            self.assertEqual(
                forward_batch.global_num_tokens_cpu,
                [6 * x for x in forward_batch.original_global_num_tokens_cpu],
            )
            return forward_batch.can_run_decode_cuda_graph

        proposer = self._proposer(SimpleNamespace(can_run_graph=admit))
        # Reuse one proposer across active/idle and graph/eager transitions.
        for counts, local_bs, admitted in [
            ([1, 3, 0, 2], 1, True),
            ([0, 1, 4, 0], 0, True),
            ([2, 0, 1, 3], 2, False),
            ([1, 0, 0, 0], 1, True),
        ]:
            with self.subTest(counts=counts, admitted=admitted):
                forward_batch, can_run_graph = self._prepare(
                    proposer, counts, local_bs=local_bs, admitted=admitted
                )
                self.assertEqual(can_run_graph, admitted)
                self.assertEqual(
                    forward_batch.num_token_non_padded.item(), local_bs * 6
                )
                self.assertEqual(forward_batch.num_token_non_padded_cpu, local_bs * 6)
                if admitted:
                    self.assertIsNone(forward_batch.global_num_tokens_gpu)
                    self.assertIsNone(forward_batch.global_num_tokens_for_logprob_gpu)
                else:
                    self.assertEqual(
                        forward_batch.global_num_tokens_gpu.tolist(),
                        [6 * x for x in counts],
                    )
        self.assertEqual(len(admissions), 4)

    def test_other_backends_keep_device_metadata_when_graph_admitted(self):
        forward_batch, can_run_graph = self._prepare(
            self._proposer(SimpleNamespace(can_run_graph=lambda _: True)),
            [0, 2, 1, 3],
            local_bs=0,
            cuda=False,
        )
        self.assertTrue(can_run_graph)
        self.assertEqual(forward_batch.global_num_tokens_gpu.tolist(), [0, 12, 6, 18])
        self.assertEqual(
            forward_batch.global_num_tokens_for_logprob_gpu.tolist(), [0, 1, 1, 1]
        )

    def test_dense_draft_keeps_local_token_count_and_graph_eligibility(self):
        forward_batch, can_run_graph = self._prepare(
            self._proposer(
                SimpleNamespace(can_run_graph=lambda fb: fb.can_run_decode_cuda_graph),
                dp_moe_sync=False,
            ),
            [1, 3, 0, 2],
        )
        self.assertTrue(can_run_graph)
        self.assertEqual(forward_batch.num_token_non_padded.item(), 6)
        self.assertIsNone(forward_batch.global_num_tokens_gpu)


class TestBusyIdleGraphKeyIdentity(CustomTestCase):
    def test_busy_and_idle_floors_agree_on_random_topologies(self):
        rng = random.Random(20260703)
        for _ in range(2000):
            verify_num_draft_tokens = rng.randint(2, 8)
            min_verify_len = rng.randint(0, verify_num_draft_tokens - 1)
            effective_min = max(min_verify_len, 1)
            num_ranks = rng.randint(1, 8)
            contributions = []
            num_reqs_per_rank = []
            for _ in range(num_ranks):
                if rng.random() < 0.3:
                    num_reqs_per_rank.append(0)
                    contributions.append(0)
                    continue
                bs = rng.randint(1, 512)
                budget = rng.randint(0, bs * verify_num_draft_tokens)
                num_reqs_per_rank.append(bs)
                contributions.append(
                    local_verify_tier_num_tokens(
                        bs=bs,
                        verify_token_budget=budget,
                        verify_num_draft_tokens=verify_num_draft_tokens,
                        min_verify_len=min_verify_len,
                    )
                )
            tier_num_tokens = dp_global_verify_tier_num_tokens(
                global_tier_num_tokens=contributions
            )
            global_num_reqs = max(num_reqs_per_rank)
            if tier_num_tokens is None:
                self.assertEqual(global_num_reqs, 0)
                continue

            self.assertGreaterEqual(tier_num_tokens, global_num_reqs * effective_min)
            self.assertLessEqual(
                tier_num_tokens, global_num_reqs * verify_num_draft_tokens
            )

            busy_floor = min(tier_num_tokens, global_num_reqs * verify_num_draft_tokens)
            self.assertEqual(busy_floor, tier_num_tokens)

            idle_lens_total = global_num_reqs
            idle_bucket_input = max(idle_lens_total, tier_num_tokens)
            self.assertEqual(idle_bucket_input, tier_num_tokens)


if __name__ == "__main__":
    unittest.main()
