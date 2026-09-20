"""CPU contracts for inserts through a chunked prefix, including resumed walks."""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    FreeDeviceKV,
    FreeDeviceKVFullOnly,
)
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_core(page_size=1, is_bigram=False):
    params = CacheInitParams(False, None, None, page_size, is_eagle=is_bigram)
    component = FullComponent(SimpleNamespace(enable_session_radix_cache=False), params)
    return UnifiedTreeCore(params, {ComponentType.FULL: component})


def make_key(tokens, **kwargs):
    return RadixKey(
        array("q", tokens), extra_key="model", cache_salt="tenant", **kwargs
    )


def insert(core, tokens, values, **kwargs):
    step = core.begin_insert(InsertParams(make_key(tokens), values, **kwargs))
    actions = list(step.actions)
    while step.result is None:
        step = core.resume_insert()
        actions.extend(step.actions)
    return step.result, actions


def match(core, tokens):
    return core.match_prefix(MatchPrefixParams(make_key(tokens))).device_indices


def released(actions, action_type):
    chunks = [
        indices
        for action in actions
        if type(action) is action_type
        for indices in action.indices
    ]
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.int64)


class TestUnifiedTreeInsertCursor(unittest.TestCase):
    def test_chunked_prefix_branch_and_stored_suffix_ownership(self):
        # Split inside the third cached node, after traversing two whole nodes.
        # Test logical bigram boundaries and unaligned incoming token tails.
        for page_size in (1, 2, 4):
            for is_bigram in (False, True):
                with self.subTest(page_size=page_size, is_bigram=is_bigram):
                    core = make_core(page_size, is_bigram)
                    raw = array("q", range(35 + is_bigram))
                    old_values = torch.arange(32)
                    for end in (8, 16, 24, 32):
                        insert(
                            core,
                            raw[: end + is_bigram],
                            old_values[:end],
                            prev_prefix_len=max(0, end - 8),
                            chunked=True,
                        )

                    branch = raw[:]
                    branch[21] = 9999
                    new_values = torch.arange(100, 135)
                    result, actions = insert(
                        core,
                        branch,
                        new_values,
                        prev_prefix_len=8,
                        track_adopted_ranges=True,
                    )
                    prefix = (21 - is_bigram) // page_size * page_size
                    length = 35 // page_size * page_size
                    self.assertEqual(result.prefix_len, prefix)
                    self.assertEqual(
                        result.adopted_ranges, {ComponentType.FULL: [(prefix, length)]}
                    )
                    torch.testing.assert_close(
                        released(actions, FreeDeviceKV), new_values[8:prefix]
                    )
                    torch.testing.assert_close(
                        match(core, branch),
                        torch.cat((old_values[:prefix], new_values[prefix:length])),
                    )
                    torch.testing.assert_close(
                        match(core, raw[: 32 + is_bigram]), old_values
                    )

                    leaf = core.node_by_id(result.last_device_node)
                    saved_key = leaf.key.raw_token_ids()[:]
                    self.assertEqual(saved_key, branch[prefix : length + is_bigram])
                    branch[-2] = 7777
                    new_values.fill_(-1)
                    self.assertEqual(leaf.key.raw_token_ids(), saved_key)
                    torch.testing.assert_close(
                        leaf.component_data[ComponentType.FULL].value,
                        torch.arange(100 + prefix, 100 + length),
                    )

    def test_exact_hit_empty_key_and_limit_do_not_adopt_kv(self):
        for page_size in (1, 4):
            for is_bigram in (False, True):
                with self.subTest(page_size=page_size, is_bigram=is_bigram):
                    core = make_core(page_size, is_bigram)
                    raw = array("q", range(33))
                    for end in (8, 16, 24):
                        insert(
                            core,
                            raw[: end + is_bigram],
                            torch.arange(end),
                            prev_prefix_len=max(0, end - 8),
                            chunked=True,
                        )
                    step = core.begin_insert(
                        InsertParams(
                            key=make_key(raw, limit=16 + is_bigram),
                            value=torch.arange(100, 132),
                            prev_prefix_len=16,
                            track_adopted_ranges=True,
                        )
                    )
                    self.assertEqual(step.result.prefix_len, 16)
                    self.assertEqual(step.result.adopted_ranges, {})
                    self.assertEqual(step.actions, [])
                    empty, actions = insert(core, [], torch.empty(0, dtype=torch.int64))
                    self.assertEqual(empty.prefix_len, 0)
                    self.assertEqual(empty.last_device_node, core.root_node.id)
                    self.assertEqual(actions, [])
                    torch.testing.assert_close(
                        match(core, raw[: 24 + is_bigram]), torch.arange(24)
                    )

    def test_duplicate_free_ranges_keep_absolute_request_positions(self):
        core = make_core(page_size=4)
        raw = array("q", range(28))
        for end in (8, 16, 24):
            insert(
                core,
                raw[:end],
                torch.arange(end),
                prev_prefix_len=max(0, end - 8),
                chunked=True,
            )
        fresh = torch.arange(100, 128)
        result, actions = insert(
            core,
            raw,
            fresh,
            prev_prefix_len=6,
            swa_evicted_seqlen=18,
            track_adopted_ranges=True,
        )
        self.assertEqual(result.prefix_len, 24)
        self.assertEqual(result.adopted_ranges, {ComponentType.FULL: [(24, 28)]})
        torch.testing.assert_close(released(actions, FreeDeviceKVFullOnly), fresh[6:18])
        torch.testing.assert_close(released(actions, FreeDeviceKV), fresh[18:24])
        torch.testing.assert_close(
            match(core, raw), torch.cat((torch.arange(24), fresh[24:]))
        )

    def test_resume_after_each_backup_preserves_cursor_and_leaf(self):
        for is_bigram in (False, True):
            with self.subTest(is_bigram=is_bigram):
                core = make_core(page_size=4, is_bigram=is_bigram)
                raw = array("q", range(33))
                nodes = []
                for end in (8, 16, 24):
                    result, _ = insert(
                        core,
                        raw[: end + is_bigram],
                        torch.arange(end),
                        prev_prefix_len=max(0, end - 8),
                        chunked=True,
                    )
                    nodes.append(result.last_device_node)
                core.enable_hicache = True
                core.write_through_threshold = 1
                fresh = torch.arange(100, 132)
                step = core.begin_insert(
                    InsertParams(
                        make_key(raw[: 32 + is_bigram]),
                        fresh,
                        prev_prefix_len=24,
                        track_adopted_ranges=True,
                    )
                )
                backed_up = []
                while step.result is None:
                    self.assertTrue(core.has_ongoing_insert())
                    self.assertTrue(step.actions)
                    for action in step.actions:
                        self.assertIsInstance(action, BackupKV)
                        backed_up.extend(action.node_ids)
                        for node_id in action.node_ids:
                            cd = core.node_by_id(node_id).component_data[
                                ComponentType.FULL
                            ]
                            cd.host_value = cd.value.clone()
                    step = core.resume_insert()
                backed_up.extend(
                    node_id
                    for action in step.actions
                    if isinstance(action, BackupKV)
                    for node_id in action.node_ids
                )
                self.assertFalse(core.has_ongoing_insert())
                self.assertEqual(backed_up, nodes + [step.result.last_device_node])
                self.assertEqual(step.result.prefix_len, 24)
                self.assertEqual(
                    step.result.adopted_ranges, {ComponentType.FULL: [(24, 32)]}
                )
                torch.testing.assert_close(
                    match(core, raw[: 32 + is_bigram]),
                    torch.cat((torch.arange(24), fresh[24:])),
                )

    def test_restore_evicted_middle_node_adopts_correct_request_slice(self):
        core = make_core(page_size=4)
        raw = array("q", range(28))
        nodes = []
        for end in (8, 16, 24):
            result, _ = insert(
                core,
                raw[:end],
                torch.arange(end),
                prev_prefix_len=max(0, end - 8),
                chunked=True,
            )
            nodes.append(core.node_by_id(result.last_device_node))
        # Full KV may be gone while auxiliary/host state preserves the node.
        middle = nodes[1].component_data[ComponentType.FULL]
        middle.host_value = middle.value.clone()
        middle.value = None
        core.component_evictable_size_[ComponentType.FULL] -= 8
        fresh = torch.arange(100, 128)
        result, actions = insert(core, raw, fresh, track_adopted_ranges=True)
        self.assertEqual(result.prefix_len, 24)
        self.assertEqual(
            result.adopted_ranges, {ComponentType.FULL: [(8, 16), (24, 28)]}
        )
        torch.testing.assert_close(
            released(actions, FreeDeviceKV), torch.cat((fresh[:8], fresh[16:24]))
        )
        torch.testing.assert_close(middle.value, fresh[8:16])
        torch.testing.assert_close(
            match(core, raw),
            torch.cat((torch.arange(8), fresh[8:16], torch.arange(16, 24), fresh[24:])),
        )


if __name__ == "__main__":
    unittest.main()
