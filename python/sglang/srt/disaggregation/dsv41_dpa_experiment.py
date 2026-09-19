"""Opt-in, isolated DeepSeek-V4.1 PD decode layouts with attention DP.

Default off. The model and KV layout stay unchanged. These predicates
intentionally enumerate the *only* admitted layouts; every other combination
(mixed attention TP sizes, other DP widths, other backends) is rejected, and all
cache-layout checks remain mandatory.

Admitted decode layouts (global tp=ep=8):
  * attention TP1 x DP8  (validated correctness layout)
  * attention TP2 x DP4  (v3; this round's target)

The peer (prefill) side is always attention TP8 / DP1, so the decode side must
receive MLA state from a wider attention-TP group than it holds. That re-shard is
admitted only for the pairs listed below.
"""

import os

# (tp_size, ep_size, dp_size, attn_tp_size, attn_dp_size)
_ADMITTED_DECODE_LAYOUTS = (
    (8, 8, 8, 1, 8),  # attention TP1 x DP8
    (8, 8, 4, 2, 4),  # attention TP2 x DP4
)

# (local_attn_tp, local_attn_dp, remote_attn_tp, remote_dp)
_ADMITTED_PEER_PAIRS = (
    (1, 8, 8, 1),  # decode attn TP1 x DP8  <- prefill attn TP8 / DP1
    (2, 4, 8, 1),  # decode attn TP2 x DP4  <- prefill attn TP8 / DP1
)


def enabled() -> bool:
    return os.environ.get("SGLANG_DSV41_EXPERIMENTAL_DPA_PD") == "1"


def _attention_widths(cfg):
    """(attn_tp_size, attn_dp_size) from the configured leaves.

    attn_cp_size is enforced to 1 by the caller (the V4.1 feature hook), so the
    quotient below is the attention tensor width.
    """
    attn_dp_size = cfg.dp_size if cfg.enable_dp_attention else 1
    return cfg.tp_size // attn_dp_size, attn_dp_size


def allow_decode_config(cfg) -> bool:
    if not enabled():
        return False
    if cfg.disaggregation_mode != "decode":
        return False
    if not (cfg.enable_dp_attention and cfg.enable_dp_lm_head):
        return False
    if cfg.moe_a2a_backend != "none":
        return False
    attn_tp_size, attn_dp_size = _attention_widths(cfg)
    return (
        cfg.tp_size,
        cfg.ep_size,
        cfg.dp_size,
        attn_tp_size,
        attn_dp_size,
    ) in _ADMITTED_DECODE_LAYOUTS


def allow_peer_topology(*, is_mla, local_tp, local_ep, local_attn_tp,
                        local_attn_dp, remote_attn_tp, remote_dp) -> bool:
    return (
        enabled()
        and is_mla
        and local_tp == local_ep == 8
        and (local_attn_tp, local_attn_dp, remote_attn_tp, remote_dp)
        in _ADMITTED_PEER_PAIRS
    )


def validate_c2_strides(source, destination) -> None:
    # Generic state transfer uses the sender's item size to address both sides.
    # A different receiver stride must fail before issuing an RDMA/TCP write.
    if list(source) != list(destination):
        raise ValueError(
            "DeepSeek-V4.1 C2 state strides differ between P and D: "
            f"source={list(source)}, destination={list(destination)}"
        )
