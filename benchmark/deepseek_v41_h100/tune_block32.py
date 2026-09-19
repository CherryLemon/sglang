"""Tune existing block32 Hopper GEMM paths without changing a serving process."""

import argparse
import itertools
import json
from pathlib import Path
import statistics
import time
import torch
import triton
from sglang.kernels.ops.quantization import fp8_kernel as impl

DEFAULT = dict(
    BLOCK_SIZE_M=64,
    BLOCK_SIZE_N=32,
    BLOCK_SIZE_K=32,
    GROUP_SIZE_M=32,
    num_warps=4,
    num_stages=3,
)
SHAPES = [(576, 5120), (1792, 5120), (16384, 1280), (5120, 4096), (25600, 6144)]


def measure(fn, iterations=80):
    for _ in range(25):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iterations):
            fn()
    vals = []
    for _ in range(5):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        vals.append(a.elapsed_time(b) * 1000 / iterations)
    return {"p50_us": statistics.median(vals), "min_us": min(vals), "max_us": max(vals)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--shapes", default="0,1,2,3,4")
    p.add_argument("--rows", default="1,6,24,64")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(431)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_capability() == (9, 0)
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "scope": "captured operator call including output, split workspace and reduction; no serving changes",
        "results": [],
        "started_utc": time.time(),
    }
    original = impl.get_w8a8_block_fp8_configs
    configs = [DEFAULT]
    for bn, sk, swap in itertools.product([32, 64], [2, 4, 8, 16], [False, True]):
        if a.quick and (sk not in [4, 8] or not swap):
            continue
        configs.append(
            dict(
                BLOCK_SIZE_M=16,
                BLOCK_SIZE_N=bn,
                BLOCK_SIZE_K=32,
                GROUP_SIZE_M=1,
                num_warps=4,
                num_stages=3,
                SWAP_AB=swap,
                SPLIT_K=sk,
            )
        )
    for shape_index in map(int, a.shapes.split(",")):
        n, k = SHAPES[shape_index]
        w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        ws = torch.exp2(torch.randint(-4, 4, (n // 32, k // 32), device="cuda").float())
        wf = w.float() * ws.repeat_interleave(32, 0).repeat_interleave(32, 1)
        for m in map(int, a.rows.split(",")):
            q = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
            qs = torch.exp2(torch.randint(-4, 4, (m, k // 32), device="cuda").float())
            qf = q.float() * qs.repeat_interleave(32, 1)
            reference = qf @ wf.T
            ref_rms = reference.square().mean().sqrt().item()
            reference_bf16 = reference.to(torch.bfloat16)
            impl.get_w8a8_block_fp8_configs = lambda *args: {m: DEFAULT}
            control32 = impl.w8a8_block_fp8_matmul_triton(
                q, w, qs, ws, [32, 32], output_dtype=torch.float32
            )
            control16 = impl.w8a8_block_fp8_matmul_triton(
                q, w, qs, ws, [32, 32], output_dtype=torch.bfloat16
            )

            def errors(y):
                e = y.float() - reference
                return {
                    "max": e.abs().max().item(),
                    "rmse": e.square().mean().sqrt().item(),
                }

            control_errors = {"fp32": errors(control32), "bf16": errors(control16)}
            record = {
                "M": m,
                "N": n,
                "K": k,
                "reference_rms": ref_rms,
                "control_error": control_errors,
                "candidates": [],
            }
            for cfg in configs:
                impl.get_w8a8_block_fp8_configs = lambda *args, config=cfg: {m: config}

                def fn(dtype=torch.bfloat16):
                    return impl.w8a8_block_fp8_matmul_triton(
                        q, w, qs, ws, [32, 32], output_dtype=dtype
                    )

                try:
                    # FP32 isolates accumulation error from BF16 output rounding.
                    x = fn(torch.float32)
                    xe = errors(x)
                    assert (
                        xe["rmse"]
                        <= control_errors["fp32"]["rmse"] * 1.05 + ref_rms * 5e-6
                    ), xe
                    assert (
                        xe["max"]
                        <= control_errors["fp32"]["max"] * 1.05 + ref_rms * 2e-5
                    ), xe
                    y = fn()
                    ye = errors(y)
                    assert (
                        ye["rmse"]
                        <= control_errors["bf16"]["rmse"] * 1.05 + ref_rms * 2e-5
                    ), ye
                    assert (
                        ye["max"]
                        <= control_errors["bf16"]["max"] * 1.05 + ref_rms * 2e-5
                    ), ye
                    torch.testing.assert_close(
                        y.float(),
                        reference,
                        rtol=0.004,
                        atol=control_errors["fp32"]["max"] * 1.1 + ref_rms * 2e-5,
                    )
                    timing = measure(fn)
                    e = {
                        "config": cfg,
                        "correct": True,
                        "fp32_max_abs_error": (x - reference).abs().max().item(),
                        "reference_error": {"fp32": xe, "bf16": ye},
                        "bf16_differing_fraction": (y != reference_bf16)
                        .float()
                        .mean()
                        .item(),
                        **timing,
                    }
                except Exception as exc:
                    e = {"config": cfg, "correct": False, "error": str(exc)[:600]}
                record["candidates"].append(e)
            passed = [x for x in record["candidates"] if x["correct"]]
            assert record["candidates"][0]["correct"], record["candidates"][0]
            best = min(passed, key=lambda x: x["p50_us"])
            record["best"] = best
            record["baseline"] = record["candidates"][0]
            record["speedup"] = record["baseline"]["p50_us"] / best["p50_us"]
            report["results"].append(record)
            out.write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    {
                        "shape": [m, n, k],
                        "baseline_us": record["baseline"]["p50_us"],
                        "best_us": best["p50_us"],
                        "speedup": record["speedup"],
                        "config": best["config"],
                    }
                ),
                flush=True,
            )
        del w, ws, wf
        torch.cuda.empty_cache()
    impl.get_w8a8_block_fp8_configs = original
    report["finished_utc"] = time.time()
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
