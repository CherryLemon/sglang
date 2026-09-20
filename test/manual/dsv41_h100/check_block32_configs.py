"""Verify tuned dispatch on boundary/graph shapes without installing configs."""

import argparse
import json
import time
from pathlib import Path
import torch
from sglang.kernels.ops.quantization import fp8_kernel as impl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--shapes",
        default="576:5120,1792:5120,16384:1280,5120:4096,25600:6144",
        help="Comma-separated N:K weight shapes to validate",
    )
    p.add_argument(
        "--rows",
        default="0,1,2,3,4,5,6,7,8,12,16,24,31,32,48,64,80,96,128",
        help="Rows, including CUDA graph tiers and configuration boundaries",
    )
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    default = dict(
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=32,
        num_warps=4,
        num_stages=3,
    )
    results = []
    paths = sorted(
        Path(a.configs).glob(
            "N=*,K=*,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=*32, 32*.json"
        )
    )
    expected = {tuple(map(int, shape.split(":"))) for shape in a.shapes.split(",")}
    paths = [
        path
        for path in paths
        if (int(path.name.split(",")[0][2:]), int(path.name.split(",")[1][2:]))
        in expected
    ]
    assert len(paths) == len(expected), "Missing requested H100 block32 configs"
    for path in paths:
        n = int(path.name.split(",")[0][2:])
        k = int(path.name.split(",")[1][2:])
        cfg = {int(x): v for x, v in json.loads(path.read_text()).items()}
        for m in map(int, a.rows.split(",")):
            torch.manual_seed(m + 37)
            w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
            ws = torch.exp2(
                torch.randint(-4, 4, (n // 32, k // 32), device="cuda").float()
            )
            q = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
            qs = torch.exp2(torch.randint(-4, 4, (m, k // 32), device="cuda").float())
            chosen = cfg[min(cfg, key=lambda b: abs(m - b))]

            def call(config, dtype=torch.bfloat16):
                impl.get_w8a8_block_fp8_configs = lambda *args: {m: config}
                return impl.w8a8_block_fp8_matmul_triton(
                    q, w, qs, ws, [32, 32], output_dtype=dtype
                )

            # Original operator is the compatibility control; FP32 comparison catches scale/stride faults.
            base32 = call(default, torch.float32)
            test32 = call(chosen, torch.float32)
            if m:
                rms = base32.square().mean().sqrt().item()
                torch.testing.assert_close(test32, base32, rtol=1e-4, atol=rms * 3e-4)
            base = call(default)
            test = call(chosen)
            if m:
                torch.testing.assert_close(
                    test.float(), base.float(), rtol=0.008, atol=rms * 3e-4
                )
            for _ in range(2):
                call(chosen)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = call(chosen)
            for seed in [11, 31, 97]:
                torch.manual_seed(seed)
                q.copy_(torch.randn(m, k, device="cuda").to(q.dtype))
                qs.copy_(
                    torch.exp2(torch.randint(-5, 5, qs.shape, device="cuda").float())
                )
                g.replay()
                torch.cuda.synchronize()
                ref = call(default)
                if m:
                    rr = ref.float().square().mean().sqrt().item()
                    torch.testing.assert_close(
                        out.float(), ref.float(), rtol=0.008, atol=rr * 3e-4
                    )
                else:
                    assert out.shape == (0, n)
            results.append({"shape": [m, n, k], "passed": True, "graph_replays": 3})
            print("PASS", m, n, k, flush=True)
            del q, qs, w, ws, base32, test32, base, test, g, out, ref
        torch.cuda.empty_cache()
    Path(a.output).write_text(
        json.dumps({"passed": True, "checks": results, "utc": time.time()}, indent=2)
    )


if __name__ == "__main__":
    main()
