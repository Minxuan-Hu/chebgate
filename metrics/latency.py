import os
import time
import json
import numpy as np
import torch

from chebgate.core.io import append_csv_row, write_json
from chebgate.core.parse import amp_dtype_name


def _as_device(device):
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


@torch.no_grad()
def latency_ms_samples(
    net,
    shape=(1, 3, 32, 32),
    iters=200,
    warmup=50,
    device="cuda",
    amp_dtype=None,
    return_samples: bool = False,
):
    """
    CUDA-event timing when CUDA; perf_counter on CPU.
    Returns dict with mean/std/median/p10/p90/imgs_per_s (+ samples_ms optional).
    """
    device = _as_device(device)
    net.eval()

    x = torch.randn(*shape, device=device).contiguous(memory_format=torch.channels_last)
    use_amp = (amp_dtype is not None and device.type == "cuda")

    # Warmup (includes compilation/autotune if any)
    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _ = net(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    samples_ms = []
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            starter.record()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _ = net(x)
            ender.record()
            torch.cuda.synchronize()
            samples_ms.append(starter.elapsed_time(ender))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _ = net(x)
            samples_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(samples_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    stats = {
        "mean_ms": mean_ms,
        "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median_ms": float(np.median(arr)),
        "p10_ms": float(np.percentile(arr, 10)),
        "p90_ms": float(np.percentile(arr, 90)),
        "imgs_per_s": (shape[0] * 1000.0 / mean_ms) if mean_ms > 0 else float("inf"),
    }
    if return_samples:
        stats["samples_ms"] = [float(x) for x in arr]
    return stats


@torch.no_grad()
def latency_sweep_csv(
    net,
    logdir: str,
    device="cuda",
    amp_dtype=None,
    batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128),
    iters=200,
    warmup=50,
    tag="model",
    samples_dir: str = None,
):
    device = _as_device(device)
    path = os.path.join(logdir, f"latency_sweep_{tag}.csv")
    header = [
        "batch_size",
        "mean_ms",
        "std_ms",
        "median_ms",
        "p10_ms",
        "p90_ms",
        "imgs_per_s",
        "iters",
        "warmup",
        "amp_dtype",
    ]
    if samples_dir:
        os.makedirs(samples_dir, exist_ok=True)

    for bs in batch_sizes:
        stats = latency_ms_samples(
            net,
            shape=(bs, 3, 32, 32),
            iters=iters,
            warmup=warmup,
            device=device,
            amp_dtype=amp_dtype,
            return_samples=bool(samples_dir),
        )
        append_csv_row(
            path,
            header,
            {
                "batch_size": bs,
                "mean_ms": stats["mean_ms"],
                "std_ms": stats["std_ms"],
                "median_ms": stats["median_ms"],
                "p10_ms": stats["p10_ms"],
                "p90_ms": stats["p90_ms"],
                "imgs_per_s": stats["imgs_per_s"],
                "iters": iters,
                "warmup": warmup,
                "amp_dtype": amp_dtype_name(amp_dtype),
            },
        )

        if samples_dir and "samples_ms" in stats:
            write_json(
                {"batch_size": bs, "samples_ms": stats["samples_ms"]},
                os.path.join(samples_dir, f"latency_samples_bs{bs}.json"),
            )


@torch.no_grad()
def peak_mem_mb(net, shape=(128, 3, 32, 32), device="cuda", amp_dtype=None):
    """
    Peak allocated memory during a single forward.
    Returns None if CUDA not available.
    """
    device = _as_device(device)
    if not torch.cuda.is_available() or device.type != "cuda":
        return None

    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(*shape, device=device).contiguous(memory_format=torch.channels_last)
    net.eval()

    use_amp = (amp_dtype is not None and device.type == "cuda")
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        _ = net(x)

    return torch.cuda.max_memory_allocated() / (1024 * 1024)
