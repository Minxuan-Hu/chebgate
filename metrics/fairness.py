import os
import torch

from chebgate.core.hashing import state_dict_sha256
from chebgate.core.parse import amp_dtype_name
from chebgate.core.io import append_csv_row
from chebgate.core.state_dict import strip_orig_mod_prefix, load_state_dict_portable
from chebgate.model.net import ChebResNet
from .latency import latency_ms_samples


def can_run_compiled(device: torch.device) -> bool:
    if not hasattr(torch, "compile"):
        return False
    if not (isinstance(device, torch.device) and device.type == "cuda"):
        return False
    try:
        major, _ = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return major >= 7  # Volta+


def get_exec_mode(model) -> str:
    """
    Best-effort, non-invasive exec-mode label.
    - torch.compile typically wraps into an OptimizedModule with `_orig_mod`.
    """
    if hasattr(model, "_orig_mod"):
        return "compiled"
    name = model.__class__.__name__.lower()
    if "optimizedmodule" in name:
        return "compiled"
    return "eager"


@torch.no_grad()
def fairness_latency_all(
    base_state_dict,
    model_cfg,
    realizations,
    logdir,
    device,
    amp_dtype,
    batch_sizes,
    compiled: bool = False,
    compile_mode: str = "reduce-overhead",
    fullgraph: bool = False,
    tag: str = "all_realizations",
):
    """
    Same-weights latency sweep across a list of realizations.

    Writes one CSV with rows:
      realization, batch_size, mean/std/median/p10/p90, imgs/s,
      exec_mode, compile_attempted/compiled, compile_mode/fullgraph, amp dtype, sha256.
    """
    base_state_dict = strip_orig_mod_prefix(base_state_dict)
    weight_meta = state_dict_sha256(base_state_dict)

    path = os.path.join(logdir, f"latency_sweep_{tag}.csv")
    header = [
        "realization",
        "batch_size",
        "mean_ms",
        "std_ms",
        "median_ms",
        "p10_ms",
        "p90_ms",
        "imgs_per_s",
        "iters",
        "warmup",
        "exec_mode",
        "compile_attempted",
        "compiled",
        "compile_mode",
        "fullgraph",
        "amp_dtype",
        "weights_sha256",
        "weights_numel",
    ]

    iters, warmup = 200, 50
    amp_name = amp_dtype_name(amp_dtype)

    for r in realizations:
        net = ChebResNet(
            classes=model_cfg["classes"],
            K=model_cfg["K"],
            depth=model_cfg["depth"],
            widths=model_cfg["widths"],
            drop_rate=model_cfg["drop_rate"],
            lap=model_cfg["lap"],
            realization=r,
            gate_mode=model_cfg["gate_mode"],
            stabilize_cheb=model_cfg["stabilize_cheb"],
        ).to(device)

        net = net.to(memory_format=torch.channels_last)
        load_state_dict_portable(net, base_state_dict, strict=True)

        net.eval()

        compile_attempted = 0
        compiled_flag = 0
        if compiled and hasattr(torch, "compile"):
            compile_attempted = 1
            try:
                net = torch.compile(net, mode=compile_mode, fullgraph=bool(fullgraph))
                compiled_flag = 1
            except Exception:
                compiled_flag = 0

        exec_mode = get_exec_mode(net)

        for bs in batch_sizes:
            stats = latency_ms_samples(
                net,
                shape=(bs, 3, 32, 32),
                iters=iters,
                warmup=warmup,
                device=device,
                amp_dtype=amp_dtype,
            )
            append_csv_row(
                path,
                header,
                {
                    "realization": r,
                    "batch_size": bs,
                    "mean_ms": stats["mean_ms"],
                    "std_ms": stats["std_ms"],
                    "median_ms": stats["median_ms"],
                    "p10_ms": stats["p10_ms"],
                    "p90_ms": stats["p90_ms"],
                    "imgs_per_s": stats["imgs_per_s"],
                    "iters": iters,
                    "warmup": warmup,
                    "exec_mode": exec_mode,
                    "compile_attempted": int(compile_attempted),
                    "compiled": int(compiled_flag),
                    "compile_mode": compile_mode if compiled_flag else "",
                    "fullgraph": int(bool(fullgraph) if compiled_flag else 0),
                    "amp_dtype": amp_name,
                    "weights_sha256": weight_meta["sha256"],
                    "weights_numel": weight_meta["numel"],
                },
            )
