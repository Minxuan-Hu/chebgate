import os
import csv
import torch

from chebgate.core.io import write_json
from chebgate.core.state_dict import _unwrap_compiled
from chebgate.model.chebconv import ChebConv2d


@torch.no_grad()
def collect_gate_stats(model, loader, device, logdir, amp_dtype=None):
    for m in _unwrap_compiled(model).modules():
        if isinstance(m, ChebConv2d):
            m.reset_gate_stats()

    model.eval()
    use_amp = (amp_dtype is not None and device.type == "cuda")

    for xb, _ in loader:
        xb = xb.to(device).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _ = model(xb)

    all_stats = []
    for name, m in _unwrap_compiled(model).named_modules():
        if isinstance(m, ChebConv2d):
            s = m.gate_stats()
            if s:
                rec = {"layer": name}
                rec.update(s)
                all_stats.append(rec)

    write_json(all_stats, os.path.join(logdir, "gate_stats_final.json"))
    return all_stats


@torch.no_grad()
def dump_order_scales(model, logdir: str):
    rows = []
    for name, m in _unwrap_compiled(model).named_modules():
        if isinstance(m, ChebConv2d):
            a = m.order_scales.detach().cpu().float().tolist()
            rows.append({"layer": name, "K_plus_1": len(a), "alpha": a})

    write_json(rows, os.path.join(logdir, "F3_order_scales.json"))

    if rows:
        Kmax = max(r["K_plus_1"] for r in rows)
        with open(os.path.join(logdir, "F3_order_scales.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer"] + [f"alpha_k{k}" for k in range(Kmax)])
            for r in rows:
                arr = r["alpha"] + [""] * (Kmax - len(r["alpha"]))
                w.writerow([r["layer"]] + [f"{x:.8f}" if x != "" else "" for x in arr])

    return rows
