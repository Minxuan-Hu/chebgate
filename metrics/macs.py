import csv
import torch
import torch.nn as nn

from chebgate.model.chebconv import ChebConv2d


@torch.no_grad()
def profile_macs(model: nn.Module, input_size=(1, 3, 32, 32), device=None) -> int:
    was_training = model.training
    model.eval()

    macs_total = 0
    handles = []

    def conv_hook(m: nn.Conv2d, inp, out):
        nonlocal macs_total
        if not isinstance(out, torch.Tensor):
            return
        N, Cout, Hout, Wout = out.shape
        Kh = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
        Kw = m.kernel_size[1] if isinstance(m.kernel_size, tuple) else m.kernel_size
        Cin = m.in_channels
        groups = m.groups if m.groups else 1
        macs_total += int(N * Hout * Wout * Cout * (Cin // groups) * Kh * Kw)

    def linear_hook(m: nn.Linear, inp, out):
        nonlocal macs_total
        if not isinstance(out, torch.Tensor):
            return
        N = out.shape[0]
        macs_total += int(N * m.in_features * m.out_features)

    def cheb_hook(m: ChebConv2d, inp, out):
        nonlocal macs_total
        if not isinstance(out, torch.Tensor):
            return
        if m.realization == "concat":
            return
        N, Cout, Hout, Wout = out.shape
        Cin = m.Cin
        K1 = m.K + 1
        macs_total += int(N * Hout * Wout * Cout * Cin * K1)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, ChebConv2d):
            handles.append(m.register_forward_hook(cheb_hook))

    x = torch.randn(*input_size)
    if device is not None:
        x = x.to(device).contiguous(memory_format=torch.channels_last)
        model.to(device)
    _ = model(x)

    for h in handles:
        h.remove()

    if was_training:
        model.train()

    return macs_total


@torch.no_grad()
def profile_macs_breakdown(
    model: nn.Module,
    input_size=(1, 3, 32, 32),
    device=None,
    save_csv: str = None,
):
    was_training = model.training
    model.eval()

    name_of = {id(m): n for n, m in model.named_modules()}
    rows = []
    totals = {"stem": 0, "l1": 0, "l2": 0, "l3": 0, "fc": 0, "other": 0}
    total_macs = 0
    handles = []

    def add_row(m, macs: int, typ: str, notes: str = ""):
        nonlocal total_macs
        name = name_of.get(id(m), "")
        rows.append({"name": name, "type": typ, "macs": int(macs), "notes": notes})
        total_macs += int(macs)

        bucket = "other"
        for b in ["stem", "l1", "l2", "l3", "fc"]:
            if name.startswith(b):
                bucket = b
                break
        totals[bucket] += int(macs)

    def conv_hook(m: nn.Conv2d, inp, out):
        if not isinstance(out, torch.Tensor):
            return
        N, Cout, Hout, Wout = out.shape
        Kh = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
        Kw = m.kernel_size[1] if isinstance(m.kernel_size, tuple) else m.kernel_size
        Cin = m.in_channels
        groups = m.groups if m.groups else 1
        macs = int(N * Hout * Wout * Cout * (Cin // groups) * Kh * Kw)
        add_row(m, macs, "Conv2d")

    def linear_hook(m: nn.Linear, inp, out):
        if not isinstance(out, torch.Tensor):
            return
        N = out.shape[0]
        macs = int(N * m.in_features * m.out_features)
        add_row(m, macs, "Linear")

    def cheb_hook(m: ChebConv2d, inp, out):
        if not isinstance(out, torch.Tensor):
            return
        if m.realization == "concat":
            return
        N, Cout, Hout, Wout = out.shape
        Cin = m.Cin
        K1 = m.K + 1
        macs = int(N * Hout * Wout * Cout * Cin * K1)
        add_row(m, macs, "ChebConv2d", notes="big_1x1_equiv")

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, ChebConv2d):
            handles.append(m.register_forward_hook(cheb_hook))

    x = torch.randn(*input_size)
    if device is not None:
        x = x.to(device).contiguous(memory_format=torch.channels_last)
        model.to(device)
    _ = model(x)

    for h in handles:
        h.remove()

    if was_training:
        model.train()

    if save_csv:
        header = ["name", "type", "macs", "percent_of_total", "notes"]
        with open(save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in sorted(rows, key=lambda z: -z["macs"]):
                pct = (100.0 * r["macs"] / total_macs) if total_macs > 0 else 0.0
                w.writerow({**r, "percent_of_total": f"{pct:.4f}"})

            w.writerow(
                {
                    "name": "--stage_totals--",
                    "type": "",
                    "macs": total_macs,
                    "percent_of_total": "100.0000",
                    "notes": "",
                }
            )
            for b in ["stem", "l1", "l2", "l3", "fc", "other"]:
                pct = (100.0 * totals[b] / total_macs) if total_macs > 0 else 0.0
                w.writerow(
                    {
                        "name": f"stage:{b}",
                        "type": "",
                        "macs": totals[b],
                        "percent_of_total": f"{pct:.4f}",
                        "notes": "bucketed by name prefix",
                    }
                )

    return {"rows": rows, "total_macs": total_macs, "stage_totals": totals}
