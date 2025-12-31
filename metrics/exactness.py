import torch

from chebgate.core.fp32 import fp32_reference_mode
from chebgate.core.state_dict import strip_orig_mod_prefix, load_state_dict_portable
from chebgate.model.net import ChebResNet


@torch.no_grad()
def network_exactness_check(state_dict, model_cfg, device):
    """
    Network-level strict FP32 reference equivalence across:
      - concat
      - streamed
      - mstream
      - gemm

    Runs with:
      - autocast disabled
      - TF32 disabled temporarily
      - reports max/mean abs deltas
    """
    state_dict = strip_orig_mod_prefix(state_dict)

    def build(r):
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
        load_state_dict_portable(net, state_dict, strict=True)
        net.eval()
        return net

    a = build("concat")
    b = build("streamed")
    c = build("mstream")
    d = build("gemm")

    x = torch.randn(8, 3, 32, 32, device=device, dtype=torch.float32)
    device_type = "cuda" if device.type == "cuda" else "cpu"

    with fp32_reference_mode(strict_tf32_off=True, disable_cudnn_benchmark=True):
        with torch.autocast(device_type=device_type, enabled=False):
            ya = a(x).to(dtype=torch.float32)
            yb = b(x).to(dtype=torch.float32)
            yc = c(x).to(dtype=torch.float32)
            yd = d(x).to(dtype=torch.float32)

    def dxy(u, v):
        dv = (u - v).abs()
        return {"max_abs": float(dv.max().item()), "mean_abs": float(dv.mean().item())}

    return {
        "mode": {
            "autocast": "disabled",
            "tf32": "disabled",
            "dtype": "fp32",
            "note": "deltas reflect floating-point reassociation / kernel differences across realizations",
        },
        "C_vs_S": dxy(ya, yb),
        "C_vs_M": dxy(ya, yc),
        "C_vs_G": dxy(ya, yd),
        "S_vs_M": dxy(yb, yc),
        "S_vs_G": dxy(yb, yd),
        "M_vs_G": dxy(yc, yd),
    }
