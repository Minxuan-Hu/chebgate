import hashlib
from typing import Dict
import torch


def _tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    try:
        return t.numpy().tobytes()  # fast path (works on modern NumPy incl. bfloat16)
    except Exception:
        return t.view(torch.uint8).cpu().numpy().tobytes()


def state_dict_sha256(sd: Dict[str, torch.Tensor]) -> Dict[str, object]:
    h = hashlib.sha256()
    numel = 0
    for k in sorted(sd.keys()):
        v = sd[k]
        numel += v.numel()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(v.shape)).encode("utf-8"))
        h.update(str(v.dtype).encode("utf-8"))
        h.update(_tensor_bytes(v))
    return {"sha256": h.hexdigest(), "numel": int(numel)}
