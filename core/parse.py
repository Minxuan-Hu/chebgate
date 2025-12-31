from typing import Tuple
import torch


def parse_tuple_ints(s: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")
    if not vals:
        raise ValueError(f"Bad tuple string: {s}")
    return vals


def amp_dtype_name(dt):
    if dt is None:
        return "fp32"
    if dt == torch.float16:
        return "fp16"
    if dt == torch.bfloat16:
        return "bf16"
    return str(dt)
