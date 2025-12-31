from typing import Dict
import torch
import torch.nn as nn


def _unwrap_compiled(model):
    return getattr(model, "_orig_mod", model)


def state_dict_uncompiled(model: nn.Module) -> Dict[str, torch.Tensor]:
    return _unwrap_compiled(model).state_dict()


def strip_orig_mod_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("_orig_mod.") for k in sd.keys()):
        return sd
    return {k.split(".", 1)[1]: v for k, v in sd.items()}


def load_state_dict_portable(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = True):
    _unwrap_compiled(model).load_state_dict(strip_orig_mod_prefix(state_dict), strict=strict)
