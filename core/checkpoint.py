import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def _cpu_tensor(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def _to_numpy_int64(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(np.int64, copy=False)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.int64, copy=False)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.int64)
    raise TypeError(f"Unsupported index type: {type(x)}")


def capture_rng_state() -> Dict[str, Any]:
    """
    Capture RNG states so resume does not change stochastic behavior.
    This is best-effort and safe on CPU-only environments.
    """
    state: Dict[str, Any] = {}
    state["python_random_state"] = random.getstate()
    state["numpy_random_state"] = np.random.get_state()
    state["torch_rng_state"] = torch.get_rng_state()

    if torch.cuda.is_available():
        try:
            state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            # Some environments disallow this; keep resume robust.
            state["torch_cuda_rng_state_all"] = None
    else:
        state["torch_cuda_rng_state_all"] = None

    # Torch XLA RNG is not handled here intentionally; you will manage that in TPU runner
    # (since torch_xla may not be installed in non-TPU environments).
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """
    Restore RNG states. Best-effort: missing fields are ignored.
    """
    if not state:
        return

    try:
        if "python_random_state" in state and state["python_random_state"] is not None:
            random.setstate(state["python_random_state"])
    except Exception:
        pass

    try:
        if "numpy_random_state" in state and state["numpy_random_state"] is not None:
            np.random.set_state(state["numpy_random_state"])
    except Exception:
        pass

    try:
        if "torch_rng_state" in state and state["torch_rng_state"] is not None:
            torch.set_rng_state(_cpu_tensor(state["torch_rng_state"]))
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            cuda_states = state.get("torch_cuda_rng_state_all", None)
            if cuda_states is not None:
                # ensure tensors are on CPU when passing in
                cuda_states = [_cpu_tensor(s) for s in cuda_states]
                torch.cuda.set_rng_state_all(cuda_states)
        except Exception:
            pass


def pack_split_indices(train_idx, val_idx) -> Dict[str, Any]:
    """
    Store split indices in checkpoint.
    We store as int64 numpy arrays for compactness and stability.
    """
    tr = _to_numpy_int64(train_idx)
    va = _to_numpy_int64(val_idx)
    if tr is None or va is None:
        raise ValueError("train_idx and val_idx must both be provided for TinyImageNet resume.")
    return {
        "train_idx": tr,
        "val_idx": va,
        "train_len": int(tr.shape[0]),
        "val_len": int(va.shape[0]),
    }


def unpack_split_indices(split_dict: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not split_dict:
        return None, None
    tr = split_dict.get("train_idx", None)
    va = split_dict.get("val_idx", None)
    if tr is None or va is None:
        return None, None
    return _to_numpy_int64(tr), _to_numpy_int64(va)


def save_checkpoint(
    path: str,
    *,
    epoch: int,
    best_val: float,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    split_dict: Optional[Dict[str, Any]] = None,
    rng_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save checkpoint to path. Caller controls rank (TPU master only).
    The checkpoint schema is intentionally explicit to avoid accidental mismatch.

    split_dict: output of pack_split_indices(...) OR None.
    rng_state: capture_rng_state() OR None.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    ckpt: Dict[str, Any] = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if (scheduler is not None and hasattr(scheduler, "state_dict")) else None,
        "split": split_dict,
        "rng": rng_state,
        "extra": extra or {},
        "format_version": 1,
    }

    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
    strict_model: bool = True,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    """
    Load checkpoint from path into provided model/optimizer/scheduler.
    Returns a dict containing:
      epoch, best_val, split (train_idx/val_idx), rng, extra, format_version.

    This is backward-compatible: missing keys are tolerated.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ckpt = torch.load(path, map_location=map_location)

    # Required
    epoch = int(ckpt.get("epoch", -1))
    best_val = float(ckpt.get("best_val", 0.0))

    # Model
    state = ckpt.get("model", None)
    if state is None:
        raise RuntimeError("Checkpoint missing 'model' state_dict.")
    model.load_state_dict(state, strict=strict_model)

    # Optimizer
    opt_state = ckpt.get("optimizer", None)
    if optimizer is not None and opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state)
        except Exception:
            # Don't hard fail resume; caller can decide
            pass

    # Scheduler
    sched_state = ckpt.get("scheduler", None)
    if scheduler is not None and sched_state is not None and hasattr(scheduler, "load_state_dict"):
        try:
            scheduler.load_state_dict(sched_state)
        except Exception:
            pass

    split_dict = ckpt.get("split", None)
    train_idx, val_idx = unpack_split_indices(split_dict or {})

    rng_state = ckpt.get("rng", None)
    if restore_rng and rng_state is not None:
        restore_rng_state(rng_state)

    out: Dict[str, Any] = {
        "epoch": epoch,
        "best_val": best_val,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "rng": rng_state,
        "extra": ckpt.get("extra", {}) or {},
        "format_version": int(ckpt.get("format_version", 0)),
    }
    return out
