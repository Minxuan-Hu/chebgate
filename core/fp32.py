from contextlib import contextmanager
import torch


@contextmanager
def fp32_reference_mode(strict_tf32_off: bool = True, disable_cudnn_benchmark: bool = True):
    """
    Context for strict FP32 reference checks:
      - disables TF32 (matmul + cuDNN) when strict_tf32_off=True
      - sets float32 matmul precision to 'highest' when available
      - optionally disables cudnn.benchmark to reduce kernel variability

    Restores all changed global flags on exit.
    """
    cuda_ok = torch.cuda.is_available()
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32 if cuda_ok else None
    old_cudnn_tf32  = torch.backends.cudnn.allow_tf32 if cuda_ok else None
    old_bench = torch.backends.cudnn.benchmark

    old_prec = None
    has_get = hasattr(torch, "get_float32_matmul_precision")
    has_set = hasattr(torch, "set_float32_matmul_precision")
    if has_get:
        try:
            old_prec = torch.get_float32_matmul_precision()
        except Exception:
            old_prec = None

    try:
        if cuda_ok and strict_tf32_off:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if has_set:
                try:
                    torch.set_float32_matmul_precision("highest")
                except Exception:
                    pass

        if disable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = False

        yield

    finally:
        if cuda_ok:
            if old_matmul_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            if old_cudnn_tf32 is not None:
                torch.backends.cudnn.allow_tf32 = old_cudnn_tf32

        torch.backends.cudnn.benchmark = old_bench

        if old_prec is not None and has_set:
            try:
                torch.set_float32_matmul_precision(old_prec)
            except Exception:
                pass
