import torch


def _sync_if_cuda(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize()
