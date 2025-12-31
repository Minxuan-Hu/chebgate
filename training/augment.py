import numpy as np
import torch


def cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0, 0.0

    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    y1, y2 = y, y[perm]

    H, W = x.size(2), x.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    ch, cw = int(H * cut_rat), int(W * cut_rat)
    cy, cx = np.random.randint(H), np.random.randint(W)

    y1_i, x1_i = max(cy - ch // 2, 0), max(cx - cw // 2, 0)
    y2_i, x2_i = min(cy + ch // 2, H), min(cx + cw // 2, W)

    x[:, :, y1_i:y2_i, x1_i:x2_i] = x[perm, :, y1_i:y2_i, x1_i:x2_i]

    # Corrected lam_adj (keep only the correct formula)
    lam_adj = 1 - ((y2_i - y1_i) * (x2_i - x1_i) / (H * W))
    return x, y1, y2, lam_adj, 1 - lam_adj


@torch.no_grad()
def accuracy_mix(out: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, lam1: float, lam2: float):
    pred = out.argmax(1)
    m1 = (pred == y1).float()
    m2 = (pred == y2).float()
    return float(((lam1 * m1 + lam2 * m2).mean() * 100).item())
