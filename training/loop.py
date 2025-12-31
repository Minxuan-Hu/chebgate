import time
import torch

from chebgate.core.sync import _sync_if_cuda
from .augment import cutmix, accuracy_mix


def train_epoch(net, loader, crit, opt, scaler, cfg, device, amp_dtype):
    """
    Returns: (loss, acc, data_time_s, compute_time_s)
      data_time_s   = loader wait + H2D copy
      compute_time_s= fwd + bwd + step (CUDA synchronized for accuracy)
    """
    net.train()
    tot_loss = tot_acc = tot_n = 0
    step = 0
    use_amp = bool(cfg.amp) and device.type == "cuda"
    opt.zero_grad(set_to_none=True)

    data_time = 0.0
    compute_time = 0.0
    last_iter_end = time.perf_counter()

    for xb, yb in loader:
        # measure loader wait until this batch is yielded
        t_yield = time.perf_counter()
        data_time += (t_yield - last_iter_end)

        # H2D copy (synchronized to account for time)
        _sync_if_cuda(device)
        t_copy0 = time.perf_counter()
        xb = xb.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)
        _sync_if_cuda(device)
        t_copy1 = time.perf_counter()
        data_time += (t_copy1 - t_copy0)

        xm, y1, y2, l1, l2 = cutmix(xb, yb, cfg.cut_alpha)

        # compute (forward/backward/step), synchronized
        _sync_if_cuda(device)
        t_comp0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = net(xm)
            loss = (l1 * crit(out, y1) + l2 * crit(out, y2)) / max(1, cfg.accum_steps)

        if not torch.isfinite(loss):
            print("[warn] non-finite loss detected; skipping step")
            opt.zero_grad(set_to_none=True)
            _sync_if_cuda(device)
            t_comp1 = time.perf_counter()
            compute_time += (t_comp1 - t_comp0)
            last_iter_end = time.perf_counter()
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        step += 1

        if step % cfg.accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(opt)
            if cfg.clip_every <= 1 or (step // cfg.accum_steps) % cfg.clip_every == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            if scaler is not None:
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        _sync_if_cuda(device)
        t_comp1 = time.perf_counter()
        compute_time += (t_comp1 - t_comp0)

        bs = xb.size(0)
        tot_loss += (loss.detach() * max(1, cfg.accum_steps)).item() * bs
        tot_acc += accuracy_mix(out, y1, y2, l1, l2) * bs
        tot_n += bs

        last_iter_end = time.perf_counter()

    # flush remainder step if any (compute-only)
    if step % max(1, cfg.accum_steps) != 0:
        _sync_if_cuda(device)
        t_comp0 = time.perf_counter()
        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        if scaler is not None:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        _sync_if_cuda(device)
        t_comp1 = time.perf_counter()
        compute_time += (t_comp1 - t_comp0)

    return tot_loss / tot_n, tot_acc / tot_n, data_time, compute_time


@torch.no_grad()
def evaluate(net, loader, crit, device, amp_dtype):
    net.eval()
    loss = acc = 0
    use_amp = (amp_dtype is not None and device.type == "cuda")
    data_time = 0.0
    compute_time = 0.0
    last_iter_end = time.perf_counter()

    for xb, yb in loader:
        t_yield = time.perf_counter()
        data_time += (t_yield - last_iter_end)

        _sync_if_cuda(device)
        t_copy0 = time.perf_counter()
        xb = xb.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)
        _sync_if_cuda(device)
        t_copy1 = time.perf_counter()
        data_time += (t_copy1 - t_copy0)

        _sync_if_cuda(device)
        t_comp0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = net(xb)
            loss += crit(out, yb).item() * xb.size(0)
        _sync_if_cuda(device)
        t_comp1 = time.perf_counter()
        compute_time += (t_comp1 - t_comp0)

        acc += out.argmax(1).eq(yb).sum().item()
        last_iter_end = time.perf_counter()

    N = len(loader.dataset)
    return loss / N, 100.0 * acc / N, data_time, compute_time
