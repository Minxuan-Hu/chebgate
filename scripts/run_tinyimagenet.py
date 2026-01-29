#!/usr/bin/env python3

import argparse
import os
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from model import ChebResNet

os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker_init_fn(base_seed: int, rank: int):
    def _init(worker_id: int):
        s = int(base_seed) + int(rank) * 1000 + int(worker_id)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    return _init


# ----------------------------
# Tiny-ImageNet utilities
# ----------------------------
def ensure_tinyimagenet_ready(data_root: str, log_fn=None) -> str:
    ds_root = os.path.join(data_root, "tiny-imagenet-200")
    if os.path.isdir(ds_root):
        return ds_root

    os.makedirs(data_root, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")

    if log_fn:
        log_fn(f"[data] Downloading Tiny-ImageNet from {url} -> {zip_path}")
    else:
        print(f"[data] Downloading Tiny-ImageNet -> {zip_path}", flush=True)

    urllib.request.urlretrieve(url, zip_path)

    if log_fn:
        log_fn(f"[data] Extracting {zip_path} -> {data_root}")
    else:
        print(f"[data] Extracting...", flush=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)

    try:
        os.remove(zip_path)
    except Exception:
        pass

    return ds_root


def ensure_val_reorganized(ds_root: str, log_fn=None) -> None:
    marker = os.path.join(ds_root, "val", ".reorg_done")
    if os.path.isfile(marker):
        return

    vdir = os.path.join(ds_root, "val")
    imgs_dir = os.path.join(vdir, "images")
    ann = os.path.join(vdir, "val_annotations.txt")
    if (not os.path.isdir(imgs_dir)) or (not os.path.isfile(ann)):
        raise RuntimeError(f"Unexpected Tiny-ImageNet val structure under: {vdir}")

    if log_fn:
        log_fn("[data] Reorganizing val/ images by class...")
    else:
        print("[data] Reorganizing val/ images by class...", flush=True)

    moved = 0
    with open(ann, "r") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            img, cls = parts[0], parts[1]
            src = os.path.join(imgs_dir, img)
            if not os.path.exists(src):
                continue
            dst_dir = os.path.join(imgs_dir, cls)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, os.path.join(dst_dir, img))
            moved += 1

    with open(marker, "w") as f:
        f.write(f"moved={moved}\n")


def stratified_split_indices(labels: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    n = labels.shape[0]

    train_idx = []
    val_idx = []
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        nv = int(round(len(idx_c) * float(val_frac)))
        nv = max(1, nv)
        val_idx.append(idx_c[:nv])
        train_idx.append(idx_c[nv:])

    train_idx = np.concatenate(train_idx) if len(train_idx) else np.arange(n)
    val_idx = np.concatenate(val_idx) if len(val_idx) else np.array([], dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


# ----------------------------
# MixUp / CutMix on CPU (collate_fn)
# ----------------------------
def _as_label_tensor(labels):
    if torch.is_tensor(labels[0]):
        return torch.stack([lab.view(()) for lab in labels]).long()
    return torch.tensor(labels, dtype=torch.long)


def mixup_cpu(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha is None or float(alpha) <= 0.0:
        return x, y, y, 1.0, 0.0
    lam = np.random.beta(alpha, alpha)
    B = x.size(0)
    perm = torch.randperm(B)
    x2 = x[perm]
    y2 = y[perm]
    x_mix = lam * x + (1.0 - lam) * x2
    return x_mix, y, y2, float(lam), float(1.0 - lam)


def cutmix_cpu(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha is None or float(alpha) <= 0.0:
        return x, y, y, 1.0, 0.0

    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.shape
    perm = torch.randperm(B)
    y1, y2 = y, y[perm]

    cut_rat = np.sqrt(1.0 - lam)
    ch, cw = int(H * cut_rat), int(W * cut_rat)
    cy, cx = np.random.randint(H), np.random.randint(W)

    y1_i, x1_i = max(cy - ch // 2, 0), max(cx - cw // 2, 0)
    y2_i, x2_i = min(cy + ch // 2, H), min(cx + cw // 2, W)

    x = x.clone()
    x[:, :, y1_i:y2_i, x1_i:x2_i] = x[perm, :, y1_i:y2_i, x1_i:x2_i]

    lam_adj = 1.0 - ((y2_i - y1_i) * (x2_i - x1_i) / (H * W))
    lam1 = float(lam_adj)
    lam2 = float(1.0 - lam1)
    return x, y1, y2, lam1, lam2


def make_mix_collate(mix_alpha: float, cut_alpha: float, cut_prob: float):
    mix_alpha = float(mix_alpha)
    cut_alpha = float(cut_alpha)
    cut_prob = float(cut_prob)

    def collate(batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)
        y = _as_label_tensor(ys)

        if (mix_alpha <= 0.0) and (cut_alpha <= 0.0):
            return x, y, y, 1.0, 0.0

        use_cut = (cut_alpha > 0.0) and (random.random() < cut_prob)
        if use_cut:
            return cutmix_cpu(x, y, cut_alpha)
        else:
            return mixup_cpu(x, y, mix_alpha)

    return collate


@torch.no_grad()
def accuracy_mix(logits: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, lam1: float, lam2: float) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    c1 = (pred == y1).to(torch.float32)
    c2 = (pred == y2).to(torch.float32)
    return float(lam1) * c1.mean() + float(lam2) * c2.mean()


# ----------------------------
# Checkpoint helpers
# ----------------------------
def _to_cpu_obj(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_to_cpu_obj(v) for v in obj]
        return type(obj)(t) if isinstance(obj, tuple) else t
    return obj


def _atomic_torch_save(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


# ----------------------------
# XLA-safe autocast
# ----------------------------
def _safe_autocast_xla(dtype: Optional[torch.dtype]):
    if dtype is None:
        return torch.autocast("cpu", enabled=False)
    try:
        return torch.autocast(device_type="xla", dtype=dtype)
    except Exception:
        return torch.autocast("cpu", enabled=False)


# ----------------------------
# Eval / Train
# ----------------------------
@torch.no_grad()
def evaluate_xla(model: nn.Module, loader: Iterable, crit: nn.Module, device, amp_dtype: Optional[torch.dtype]):
    import torch_xla.core.xla_model as xm

    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.float32)
    total = torch.zeros((), device=device, dtype=torch.float32)

    for xb, yb in loader:
        with _safe_autocast_xla(amp_dtype):
            logits = model(xb)
            loss = crit(logits, yb)

        bs = float(yb.numel())
        loss_sum += loss.detach() * bs
        pred = logits.argmax(dim=1)
        correct += (pred == yb).to(torch.float32).sum()
        total += bs
        xm.mark_step()

    vec = torch.stack([loss_sum, correct, total])
    vec = xm.all_reduce(xm.REDUCE_SUM, vec)
    loss_avg = (vec[0] / vec[2]).item() if vec[2].item() > 0 else float("nan")
    acc = (vec[1] / vec[2] * 100.0).item() if vec[2].item() > 0 else float("nan")
    return loss_avg, acc


def train_epoch_xla(
    model: nn.Module,
    loader: Iterable,
    crit: nn.Module,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler._LRScheduler,
    device,
    amp_dtype: Optional[torch.dtype],
    log0,
    log_every: int,
):
    import torch_xla.core.xla_model as xm

    model.train()
    loss_sum = torch.zeros((), device=device)
    correct_sum = torch.zeros((), device=device, dtype=torch.float32)
    total = torch.zeros((), device=device, dtype=torch.float32)

    steps = 0
    for batch in loader:
        xb, y1, y2, lam1, lam2 = batch
        lam1f, lam2f = float(lam1), float(lam2)

        opt.zero_grad(set_to_none=True)

        with _safe_autocast_xla(amp_dtype):
            logits = model(xb)
            loss = lam1f * crit(logits, y1) + lam2f * crit(logits, y2)

        loss.backward()
        xm.optimizer_step(opt, barrier=True)
        sched.step()

        bs = float(y1.numel())
        loss_sum += loss.detach() * bs
        correct_sum += (accuracy_mix(logits, y1, y2, lam1f, lam2f) * bs)
        total += bs

        steps += 1
        xm.mark_step()

        if log_every > 0 and (steps % log_every == 0):
            lr_show = opt.param_groups[0]["lr"]
            log0(f"  step {steps:05d} | loss {loss.item():.4f} | lr {lr_show:.6g} | lam1 {lam1f:.3f}")

    vec = torch.stack([loss_sum, correct_sum, total])
    vec = xm.all_reduce(xm.REDUCE_SUM, vec)
    loss_avg = (vec[0] / vec[2]).item() if vec[2].item() > 0 else float("nan")
    acc = (vec[1] / vec[2] * 100.0).item() if vec[2].item() > 0 else float("nan")
    return loss_avg, acc, steps


# ----------------------------
# TPU worker
# ----------------------------
def _mp_fn(index, cfg):
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.parallel_loader as pl

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    device = torch_xla.device()
    rank, world = xr.global_ordinal(), xr.world_size()
    is_rank0 = (rank == 0)

    os.makedirs(cfg.logdir, exist_ok=True)
    rank0_log_f = None
    if is_rank0:
        rank0_log_f = open(os.path.join(cfg.logdir, "stdout_rank0.log"), "a", buffering=1)

    def log0(msg: str) -> None:
        if is_rank0:
            print(msg, flush=True)
            if rank0_log_f is not None:
                rank0_log_f.write(msg + "\n")
                rank0_log_f.flush()

    set_seed(int(cfg.seed) + int(rank))

    if is_rank0:
        ds_root = ensure_tinyimagenet_ready(cfg.data, log_fn=log0)
    xm.rendezvous("tinyimagenet_downloaded")

    ds_root = os.path.join(cfg.data, "tiny-imagenet-200")
    if is_rank0:
        ensure_val_reorganized(ds_root, log_fn=log0)
    xm.rendezvous("tinyimagenet_val_reorg")

    if is_rank0:
        log0(f"[TPU] PJRT_DEVICE={os.environ.get('PJRT_DEVICE','')} | world_size={world} | device={device}")
        log0(
            f"[DL] workers={cfg.num_workers} | persistent_workers={cfg.persistent_workers} | "
            f"prefetch_factor={cfg.prefetch_factor} | pin_memory={cfg.pin_memory}"
        )
        log0(
            f"[Mix] mix_alpha={cfg.mix_alpha} | cut_alpha={cfg.cut_alpha} | cut_prob={cfg.cut_prob} "
            f"(CPU collate; XLA RNG avoided)"
        )
        log0(
            f"[Model] realization={cfg.realization} | gate_mode={cfg.gate_mode} | "
            f"lambda_lap={cfg.lambda_lap} | stabilize_cheb={cfg.stabilize_cheb} | drop_rate={cfg.drop_rate}"
        )

    # Transforms
    mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
    try:
        ra = T.RandAugment(int(cfg.ra_n), int(cfg.ra_m))
    except Exception:
        ra = nn.Identity()

    tf_train = T.Compose(
        [
            ra,
            T.RandomCrop(64, padding=8, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ]
    )
    tf_eval = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    # Build datasets + split (resume uses saved indices)
    train_root = os.path.join(ds_root, "train")
    val_images_root = os.path.join(ds_root, "val", "images")

    full_for_labels = torchvision.datasets.ImageFolder(train_root, transform=None)
    labels = np.array([y for _, y in full_for_labels.samples], dtype=np.int64)

    ckpt_path = os.path.join(cfg.logdir, "checkpoint.pth")
    best_path = os.path.join(cfg.logdir, "best_model.pth")

    start_ep = 0
    best_val = -float("inf")
    train_idx = None
    val_idx = None

    # ----------------------------
    # Model construction
    # ----------------------------
    net = ChebResNet(
        classes=200,
        K=tuple(cfg.K),
        depth=tuple(cfg.depth),
        widths=tuple(cfg.widths),
        drop_rate=float(cfg.drop_rate),
        lap=float(cfg.lambda_lap),
        realization=str(cfg.realization),
        gate_mode=str(cfg.gate_mode),
        stabilize_cheb=bool(int(cfg.stabilize_cheb)),
    ).to(device)

    if int(cfg.channels_last):
        net = net.to(memory_format=torch.channels_last)

    # Optimizer / loss / scheduler
    opt = torch.optim.SGD(
        net.parameters(),
        lr=float(cfg.lr),
        momentum=0.9,
        weight_decay=float(cfg.wd),
        nesterov=True,
    )
    crit = nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))

    # Datasets need split indices; try load checkpoint early for split recovery
    ckpt = load_checkpoint_if_exists(ckpt_path)
    if ckpt is not None:
        try:
            train_idx = ckpt.get("train_idx", None)
            val_idx = ckpt.get("val_idx", None)
            start_ep = int(ckpt.get("epoch", -1)) + 1
            best_val = float(ckpt.get("best_val", -float("inf")))
            if is_rank0:
                log0(f"[resume] Found {ckpt_path} -> start_ep={start_ep}, best_val={best_val:.2f}")
        except Exception as e:
            if is_rank0:
                log0(f"[resume] Failed to parse checkpoint meta; will ignore: {e}")
            ckpt = None

    if train_idx is None or val_idx is None:
        tr_idx, va_idx = stratified_split_indices(labels, val_frac=float(cfg.val_frac), seed=int(cfg.seed))
        train_idx, val_idx = tr_idx, va_idx
        if is_rank0:
            log0(f"[split] Stratified split: train={len(train_idx)} val={len(val_idx)} (seed={cfg.seed})")

    train_full = torchvision.datasets.ImageFolder(train_root, transform=tf_train)
    eval_full = torchvision.datasets.ImageFolder(train_root, transform=tf_eval)

    tr_ds = Subset(train_full, train_idx)
    va_ds = Subset(eval_full, val_idx)
    te_ds = torchvision.datasets.ImageFolder(val_images_root, transform=tf_eval)

    sam_tr = DistributedSampler(tr_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=bool(cfg.drop_last))
    sam_va = DistributedSampler(va_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
    sam_te = DistributedSampler(te_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False)

    dl_kw: Dict[str, Any] = dict(
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        drop_last=bool(cfg.drop_last),
        worker_init_fn=_worker_init_fn(int(cfg.seed), int(rank)),
    )
    if int(cfg.num_workers) > 0:
        dl_kw["persistent_workers"] = bool(cfg.persistent_workers)
        dl_kw["prefetch_factor"] = int(cfg.prefetch_factor)

    tr_collate = make_mix_collate(cfg.mix_alpha, cfg.cut_alpha, cfg.cut_prob)

    tr_cpu = DataLoader(tr_ds, batch_size=int(cfg.bs), sampler=sam_tr, shuffle=False, collate_fn=tr_collate, **dl_kw)
    va_cpu = DataLoader(va_ds, batch_size=int(cfg.bs), sampler=sam_va, shuffle=False, **dl_kw)
    te_cpu = DataLoader(te_ds, batch_size=int(cfg.bs), sampler=sam_te, shuffle=False, **dl_kw)

    tr_loader = pl.MpDeviceLoader(tr_cpu, device)
    va_loader = pl.MpDeviceLoader(va_cpu, device)
    te_loader = pl.MpDeviceLoader(te_cpu, device)

    steps_per_epoch = len(tr_cpu)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=float(cfg.lr),
        epochs=int(cfg.epochs),
        steps_per_epoch=int(steps_per_epoch),
        pct_start=float(cfg.pct_start),
        anneal_strategy="cos",
    )

    if ckpt is not None:
        try:
            net.load_state_dict(ckpt["model"], strict=True)
            if ckpt.get("opt") is not None:
                opt.load_state_dict(ckpt["opt"])
            if ckpt.get("sched") is not None:
                sched.load_state_dict(ckpt["sched"])
            if is_rank0:
                log0("[resume] Loaded model/opt/sched state successfully.")
        except Exception as e:
            if is_rank0:
                log0(f"[resume] State load failed (will continue from scratch): {e}")
            start_ep = 0
            best_val = -float("inf")

    # Broadcast model params so all ranks start identical
    try:
        xm.broadcast_master_param(net)
    except Exception:
        pass

    amp_dtype = torch.bfloat16 if int(cfg.amp) else None

    wall0 = time.time()
    for ep in range(start_ep, int(cfg.epochs)):
        sam_tr.set_epoch(ep)

        if is_rank0:
            log0(f"=== Epoch {ep+1}/{int(cfg.epochs)} begin ===")
            log0(f"[epoch] steps_per_epoch={steps_per_epoch} | bs={cfg.bs} | world={world}")

        t0 = time.time()
        tr_loss, tr_acc, _ = train_epoch_xla(
            net,
            tr_loader,
            crit,
            opt,
            sched,
            device=device,
            amp_dtype=amp_dtype,
            log0=log0,
            log_every=int(cfg.log_steps),
        )
        epoch_sec = time.time() - t0

        do_val = (ep % max(1, int(cfg.val_every)) == 0)
        if do_val:
            va_loss, va_acc = evaluate_xla(net, va_loader, crit, device=device, amp_dtype=amp_dtype)
        else:
            va_loss, va_acc = float("nan"), float("nan")

        if is_rank0:
            lr_show = opt.param_groups[0]["lr"]
            train_images = len(tr_ds)
            imgs_per_s = (train_images / epoch_sec) if epoch_sec > 0 else float("inf")
            log0(
                f"[Ep {ep:03d}] train L {tr_loss:.3f} A {tr_acc:.2f}% | "
                f"val L {va_loss:.3f} A {va_acc:.2f}% | "
                f"lr {lr_show:.6g} | epoch {epoch_sec:.2f}s | {imgs_per_s:.1f} img/s"
            )

            ckpt_obj = {
                "epoch": int(ep),
                "best_val": float(best_val),
                "model": _to_cpu_obj(net.state_dict()),
                "opt": _to_cpu_obj(opt.state_dict()),
                "sched": _to_cpu_obj(sched.state_dict()),
                "train_idx": _to_cpu_obj(train_idx),
                "val_idx": _to_cpu_obj(val_idx),
                "format_version": 1,
                "extra": {"dataset": "tinyimagenet", "world_size": int(world)},
                "model_cfg": {
                    "K": list(cfg.K),
                    "depth": list(cfg.depth),
                    "widths": list(cfg.widths),
                    "lambda_lap": float(cfg.lambda_lap),
                    "realization": str(cfg.realization),
                    "gate_mode": str(cfg.gate_mode),
                    "stabilize_cheb": int(cfg.stabilize_cheb),
                    "drop_rate": float(cfg.drop_rate),
                },
            }
            _atomic_torch_save(ckpt_obj, ckpt_path)

            if do_val and np.isfinite(va_acc) and float(va_acc) > float(best_val):
                best_val = float(va_acc)
                _atomic_torch_save(_to_cpu_obj(net.state_dict()), best_path)
                log0(f"[best] New best val acc {best_val:.2f}% -> {best_path}")

        xm.rendezvous(f"epoch_{ep}_done")

    xm.rendezvous("train_done")

    if os.path.isfile(best_path):
        if is_rank0:
            sd = torch.load(best_path, map_location="cpu", weights_only=False)
            net.load_state_dict(sd, strict=True)
            log0(f"[best] Loaded best model (best_val={best_val:.2f}%)")
        xm.rendezvous("best_loaded")
        try:
            xm.broadcast_master_param(net)
        except Exception:
            if not is_rank0:
                sd = torch.load(best_path, map_location="cpu", weights_only=False)
                net.load_state_dict(sd, strict=True)
    else:
        xm.rendezvous("best_loaded")

    @torch.no_grad()
    def evaluate_local_xla(model, loader, crit, device, amp_dtype):
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
        model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with _safe_autocast_xla(amp_dtype):
                out = model(xb)
                loss = crit(out, yb)
            bs = xb.size(0)
            loss_sum += float(loss.item()) * bs
            correct += int(out.argmax(1).eq(yb).sum().item())
            total += int(bs)
            xm.mark_step()
        return loss_sum / max(1, total), 100.0 * correct / max(1, total)

    xm.rendezvous("before_final_test")
    
    if is_rank0:
        # IMPORTANT: no DistributedSampler, no MpDeviceLoader, no all_reduce
        test_ds = torchvision.datasets.ImageFolder(val_images_root, transform=tf_eval)
        test_loader = DataLoader(
            test_ds,
            batch_size=int(cfg.bs),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        te_loss, te_acc = evaluate_local_xla(net, test_loader, crit, device, amp_dtype)
        log0(f"\nFinal held-out (val/) -> loss {te_loss:.3f} acc {te_acc:.2f}%")
    
    xm.rendezvous("after_final_test")


# ----------------------------
# Main / CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data", default="/data")
    p.add_argument("--logdir", default="/cheb_logs_tiny")

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)

    # Model hyperparams
    p.add_argument("--K", type=int, nargs=3, default=[3,5,5])
    p.add_argument("--depth", type=int, nargs=3, default=[7,7,7])
    p.add_argument("--widths", type=int, nargs=3, default=[224,448,896])
    p.add_argument("--channels_last", type=int, default=0)

    # Model-architecture knobs
    p.add_argument("--lambda_lap", type=float, default=0.25)
    p.add_argument("--realization", type=str, default="concat", choices=["streamed", "concat", "gemm", "mstream"])
    p.add_argument("--gate_mode", type=str, default="on", choices=["on", "off"])
    p.add_argument("--stabilize_cheb", type=int, default=0)
    p.add_argument("--drop_rate", type=float, default=0.15)

    # MixUp / CutMix
    p.add_argument("--mix_alpha", type=float, default=0.0)
    p.add_argument("--cut_alpha", type=float, default=1.0)
    p.add_argument("--cut_prob", type=float, default=0.5)

    # OneCycleLR
    p.add_argument("--pct_start", type=float, default=0.1)

    # Augment knobs
    p.add_argument("--ra_n", type=int, default=3)
    p.add_argument("--ra_m", type=int, default=9)

    # Train/eval knobs
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--log_steps", type=int, default=400)

    # DataLoader knobs
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=16)
    p.add_argument("--pin_memory", type=int, default=0)
    p.add_argument("--drop_last", action="store_true")

    # AMP (bf16)
    p.add_argument("--amp", type=int, default=0)

    # Multiprocessing
    p.add_argument("--mp_start", type=str, default="fork", choices=["spawn", "fork"])

    cfg, _ = p.parse_known_args()
    os.makedirs(cfg.logdir, exist_ok=True)

    # IMPORTANT for Kaggle v5e-8 (PJRT): set before importing torch_xla in workers
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)
    os.environ["PJRT_DEVICE"] = "TPU"

    print(f"[launch] PJRT_DEVICE={os.environ.get('PJRT_DEVICE','')} | mp_start={cfg.mp_start}", flush=True)
    print(f"[launch] logdir={cfg.logdir}", flush=True)

    import json
    with open(os.path.join(cfg.logdir, "config_args.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2, sort_keys=True)

    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(_mp_fn, args=(cfg,), nprocs=None, start_method=cfg.mp_start)


if __name__ == "__main__":
    main()
