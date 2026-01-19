#!/usr/bin/env python3
"""
ChebGate / ChebResNet TinyImageNet TPU training + evaluation + profiling entrypoint.
"""

import argparse
import inspect
import math
import os
import platform
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from core import (
    set_seed,
    ensure_logdir,
    write_json,
    parse_tuple_ints,
    amp_dtype_name,
    state_dict_uncompiled,
    _unwrap_compiled,
)
from core import append_csv_row
from core import load_state_dict_portable
from core import state_dict_sha256

from core.checkpoint import (
    load_checkpoint,
    pack_split_indices,
)

from data import (
    ensure_tinyimagenet_ready,
    get_tinyimagenet_train_labels,
    make_stratified_split_indices,
    build_tinyimagenet_datasets,
)

from model import ChebResNet
from training import train_epoch, evaluate
from metrics import (
    count_parameters,
    profile_macs,
    profile_macs_breakdown,
    save_learning_curve_csv,
    collect_gate_stats,
    dump_order_scales,
)


def snapshot_hardware_tpu(logdir: str, cfg, device_str: str, rank: int, world: int, epoch_seconds=None):
    info = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": device_str,
        "xla_rank": int(rank),
        "xla_world_size": int(world),
        "deterministic": bool(getattr(cfg, "deterministic", 0)),
        "amp": bool(getattr(cfg, "amp", 0)),
        "amp_dtype_train": amp_dtype_name(getattr(cfg, "amp_dtype_train", None)),
        "eval_amp": bool(getattr(cfg, "eval_amp", 0)),
        "amp_dtype_eval": amp_dtype_name(getattr(cfg, "amp_dtype_eval", None)),
        "batch_size": int(cfg.bs),
        "epochs": int(cfg.epochs),
        "dataset": "tinyimagenet",
        "realization": cfg.realization,
        "gate_mode": cfg.gate_mode,
        "lambda_lap": float(cfg.lambda_lap),
        "stabilize_cheb": bool(cfg.stabilize_cheb),
    }
    if epoch_seconds:
        info.update(
            {
                "epoch_time_mean_s": float(np.mean(epoch_seconds)),
                "epoch_time_median_s": float(np.median(epoch_seconds)),
                "epoch_time_minmax_s": [float(np.min(epoch_seconds)), float(np.max(epoch_seconds))],
            }
        )
    write_json(info, os.path.join(logdir, "hardware_env.json"))
    return info

# ----------------------------
# TPU worker
# ----------------------------

def _mp_fn(index, cfg):
    xm, xr, pl, _ = _xla_imports()

    device = xm.xla_device()
    rank, world = xr.global_ordinal(), xr.world_size()
    mprint = xm.master_print

    # Seed per rank (consistent with your reference TPU code)
    set_seed(int(cfg.seed) + int(rank))

    # Make sure data exists (locks/markers handle mp safety)
    ensure_tinyimagenet_ready(cfg.data)

    # Transforms (TinyImageNet 64x64)
    mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)

    class _Identity:
        def __call__(self, x):
            return x

    try:
        aug = T.RandAugment(3, 9)
    except Exception:
        aug = _Identity()

    tf_train = T.Compose(
        [
            aug,
            T.RandomCrop(64, padding=8, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    tf_eval = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    # Model cfg
    widths = parse_tuple_ints(cfg.widths)
    Ktup = parse_tuple_ints(cfg.K)
    depth = parse_tuple_ints(cfg.depth)
    classes = 200

    net = ChebResNet(
        classes=classes,
        K=Ktup,
        depth=depth,
        widths=widths,
        drop_rate=cfg.drop_rate,
        lap=cfg.lambda_lap,
        realization=cfg.realization,
        gate_mode=cfg.gate_mode,
        stabilize_cheb=bool(cfg.stabilize_cheb),
    ).to(device)
    net = net.to(memory_format=torch.channels_last)

    # Optimizer / sched / loss
    order_p, other_p = [], []
    for n, par in net.named_parameters():
        (order_p if "order_scales" in n else other_p).append(par)

    used_lr = cfg.lr * (cfg.bs / 128.0) if int(cfg.auto_lr) else cfg.lr
    if xm.is_master_ordinal():
        mprint(f"[opt] bs={cfg.bs} | base_lr={cfg.lr:.4f} | auto_lr={cfg.auto_lr} → used_lr={used_lr:.4f}")

    opt = torch.optim.SGD(
        [{"params": other_p}, {"params": order_p, "lr": used_lr * 0.1}],
        lr=used_lr,
        momentum=0.9,
        weight_decay=cfg.wd,
        nesterov=True,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AMP (TPU): we only pass dtype through to your train_epoch/evaluate;
    # your implementation decides whether/how to autocast on XLA.
    amp_dtype_train = torch.bfloat16 if int(cfg.amp) else None
    amp_dtype_eval = amp_dtype_train if (int(cfg.eval_amp) and amp_dtype_train is not None) else None
    cfg.amp_dtype_train = amp_dtype_train
    cfg.amp_dtype_eval = amp_dtype_eval
    scaler = None  # no GradScaler on TPU

    # Resume (checkpoint.pth stores split indices only inside checkpoint)
    ckpt_path = os.path.join(cfg.logdir, "checkpoint.pth")
    best_path = os.path.join(cfg.logdir, "best_model.pth")
    best_meta = os.path.join(cfg.logdir, "best_model_meta.json")

    start_ep = 0
    best_val_acc = -float("inf")
    best_epoch = -1
    train_idx = None
    val_idx = None

    if os.path.isfile(ckpt_path):
        try:
            info = load_checkpoint(
                ckpt_path,
                model=net,
                optimizer=opt,
                scheduler=sched,
                map_location="cpu",
                strict_model=True,
                restore_rng=False,  # rank-seeded determinism; avoid rank-mismatch restores
            )
            start_ep = int(info["epoch"]) + 1
            best_val_acc = float(info["best_val"])
            train_idx = info["train_idx"]
            val_idx = info["val_idx"]
            if xm.is_master_ordinal():
                mprint(f"[resume] Loaded {ckpt_path} → start_ep={start_ep}, best_val_acc={best_val_acc:.2f}")
        except Exception as e:
            if xm.is_master_ordinal():
                mprint(f"[resume] Failed to load checkpoint; starting fresh. Error: {e}")

    # Split indices
    if train_idx is None or val_idx is None:
        labels = get_tinyimagenet_train_labels(cfg.data)
        tr_idx, va_idx = make_stratified_split_indices(labels, seed=int(cfg.seed), val_frac=0.1)
        train_idx, val_idx = tr_idx, va_idx
        if xm.is_master_ordinal():
            mprint(f"[split] Created stratified split: train={len(train_idx)} val={len(val_idx)} (seed={cfg.seed})")

    # Datasets
    train_ds, val_ds, test_ds = build_tinyimagenet_datasets(
        cfg.data,
        train_idx=train_idx,
        val_idx=val_idx,
        tf_train=tf_train,
        tf_eval=tf_eval,
        use_val_as_test=True,
    )

    # Loaders (DistributedSampler + MpDeviceLoader)
    sam_tr = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=False)
    sam_va = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
    sam_te = DistributedSampler(test_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False)

    loader_kw = dict(pin_memory=True)
    dl_params = set(inspect.signature(DataLoader.__init__).parameters.keys())
    if cfg.workers > 0:
        loader_kw.update(num_workers=cfg.workers)
        if "persistent_workers" in dl_params:
            loader_kw.update(persistent_workers=bool(cfg.persistent_workers))
        if "prefetch_factor" in dl_params:
            loader_kw.update(prefetch_factor=cfg.prefetch)
    else:
        loader_kw.update(num_workers=0)

    tr_cpu = DataLoader(train_ds, batch_size=cfg.bs, sampler=sam_tr, shuffle=False, drop_last=False, **loader_kw)
    va_cpu = DataLoader(val_ds, batch_size=cfg.bs, sampler=sam_va, shuffle=False, drop_last=False, **loader_kw)
    te_cpu = DataLoader(test_ds, batch_size=cfg.bs, sampler=sam_te, shuffle=False, drop_last=False, **loader_kw)

    tr_loader = pl.MpDeviceLoader(tr_cpu, device)
    val_loader = pl.MpDeviceLoader(va_cpu, device)
    te_loader = pl.MpDeviceLoader(te_cpu, device)

    # Master-only: params/MACs snapshot (CPU profiling for stability)
    if xm.is_master_ordinal():
        net_cpu = ChebResNet(
            classes=classes,
            K=Ktup,
            depth=depth,
            widths=widths,
            drop_rate=cfg.drop_rate,
            lap=cfg.lambda_lap,
            realization=cfg.realization,
            gate_mode=cfg.gate_mode,
            stabilize_cheb=bool(cfg.stabilize_cheb),
        ).to(torch.device("cpu"))
        params = count_parameters(net_cpu, trainable_only=True)
        macs = profile_macs(net_cpu, input_size=(1, 3, 64, 64), device=torch.device("cpu"))
        flops = 2 * macs
        mprint(f"[S0] Params: {params/1e6:.3f}M | MACs@64x64: {macs/1e9:.3f}G | FLOPs≈{flops/1e9:.3f}G")
        write_json(
            {
                "params_trainable": int(params),
                "macs_64x64": int(macs),
                "flops_64x64": int(flops),
                "notes": "FLOPs ≈ 2×MACs; TinyImageNet uses 64×64 inputs.",
            },
            os.path.join(cfg.logdir, "params_flops.json"),
        )

    # Hardware snapshot baseline (master-only)
    if xm.is_master_ordinal():
        snapshot_hardware_tpu(cfg.logdir, cfg, str(device), rank, world)

    wall0 = time.time()
    epoch_seconds = []

    # Training loop
    for ep in range(start_ep, cfg.epochs):
        sam_tr.set_epoch(ep)

        t0 = time.time()
        tr_l, tr_a, tr_data_s, tr_comp_s = train_epoch(
            net, tr_loader, crit, opt, scaler, cfg, device, amp_dtype_train
        )
        sched.step()

        run_val = (ep % max(1, cfg.val_every) == 0)
        if run_val:
            va_l, va_a, _, _ = evaluate(net, val_loader, crit, device, amp_dtype_eval)
        else:
            va_l = float("nan")
            va_a = float("nan")

        epoch_sec = time.time() - t0
        epoch_seconds.append(epoch_sec)

        train_images = len(train_ds)
        imgs_per_s_epoch = (train_images / epoch_sec) if epoch_sec > 0 else float("inf")

        lr_show = opt.param_groups[0]["lr"]

        if xm.is_master_ordinal():
            mprint(
                f"Epoch {ep:03d} | Train L {tr_l:.3f} A {tr_a:.2f}% | "
                f"Val L {va_l:.3f} A {va_a:.2f}% | LR {lr_show:.6f} | "
                f"epoch_sec {epoch_sec:.2f}s | {imgs_per_s_epoch:.1f} img/s | "
                f"data {tr_data_s:.2f}s ({(tr_data_s/epoch_sec)*100:.1f}%) | "
                f"compute {tr_comp_s:.2f}s ({(tr_comp_s/epoch_sec)*100:.1f}%)"
            )

            # learning curve
            save_learning_curve_csv(
                cfg.logdir,
                {
                    "epoch": ep,
                    "train_loss": tr_l,
                    "train_acc": tr_a,
                    "val_loss": va_l,
                    "val_acc": va_a,
                    "lr": lr_show,
                    "epoch_seconds": epoch_sec,
                    "wall_seconds": time.time() - wall0,
                },
            )

            # efficiency (no power on TPU)
            data_frac = (tr_data_s / epoch_sec) if epoch_sec > 0 else 0.0
            comp_frac = (tr_comp_s / epoch_sec) if epoch_sec > 0 else 0.0

            # Save checkpoint each epoch (Option B: split in checkpoint only)
            split_dict = pack_split_indices(train_idx, val_idx)

            ckpt = {
                "epoch": int(ep),
                "best_val": float(best_val_acc),
                "model": _to_cpu_obj(state_dict_uncompiled(net)),
                "optimizer": _to_cpu_obj(opt.state_dict()),
                "scheduler": _to_cpu_obj(sched.state_dict()),
                "split": split_dict,
                "rng": None,
                "extra": {"dataset": "tinyimagenet"},
                "format_version": 1,
            }
            tmp = ckpt_path + ".tmp"
            torch.save(ckpt, tmp)
            os.replace(tmp, ckpt_path)

            # Best checkpoint on val
            if run_val and math.isfinite(va_a) and (va_a > best_val_acc):
                best_val_acc = float(va_a)
                best_epoch = int(ep)
                sd_best = _to_cpu_obj(state_dict_uncompiled(net))
                torch.save(sd_best, best_path)
                write_json(
                    {
                        "best_epoch": int(best_epoch),
                        "best_val_acc": float(best_val_acc),
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "lr_at_save": float(lr_show),
                    },
                    best_meta,
                )
                mprint(f"[best] New best Val Acc {best_val_acc:.2f}% at epoch {best_epoch} → saved {best_path}")

    # Final evaluation + latency sweeps (master-only)
    if xm.is_master_ordinal():
        if os.path.isfile(best_path):
            sd = torch.load(best_path, map_location="cpu")
            load_state_dict_portable(net, sd, strict=True)
            mprint(f"[best] Loaded best model from epoch {best_epoch} (Val Acc {best_val_acc:.2f}%)")
        else:
            mprint("[best] No best_model.pth found; evaluating current weights")

        te_l, te_a, _, _ = evaluate(net, te_loader, crit, device, amp_dtype_eval)
        mprint(f"\nFinal TINYIMAGENET Test Accuracy: {te_a:.2f}%")
        write_json({"test_loss": te_l, "test_acc": te_a}, os.path.join(cfg.logdir, "tinyimagenet_test_metrics.json"))

        # Gate stats + order scales (best-effort on XLA)
        try:
            collect_gate_stats(net, val_loader, device, cfg.logdir, amp_dtype=amp_dtype_eval)
        except Exception as e:
            mprint(f"[gate_stats] skipped on XLA: {e}")
        try:
            dump_order_scales(net, cfg.logdir)
        except Exception as e:
            mprint(f"[order_scales] dump skipped: {e}")

        # Finalize env snapshot with epoch times
        snapshot_hardware_tpu(cfg.logdir, cfg, str(device), rank, world, epoch_seconds)
        mprint("[E1] Hardware/env snapshot saved:", os.path.join(cfg.logdir, "hardware_env.json"))
        mprint("\n[DONE] All metrics & artifacts saved in:", cfg.logdir)


# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser()

    # Data / training (defaults set to your TPU TinyImageNet reference where applicable)
    p.add_argument("--data", default="./data")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--auto_lr", type=int, default=1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cut_alpha", type=float, default=1.0)
    p.add_argument("--drop_rate", type=float, default=0.2)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--eval_amp", type=int, default=1)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--clip_every", type=int, default=1)

    # Model knobs (same interface as CIFAR run.py)
    p.add_argument("--widths", type=str, default="192,384,768")
    p.add_argument("--K", type=str, default="3,5,5")
    p.add_argument("--depth", type=str, default="7,7,7")
    p.add_argument("--lambda_lap", type=float, default=0.25)
    p.add_argument("--realization", type=str, default="mstream", choices=["streamed", "concat", "gemm", "mstream"])
    p.add_argument("--gate_mode", type=str, default="on", choices=["on", "off"])
    p.add_argument("--stabilize_cheb", type=int, default=0)

    # Loader (map to your existing args; no extra TPU-only knobs)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=16)
    p.add_argument("--persistent_workers", type=int, default=1)

    # Eval cadence
    p.add_argument("--val_every", type=int, default=1)

    # Determinism flags (kept for config parity; TPU path does not use cuDNN)
    p.add_argument("--deterministic", type=int, default=1)

    # Logs
    p.add_argument("--logdir", default="./chebgate_logs_tiny")

    cfg, _ = p.parse_known_args()

    ensure_logdir(cfg.logdir)
    write_json(vars(cfg), os.path.join(cfg.logdir, "config_args.json"))

    # Ensure data is prepared once before spawning (faster first epoch)
    ensure_tinyimagenet_ready(cfg.data)

    # Clear any stray TPU env (consistent with your reference)
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)

    _, _, _, xmp = _xla_imports()
    xmp.spawn(_mp_fn, args=(cfg,), nprocs=None, start_method="fork")


if __name__ == "__main__":
    main()
