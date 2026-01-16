#!/usr/bin/env python3
"""
ChebGate / ChebResNet TinyImageNet TPU training entrypoint.
"""

import argparse
import os
import platform
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from chebgate.core import (
    set_seed,
    ensure_logdir,
    write_json,
    append_csv_row,
    parse_tuple_ints,
    amp_dtype_name,
    load_state_dict_portable,
    strip_orig_mod_prefix,
    state_dict_sha256,
)
from chebgate.core.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    pack_split_indices,
    unpack_split_indices,
)
from chebgate.data import (
    ensure_tinyimagenet_ready,
    get_tinyimagenet_train_labels,
    make_stratified_split_indices,
    build_tinyimagenet_datasets,
)
from chebgate.model import ChebResNet
from chebgate.metrics import (
    count_parameters,
    profile_macs,
    save_learning_curve_csv,
    dump_order_scales,
)


# ----------------------------
# MixUp / CutMix (CPU-random, works on XLA tensors)
# ----------------------------

def mixup_cpu(x, y, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0, 0.0
    lam = random.betavariate(alpha, alpha)
    perm = torch.randperm(x.size(0), device="cpu").to(x.device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam, 1.0 - lam


def cutmix_cpu(x, y, alpha: float):
    lam = random.betavariate(alpha, alpha)
    perm = torch.randperm(x.size(0), device="cpu").to(x.device)
    y1, y2 = y, y[perm]
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cw, ch = int(W * cut_rat), int(H * cut_rat)
    cx, cy = random.randrange(W), random.randrange(H)
    bbx1, bby1 = max(0, cx - cw // 2), max(0, cy - ch // 2)
    bbx2, bby2 = min(W, cx + cw // 2), min(H, cy + ch // 2)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[perm, :, bbx1:bbx2, bby1:bby2]
    lam_adj = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / float(W * H))
    return x, y1, y2, lam_adj, 1.0 - lam_adj


def _snapshot_env_xla(logdir: str, cfg, device_str: str, world: int, rank: int) -> Dict[str, object]:
    info: Dict[str, object] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": device_str,
        "world_size": int(world),
        "rank": int(rank),
        "dataset": "tinyimagenet",
        "epochs": int(cfg.epochs),
        "batch_size_per_core": int(cfg.bs),
        "lr": float(cfg.lr),
        "wd": float(cfg.wd),
        "mix_alpha": float(cfg.mix_alpha),
        "cut_alpha": float(cfg.cut_alpha),
        "cut_prob": float(cfg.cut_prob),
        "label_smoothing": float(cfg.label_smoothing),
        "model": {
            "classes": 200,
            "widths": cfg.widths,
            "K": cfg.K,
            "depth": cfg.depth,
            "drop_rate": float(cfg.drop_rate),
            "lambda_lap": float(cfg.lambda_lap),
            "realization": cfg.realization,
            "gate_mode": cfg.gate_mode,
            "stabilize_cheb": bool(cfg.stabilize_cheb),
        },
        "loader": {
            "num_workers": int(cfg.num_workers),
            "drop_last": bool(cfg.drop_last),
            "persistent_workers": bool(cfg.persistent_workers),
            "prefetch_factor": int(cfg.prefetch_factor),
            "loader_prefetch_size": int(cfg.loader_prefetch_size),
            "device_prefetch_size": int(cfg.device_prefetch_size),
            "host_to_device_transfer_threads": int(cfg.host_to_device_transfer_threads),
        },
        "latency": {
            "measure_latency": int(cfg.measure_latency),
            "lat_warmup": int(cfg.lat_warmup),
            "lat_iters": int(cfg.lat_iters),
            "latency_all_realizations": int(cfg.latency_all_realizations),
            "latency_all_max_bs": int(cfg.latency_all_max_bs),
            "latency_all_realizations_list": list(_realization_list()),
        },
    }

    try:
        import torch_xla
        info["torch_xla"] = getattr(torch_xla, "__version__", "unknown")
    except Exception:
        info["torch_xla"] = None

    write_json(info, os.path.join(logdir, "hardware_env.json"))
    return info


def _try_params_macs_snapshot(logdir: str, model_cfg: Dict[str, object]) -> None:
    """
    Best-effort params/MACs snapshot on CPU for stability.
    """
    try:
        net_cpu = ChebResNet(**model_cfg).to(torch.device("cpu"))
        params = count_parameters(net_cpu, trainable_only=True)

        macs = None
        try:
            macs = profile_macs(net_cpu, input_size=(1, 3, 64, 64), device=torch.device("cpu"))
        except Exception:
            macs = None

        write_json(
            {
                "params_trainable": int(params),
                "macs_64x64": int(macs) if macs is not None else None,
                "flops_64x64": int(2 * macs) if macs is not None else None,
                "notes": "TinyImageNet uses 64x64 inputs. FLOPs ≈ 2×MACs (approx).",
            },
            os.path.join(logdir, "params_flops.json"),
        )
    except Exception as e:
        write_json({"error": str(e)}, os.path.join(logdir, "params_flops_error.json"))


def _realization_list() -> List[str]:
    return ["concat", "streamed", "mstream", "gemm"]


def _build_bs_list(max_bs: int) -> List[int]:
    base = [1, 2, 4, 8, 16, 32, 64, 128]
    if max_bs >= 256:
        base.append(256)
    return base


def _xla_latency_ms_samples(
    net,
    device,
    xm,
    shape: Tuple[int, int, int, int],
    iters: int = 200,
    warmup: int = 50,
) -> Dict[str, float]:
    """
    TPU/XLA latency distribution (CIFAR-like):
    - Warm up to amortize compile and cache effects
    - Then time each iteration with device sync to get per-iter samples
    """
    bs = int(shape[0])
    x = torch.randn(*shape, device=device)
    times_ms: List[float] = []

    net.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(int(warmup)):
            _ = net(x)
            xm.mark_step()
        try:
            xm.wait_device_ops()
        except Exception:
            pass

        # Timed iters (sync per iter for consistent per-sample timing)
        for _ in range(int(iters)):
            t0 = time.time()
            _ = net(x)
            xm.mark_step()
            try:
                xm.wait_device_ops()
            except Exception:
                pass
            t1 = time.time()
            times_ms.append((t1 - t0) * 1e3)

    arr = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr))
    median_ms = float(np.median(arr))
    p10_ms = float(np.percentile(arr, 10))
    p90_ms = float(np.percentile(arr, 90))
    imgs_per_s = float(bs / max(1e-12, mean_ms / 1e3))

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "median_ms": median_ms,
        "p10_ms": p10_ms,
        "p90_ms": p90_ms,
        "imgs_per_s": imgs_per_s,
        "iters": int(iters),
        "warmup": int(warmup),
    }


def _export_latency_all_realizations_xla(
    logdir: str,
    base_state_dict_cpu: Dict[str, torch.Tensor],
    model_cfg_no_realization: Dict[str, object],
    device,
    xm,
    batch_sizes: List[int],
    iters: int,
    warmup: int,
) -> str:
    """
    Master-only latency sweep across realizations on TPU.
    Writes: latency_sweep_all_realizations_xla.csv
    """
    path = os.path.join(logdir, "latency_sweep_all_realizations_xla.csv")
    header = [
        "realization",
        "batch_size",
        "mean_ms",
        "std_ms",
        "median_ms",
        "p10_ms",
        "p90_ms",
        "imgs_per_s",
        "iters",
        "warmup",
        "exec_mode",
        "status",
        "error",
        "weights_sha256",
        "weights_numel",
    ]

    base_state_dict_cpu = strip_orig_mod_prefix(base_state_dict_cpu)
    wmeta = state_dict_sha256(base_state_dict_cpu)

    for r in _realization_list():
        for bs in batch_sizes:
            row = {
                "realization": r,
                "batch_size": int(bs),
                "mean_ms": None,
                "std_ms": None,
                "median_ms": None,
                "p10_ms": None,
                "p90_ms": None,
                "imgs_per_s": None,
                "iters": int(iters),
                "warmup": int(warmup),
                "exec_mode": "xla",
                "status": "ok",
                "error": "",
                "weights_sha256": wmeta["sha256"],
                "weights_numel": wmeta["numel"],
            }

            try:
                net_r = ChebResNet(
                    **model_cfg_no_realization,
                    realization=r,
                ).to(device)

                # Load the exact same weights
                load_state_dict_portable(net_r, base_state_dict_cpu, strict=True)

                # Time steady-state inference (warmup excludes compile)
                stats = _xla_latency_ms_samples(
                    net_r,
                    device=device,
                    xm=xm,
                    shape=(int(bs), 3, 64, 64),
                    iters=int(iters),
                    warmup=int(warmup),
                )
                row.update(
                    {
                        "mean_ms": stats["mean_ms"],
                        "std_ms": stats["std_ms"],
                        "median_ms": stats["median_ms"],
                        "p10_ms": stats["p10_ms"],
                        "p90_ms": stats["p90_ms"],
                        "imgs_per_s": stats["imgs_per_s"],
                    }
                )
            except Exception as e:
                row["status"] = "fail"
                row["error"] = str(e).replace("\n", " ")[:500]

            append_csv_row(path, header, row)

    return path


def _mp_fn(index: int, cfg):
    # Lazy import so script can be imported on non-TPU machines.
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.runtime as xr
    from torch_xla import compile as xla_compile

    rank, world = xr.global_ordinal(), xr.world_size()
    device = xm.xla_device()
    device_str = str(device)

    # Rank-specific seeding
    set_seed(int(cfg.seed) + int(rank))
    random.seed(int(cfg.seed) + int(rank))
    np.random.seed(int(cfg.seed) + int(rank))

    # Master sets up logdir
    if xm.is_master_ordinal():
        ensure_logdir(cfg.logdir)
        write_json(vars(cfg), os.path.join(cfg.logdir, "config_args.json"))

    # Ensure TinyImageNet is present and val is repacked
    ensure_tinyimagenet_ready(cfg.data)

    # Split indices
    resume_path = cfg.resume
    train_idx = None
    val_idx = None
    start_ep = 0
    best_val = 0.0

    if os.path.isfile(resume_path):
        raw = torch.load(resume_path, map_location="cpu")
        tr, va = unpack_split_indices(raw.get("split", {}) or {})
        if tr is None or va is None:
            raise RuntimeError(
                "Checkpoint exists but does not contain split indices."
                "Delete checkpoint or fix it."
            )
        train_idx, val_idx = tr, va
        start_ep = int(raw.get("epoch", -1)) + 1
        best_val = float(raw.get("best_val", 0.0))
    else:
        labels = get_tinyimagenet_train_labels(cfg.data)
        tr, va = make_stratified_split_indices(labels, seed=int(cfg.seed), val_frac=float(cfg.val_frac))
        train_idx, val_idx = tr, va

    # Transforms (from your TPU reference)
    mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
    tf_train = T.Compose(
        [
            T.RandAugment(int(cfg.randaugment_n), int(cfg.randaugment_m)),
            T.RandomCrop(64, int(cfg.crop_padding), padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=float(cfg.random_erasing_p), scale=(float(cfg.re_scale_min), float(cfg.re_scale_max))),
        ]
    )
    tf_eval = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    # Build datasets (val-as-test)
    train_ds, val_ds, test_ds = build_tinyimagenet_datasets(
        cfg.data,
        train_idx=train_idx,
        val_idx=val_idx,
        tf_train=tf_train,
        tf_eval=tf_eval,
        use_val_as_test=True,
    )

    # Samplers + loaders
    sam_tr = DistributedSampler(
        train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=bool(cfg.drop_last)
    )
    sam_vl = DistributedSampler(
        val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False
    )
    sam_te = DistributedSampler(
        test_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False
    )

    dl_args = dict(
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        persistent_workers=bool(cfg.persistent_workers) if int(cfg.num_workers) > 0 else False,
        prefetch_factor=int(cfg.prefetch_factor) if int(cfg.num_workers) > 0 else 2,
    )

    tr_cpu = DataLoader(train_ds, batch_size=int(cfg.bs), sampler=sam_tr, drop_last=bool(cfg.drop_last), **dl_args)
    vl_cpu = DataLoader(val_ds, batch_size=int(cfg.bs), sampler=sam_vl, drop_last=False, **dl_args)
    te_cpu = DataLoader(test_ds, batch_size=int(cfg.bs), sampler=sam_te, drop_last=False, **dl_args)

    tr_loader = pl.MpDeviceLoader(
        tr_cpu,
        device,
        loader_prefetch_size=int(cfg.loader_prefetch_size),
        device_prefetch_size=int(cfg.device_prefetch_size),
        host_to_device_transfer_threads=int(cfg.host_to_device_transfer_threads),
    )
    vl_loader = pl.MpDeviceLoader(
        vl_cpu,
        device,
        loader_prefetch_size=int(cfg.loader_prefetch_size),
        device_prefetch_size=int(cfg.device_prefetch_size),
        host_to_device_transfer_threads=int(cfg.host_to_device_transfer_threads),
    )
    te_loader = pl.MpDeviceLoader(
        te_cpu,
        device,
        loader_prefetch_size=int(cfg.loader_prefetch_size),
        device_prefetch_size=int(cfg.device_prefetch_size),
        host_to_device_transfer_threads=int(cfg.host_to_device_transfer_threads),
    )

    # Model cfg
    widths = parse_tuple_ints(cfg.widths)
    Ktup = parse_tuple_ints(cfg.K)
    depth = parse_tuple_ints(cfg.depth)

    model_cfg_no_realization: Dict[str, object] = dict(
        classes=200,
        K=Ktup,
        depth=depth,
        widths=widths,
        drop_rate=float(cfg.drop_rate),
        lap=float(cfg.lambda_lap),
        gate_mode=str(cfg.gate_mode),
        stabilize_cheb=bool(cfg.stabilize_cheb),
    )
    model_cfg_full = dict(model_cfg_no_realization, realization=str(cfg.realization))

    net = ChebResNet(**model_cfg_full).to(device)

    # Optimizer / sched / loss
    opt = torch.optim.SGD(
        net.parameters(),
        lr=float(cfg.lr),
        momentum=0.9,
        weight_decay=float(cfg.wd),
        nesterov=True,
    )

    steps_per_epoch = len(tr_cpu)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=float(cfg.lr),
        epochs=int(cfg.epochs),
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    crit = nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))

    # Resume (do not restore a single RNG state across ranks)
    if os.path.isfile(resume_path):
        _ = load_checkpoint(
            resume_path,
            model=net,
            optimizer=opt,
            scheduler=sched,
            map_location="cpu",
            strict_model=True,
            restore_rng=False,
        )
        xm.broadcast_master_param(net)

    # Master-only snapshots
    if xm.is_master_ordinal():
        _snapshot_env_xla(cfg.logdir, cfg, device_str=device_str, world=world, rank=rank)
        _try_params_macs_snapshot(cfg.logdir, model_cfg_full)

    # Training step fn
    def step_fn(xmix, y1, y2, l1: float, l2: float):
        opt.zero_grad(set_to_none=True)
        out = net(xmix)
        loss = (l1 * crit(out, y1)) + (l2 * crit(out, y2))
        loss.backward()
        xm.optimizer_step(opt, barrier=True)
        sched.step()

        pred = out.argmax(dim=1)
        correct = (l1 * (pred == y1).float().sum()) + (l2 * (pred == y2).float().sum())
        bs_t = xmix.new_tensor(float(xmix.size(0)))
        return loss.detach(), correct.detach(), bs_t

    compiled_step = xla_compile(step_fn)  # keep your reference behavior

    # Artifacts
    eff_path = os.path.join(cfg.logdir, "epoch_efficiency.csv")
    eff_header = [
        "epoch",
        "epoch_sec",
        "train_images",
        "imgs_per_s_epoch",
        "data_time_s",
        "compute_time_s",
        "data_frac",
        "compute_frac",
        "power_method",
        "power_samples",
        "power_missing",
        "mean_watts",
        "min_watts",
        "max_watts",
        "mean_util_gpu",
        "mean_util_mem",
        "mean_mem_mb",
        "energy_joules",
        "images_for_energy",
        "energy_per_img_j",
        "amp_dtype",
        "compiled",
        "realization",
        "dataset",
        "bs",
        "exec_mode",
        "world_size",
    ]

    best_path = cfg.best_model
    best_meta = os.path.join(cfg.logdir, "best_model_meta.json")

    best_val_acc = float(best_val)
    best_epoch = int(start_ep - 1)

    split_dict = pack_split_indices(train_idx, val_idx)

    # Train loop
    for ep in range(int(start_ep), int(cfg.epochs)):
        sam_tr.set_epoch(ep)
        if xm.is_master_ordinal():
            xm.master_print(f"=== Epoch {ep+1}/{cfg.epochs} begin ===")

        t0 = time.time()

        net.train()
        loss_sum = net.fc.weight.new_tensor(0.0)
        correct_sum = net.fc.weight.new_tensor(0.0)
        sample_sum = net.fc.weight.new_tensor(0.0)

        for step, (xb, yb) in enumerate(tr_loader):
            if random.random() < float(cfg.cut_prob):
                xmix, y1, y2, l1, l2 = cutmix_cpu(xb, yb, float(cfg.cut_alpha))
            else:
                xmix, y1, y2, l1, l2 = mixup_cpu(xb, yb, float(cfg.mix_alpha))

            loss_t, corr_t, bs_t = compiled_step(xmix, y1, y2, float(l1), float(l2))
            xm.mark_step()

            loss_sum += loss_t * bs_t
            correct_sum += corr_t
            sample_sum += bs_t

            if (step % int(cfg.log_steps) == 0) and xm.is_master_ordinal():
                xm.master_print(
                    f"  step {step:04d}/{steps_per_epoch} "
                    f"loss {float(loss_t.item()):.4f} "
                    f"lr {float(sched.get_last_lr()[0]):.6f}"
                )

        # Reduce across ranks
        loss_sum = xm.all_reduce(xm.REDUCE_SUM, loss_sum)
        correct_sum = xm.all_reduce(xm.REDUCE_SUM, correct_sum)
        sample_sum = xm.all_reduce(xm.REDUCE_SUM, sample_sum)

        tr_loss = float((loss_sum / sample_sum).item())
        tr_acc = float((100.0 * correct_sum / sample_sum).item())

        # Validation
        net.eval()
        v_loss_sum = net.fc.weight.new_tensor(0.0)
        v_corr_sum = net.fc.weight.new_tensor(0.0)
        v_samp_sum = net.fc.weight.new_tensor(0.0)

        with torch.no_grad():
            for xb, yb in vl_loader:
                out = net(xb)
                loss = crit(out, yb)
                pred = out.argmax(dim=1)
                bs_t = xb.new_tensor(float(xb.size(0)))

                v_loss_sum += loss.detach() * bs_t
                v_corr_sum += (pred == yb).float().sum()
                v_samp_sum += bs_t
                xm.mark_step()

        v_loss_sum = xm.all_reduce(xm.REDUCE_SUM, v_loss_sum)
        v_corr_sum = xm.all_reduce(xm.REDUCE_SUM, v_corr_sum)
        v_samp_sum = xm.all_reduce(xm.REDUCE_SUM, v_samp_sum)

        va_loss = float((v_loss_sum / v_samp_sum).item())
        va_acc = float((100.0 * v_corr_sum / v_samp_sum).item())

        epoch_sec = time.time() - t0
        train_images = len(tr_cpu.dataset)
        imgs_per_s_epoch = float(train_images / max(1e-9, epoch_sec))

        if xm.is_master_ordinal():
            xm.master_print(
                f"[Ep {ep:03d}] train L {tr_loss:.3f} A {tr_acc:.2f}% | "
                f"val L {va_loss:.3f} A {va_acc:.2f}% | "
                f"epoch_sec {epoch_sec:.2f}s | {imgs_per_s_epoch:.1f} img/s | exec_mode=xla"
            )

            save_learning_curve_csv(
                cfg.logdir,
                {
                    "epoch": ep,
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "true_train_loss": float("nan"),
                    "true_train_acc": float("nan"),
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "lr": float(sched.get_last_lr()[0]),
                    "epoch_seconds": float(epoch_sec),
                    "wall_seconds": float(time.time()),
                },
            )

            append_csv_row(
                eff_path,
                eff_header,
                {
                    "epoch": ep,
                    "epoch_sec": float(epoch_sec),
                    "train_images": int(train_images),
                    "imgs_per_s_epoch": float(imgs_per_s_epoch),
                    "data_time_s": float("nan"),
                    "compute_time_s": float("nan"),
                    "data_frac": float("nan"),
                    "compute_frac": float("nan"),
                    "power_method": "none",
                    "power_samples": 0,
                    "power_missing": 0,
                    "mean_watts": None,
                    "min_watts": None,
                    "max_watts": None,
                    "mean_util_gpu": None,
                    "mean_util_mem": None,
                    "mean_mem_mb": None,
                    "energy_joules": 0.0,
                    "images_for_energy": int(train_images + len(vl_cpu.dataset)),
                    "energy_per_img_j": None,
                    "amp_dtype": amp_dtype_name(None),
                    "compiled": 1,  # training step uses xla_compile
                    "realization": cfg.realization,
                    "dataset": "tinyimagenet",
                    "bs": int(cfg.bs),
                    "exec_mode": "xla",
                    "world_size": int(world),
                },
            )

            save_checkpoint(
                cfg.resume,
                epoch=ep,
                best_val=best_val_acc,
                model=net,
                optimizer=opt,
                scheduler=sched,
                split_dict=split_dict,
                rng_state=None,
                extra={"exec_mode": "xla", "world_size": int(world)},
            )

            if va_acc > best_val_acc:
                best_val_acc = float(va_acc)
                best_epoch = int(ep)
                try:
                    # Save best weights (master only)
                    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, best_path)
                    write_json(
                        {
                            "best_epoch": int(best_epoch),
                            "best_val_acc": float(best_val_acc),
                            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "exec_mode": "xla",
                        },
                        best_meta,
                    )
                    xm.master_print(
                        f"[best] New best val {best_val_acc:.2f}% at epoch {best_epoch} → saved {best_path}"
                    )
                except Exception as e:
                    xm.master_print(f"[best] Save failed: {e}")

    # Final test + optional exports
    if xm.is_master_ordinal():
        # Load best weights if present
        if os.path.isfile(best_path):
            sd = torch.load(best_path, map_location="cpu")
            load_state_dict_portable(net, sd, strict=True)
            xm.master_print(f"[best] Loaded best_model (epoch {best_epoch}, val {best_val_acc:.2f}%)")

        net.eval()
        t_loss_sum = net.fc.weight.new_tensor(0.0)
        t_corr_sum = net.fc.weight.new_tensor(0.0)
        t_samp_sum = net.fc.weight.new_tensor(0.0)

        with torch.no_grad():
            for xb, yb in te_loader:
                out = net(xb)
                loss = crit(out, yb)
                pred = out.argmax(dim=1)
                bs_t = xb.new_tensor(float(xb.size(0)))

                t_loss_sum += loss.detach() * bs_t
                t_corr_sum += (pred == yb).float().sum()
                t_samp_sum += bs_t
                xm.mark_step()

        t_loss = float((t_loss_sum / t_samp_sum).item())
        t_acc = float((100.0 * t_corr_sum / t_samp_sum).item())

        xm.master_print(f"\nFinal TinyImageNet (val-as-test) → loss={t_loss:.3f} acc={t_acc:.2f}%")
        write_json(
            {"test_loss": float(t_loss), "test_acc": float(t_acc), "best_val_acc": float(best_val_acc), "best_epoch": int(best_epoch)},
            os.path.join(cfg.logdir, "tinyimagenet_test_metrics.json"),
        )

        if int(cfg.dump_order_scales) == 1:
            try:
                dump_order_scales(net, cfg.logdir)
            except Exception as e:
                xm.master_print(f"[order_scales] skipped: {e}")

        # Optional: single-realization latency (current realization)
        if int(cfg.measure_latency) == 1:
            try:
                bs_list = [1, int(cfg.bs)] if int(cfg.bs) != 1 else [1]
                out = {"exec_mode": "xla", "realization": cfg.realization}
                for bs in bs_list:
                    out[f"bs{bs}"] = _xla_latency_ms_samples(
                        net, device=device, xm=xm, shape=(int(bs), 3, 64, 64),
                        iters=int(cfg.lat_iters), warmup=int(cfg.lat_warmup)
                    )
                write_json(out, os.path.join(cfg.logdir, "latency_stats_xla.json"))
                xm.master_print("[Latency/XLA] saved latency_stats_xla.json")
            except Exception as e:
                xm.master_print(f"[Latency/XLA] skipped: {e}")

        # Optional: sweep latency across realizations
        if int(cfg.latency_all_realizations) == 1:
            try:
                # Use best weights if available, else use current net weights
                if os.path.isfile(best_path):
                    base_sd_cpu = torch.load(best_path, map_location="cpu")
                    wsrc = "best_model"
                else:
                    try:
                        xm.wait_device_ops()
                    except Exception:
                        pass
                    base_sd_cpu = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                    wsrc = "current_weights"

                bss = _build_bs_list(int(cfg.latency_all_max_bs))
                path = _export_latency_all_realizations_xla(
                    logdir=cfg.logdir,
                    base_state_dict_cpu=base_sd_cpu,
                    model_cfg_no_realization=model_cfg_no_realization,
                    device=device,
                    xm=xm,
                    batch_sizes=bss,
                    iters=int(cfg.lat_iters),
                    warmup=int(cfg.lat_warmup),
                )
                xm.master_print(f"[Latency/XLA] all-realizations sweep saved: {path} (weights_source={wsrc})")
            except Exception as e:
                xm.master_print(f"[Latency/XLA] all-realizations sweep skipped: {e}")


def main():
    p = argparse.ArgumentParser()

    # Data / logs
    p.add_argument("--data", default="./data", help="Data root (TinyImageNet will be downloaded/extracted here).")
    p.add_argument("--logdir", default="./chebgate_logs_tiny", help="Artifact output directory.")
    p.add_argument("--resume", default="", help="Checkpoint path. Default: <logdir>/checkpoint.pth")
    p.add_argument("--best_model", default="", help="Best weights path. Default: <logdir>/best_model.pth")

    # Split
    p.add_argument("--val_frac", type=float, default=0.1, help="Train split fraction used for validation (stratified).")

    # Hyperparameters (from your TPU reference)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=32, help="Per-core batch size.")
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mix_alpha", type=float, default=0.2)
    p.add_argument("--cut_alpha", type=float, default=1.0)
    p.add_argument("--cut_prob", type=float, default=0.5)
    p.add_argument("--label_smoothing", type=float, default=0.1)

    # Transforms
    p.add_argument("--randaugment_n", type=int, default=3)
    p.add_argument("--randaugment_m", type=int, default=9)
    p.add_argument("--crop_padding", type=int, default=8)
    p.add_argument("--random_erasing_p", type=float, default=0.25)
    p.add_argument("--re_scale_min", type=float, default=0.02)
    p.add_argument("--re_scale_max", type=float, default=0.2)

    # Loader / prefetch knobs
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--drop_last", action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=16)
    p.add_argument("--loader_prefetch_size", type=int, default=16)
    p.add_argument("--device_prefetch_size", type=int, default=4)
    p.add_argument("--host_to_device_transfer_threads", type=int, default=1)
    p.add_argument("--log_steps", type=int, default=100)

    # Model knobs
    p.add_argument("--widths", type=str, default="192,384,768")
    p.add_argument("--K", type=str, default="3,5,7")
    p.add_argument("--depth", type=str, default="7,7,7")
    p.add_argument("--drop_rate", type=float, default=0.2)
    p.add_argument("--lambda_lap", type=float, default=0.25)
    p.add_argument(
        "--realization",
        type=str,
        default="mstream",
        choices=["streamed", "concat", "gemm", "mstream"],
    )
    p.add_argument("--gate_mode", type=str, default="on", choices=["on", "off"])
    p.add_argument("--stabilize_cheb", type=int, default=0)

    # Optional exports
    p.add_argument("--dump_order_scales", type=int, default=1)

    # Latency: single model + all-realizations sweep
    p.add_argument("--measure_latency", type=int, default=0, help="Measure latency for the chosen realization only.")
    p.add_argument("--lat_warmup", type=int, default=50)
    p.add_argument("--lat_iters", type=int, default=200)

    p.add_argument(
        "--latency_all_realizations",
        type=int,
        default=0,
        help="If 1, run latency sweep across concat/streamed/mstream/gemm and write CSV.",
    )
    p.add_argument("--latency_all_max_bs", type=int, default=128)

    cfg, _ = p.parse_known_args()

    # Default resume/best paths under logdir
    if not cfg.resume:
        cfg.resume = os.path.join(cfg.logdir, "checkpoint.pth")
    if not cfg.best_model:
        cfg.best_model = os.path.join(cfg.logdir, "best_model.pth")

    # Clear stray TPU env var (matches your reference behavior)
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)

    # Spawn TPU workers
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(_mp_fn, args=(cfg,), start_method="fork")


if __name__ == "__main__":
    main()
