"""
ChebGate / ChebResNet training + evaluation + profiling entrypoint.
"""

import argparse
import inspect
import math
import os
import platform
import time
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from chebgate.core import (
    set_seed,
    ensure_logdir,
    write_json,
    parse_tuple_ints,
    amp_dtype_name,
    state_dict_uncompiled,
    _unwrap_compiled,
    _sync_if_cuda,
    GPUPowerSampler,
)
from chebgate.model import ChebResNet, ChebConv2d
from chebgate.training import train_epoch, evaluate
from chebgate.metrics import (
    count_parameters,
    profile_macs,
    profile_macs_breakdown,
    latency_ms_samples,
    latency_sweep_csv,
    peak_mem_mb,
    save_learning_curve_csv,
    collect_gate_stats,
    dump_order_scales,
    layer_exactness_check,
    network_exactness_check,
    can_run_compiled,
    fairness_latency_all,
)


def snapshot_hardware(logdir: str, cfg, device, epoch_seconds=None, peak_mem_eval_mb=None):
    try:
        cudnn_ver = cudnn.version()
    except Exception:
        cudnn_ver = None

    info = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda if torch.version.cuda else None,
        "cudnn": cudnn_ver,
        "deterministic": bool(getattr(cfg, "deterministic", 0)),
        "cudnn_benchmark": bool(getattr(cfg, "cudnn_benchmark", 1)),
        "amp": bool(getattr(cfg, "amp", 0)),
        "amp_dtype_train": amp_dtype_name(getattr(cfg, "amp_dtype_train", None)),
        "eval_amp": bool(getattr(cfg, "eval_amp", 0)),
        "amp_dtype_eval": amp_dtype_name(getattr(cfg, "amp_dtype_eval", None)),
        "tf32": int(getattr(cfg, "tf32", 0)) if torch.cuda.is_available() else 0,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "total_vram_mb": (
            torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            if torch.cuda.is_available()
            else None
        ),
        "batch_size": cfg.bs,
        "epochs": cfg.epochs,
        "dataset": cfg.dataset,
        "realization": cfg.realization,
        "gate_mode": cfg.gate_mode,
        "lambda_lap": cfg.lambda_lap,
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
    if peak_mem_eval_mb is not None:
        info["peak_memory_eval_mb"] = float(peak_mem_eval_mb)

    write_json(info, os.path.join(logdir, "hardware_env.json"))
    return info


def main():
    p = argparse.ArgumentParser()

    # Data / training
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data", default="./data")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--auto_lr", type=int, default=1)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cut_alpha", type=float, default=1.0)
    p.add_argument("--drop_rate", type=float, default=0.1)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument(
        "--amp_dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16"],
        help="AMP dtype for training when --amp=1; auto=bf16 if supported else fp16.",
    )
    p.add_argument(
        "--eval_amp",
        type=int,
        default=1,
        help="If 1, use autocast for eval/test; if 0, eval/test in fp32 even if training uses AMP.",
    )
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--clip_every", type=int, default=1)
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--compile_mode", type=str, default="reduce-overhead")
    p.add_argument("--compile_fullgraph", type=int, default=0)

    # Model knobs
    p.add_argument("--widths", type=str, default="128,256,512")
    p.add_argument("--K", type=str, default="3,5,5")
    p.add_argument("--depth", type=str, default="7,7,7")
    p.add_argument("--lambda_lap", type=float, default=0.25)
    p.add_argument(
        "--realization",
        type=str,
        default="streamed",
        choices=["streamed", "concat", "gemm", "mstream"],
    )
    p.add_argument("--gate_mode", type=str, default="on", choices=["on", "off"])
    p.add_argument(
        "--stabilize_cheb",
        type=int,
        default=0,
        help="Scale Laplacian so ||A||<=1 in Chebyshev recurrence (0/1).",
    )

    # Loader
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--persistent_workers", type=int, default=1)

    # Eval cadence
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--true_train_every", type=int, default=0)

    # Determinism / cuDNN
    p.add_argument("--deterministic", type=int, default=0)
    p.add_argument("--cudnn_benchmark", type=int, default=1)
    p.add_argument("--tf32", type=int, default=1, help="If 1, enable TF32 matmul/cudnn; if 0, disable TF32.")

    # Logs
    p.add_argument("--logdir", default="./chebgate_logs")

    # Latency sweep options (for the currently chosen realization)
    p.add_argument("--latency_sweep", type=int, default=0)
    p.add_argument("--latency_max_bs", type=int, default=128)
    p.add_argument("--latency_samples_dir", type=str, default="")

    # Fairness harness (unified only; eager + compiled-if-available)
    p.add_argument("--fair_all", type=int, default=0)

    # Power/Joules
    p.add_argument("--power_sample", type=int, default=1)
    p.add_argument("--power_interval", type=float, default=0.2)
    p.add_argument("--power_device", type=int, default=0)

    cfg, _ = p.parse_known_args()

    set_seed(cfg.seed)
    ensure_logdir(cfg.logdir)
    write_json(vars(cfg), os.path.join(cfg.logdir, "config_args.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TF32 & cuDNN
    tf32_flag = int(cfg.tf32) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32_flag)
        torch.backends.cudnn.allow_tf32 = bool(tf32_flag)
        try:
            torch.set_float32_matmul_precision("high" if tf32_flag else "highest")
        except Exception:
            pass

    if cfg.deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = bool(cfg.cudnn_benchmark)

    # AMP dtype selection (training) + GradScaler
    amp_dtype_train = None
    if bool(cfg.amp) and device.type == "cuda":
        if cfg.amp_dtype == "fp16":
            amp_dtype_train = torch.float16
        elif cfg.amp_dtype == "bf16":
            if not torch.cuda.is_bf16_supported():
                print("[amp] Warning: bf16 not supported on this GPU; falling back to fp16.")
                amp_dtype_train = torch.float16
            else:
                amp_dtype_train = torch.bfloat16
        else:  # auto
            amp_dtype_train = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    scaler = None
    if amp_dtype_train is not None and amp_dtype_train == torch.float16 and device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Eval AMP dtype (can be fp32 even if training uses AMP)
    amp_dtype_eval = (
        amp_dtype_train if (bool(cfg.eval_amp) and amp_dtype_train is not None and device.type == "cuda") else None
    )

    # Attach for snapshots
    cfg.amp_dtype_train = amp_dtype_train
    cfg.amp_dtype_eval = amp_dtype_eval

    print(
        f"[env] device={device} | cudnn.benchmark={cudnn.benchmark} | deterministic={cudnn.deterministic} | "
        f"tf32={tf32_flag} | amp_train={bool(cfg.amp)}({amp_dtype_name(amp_dtype_train)}) | "
        f"eval_amp={bool(cfg.eval_amp)}({amp_dtype_name(amp_dtype_eval)})"
    )

    # Dataset-specific normalization
    if cfg.dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    else:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    # Robust aug fallback for old torchvision
    class _Identity:
        def __call__(self, x):
            return x

    try:
        aug = T.RandAugment(2, 9)
    except Exception:
        try:
            aug = T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)
        except Exception:
            aug = _Identity()

    tf_train = T.Compose(
        [
            aug,
            T.RandomCrop(32, 4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    tf_eval = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    # Datasets / splits
    if cfg.dataset == "cifar10":
        ds_all = torchvision.datasets.CIFAR10(cfg.data, True, download=True)
        classes = 10
        targets_all = ds_all.targets
    else:
        ds_all = torchvision.datasets.CIFAR100(cfg.data, True, download=True)
        classes = 100
        targets_all = ds_all.targets

    idx = list(range(len(ds_all)))
    lbl = targets_all
    sss = StratifiedShuffleSplit(1, test_size=0.1, random_state=42)
    tr_idx, val_idx = next(sss.split(idx, lbl))

    if cfg.dataset == "cifar10":
        train_ds = Subset(torchvision.datasets.CIFAR10(cfg.data, True, transform=tf_train), tr_idx)
        val_ds = Subset(torchvision.datasets.CIFAR10(cfg.data, True, transform=tf_eval), val_idx)
        test_ds = torchvision.datasets.CIFAR10(cfg.data, False, transform=tf_eval, download=True)
        true_tr_ds = Subset(torchvision.datasets.CIFAR10(cfg.data, True, transform=tf_eval), tr_idx)
    else:
        train_ds = Subset(torchvision.datasets.CIFAR100(cfg.data, True, transform=tf_train), tr_idx)
        val_ds = Subset(torchvision.datasets.CIFAR100(cfg.data, True, transform=tf_eval), val_idx)
        test_ds = torchvision.datasets.CIFAR100(cfg.data, False, transform=tf_eval, download=True)
        true_tr_ds = Subset(torchvision.datasets.CIFAR100(cfg.data, True, transform=tf_eval), tr_idx)

    # DataLoader kwargs guarded for older torch
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

    tr_loader = DataLoader(train_ds, cfg.bs, True, **loader_kw)
    val_loader = DataLoader(val_ds, cfg.bs, False, **loader_kw)
    te_loader = DataLoader(test_ds, cfg.bs, False, **loader_kw)
    true_tr_loader = DataLoader(true_tr_ds, cfg.bs, False, **loader_kw)

    # Model / opt / sched
    widths = parse_tuple_ints(cfg.widths)
    Ktup = parse_tuple_ints(cfg.K)
    depth = parse_tuple_ints(cfg.depth)

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

    order_p, other_p = [], []
    for n, par in net.named_parameters():
        (order_p if "order_scales" in n else other_p).append(par)

    used_lr = cfg.lr * (cfg.bs / 128.0) if cfg.auto_lr else cfg.lr
    print(f"[opt] bs={cfg.bs} | base_lr={cfg.lr:.4f} | auto_lr={cfg.auto_lr} → used_lr={used_lr:.4f}")

    opt = torch.optim.SGD(
        [{"params": other_p}, {"params": order_p, "lr": used_lr * 0.1}],
        lr=used_lr,
        momentum=0.9,
        weight_decay=cfg.wd,
        nesterov=True,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Params & MACs snapshot (before compile for hooks)
    params = count_parameters(net, trainable_only=True)
    macs = profile_macs(net, input_size=(1, 3, 32, 32), device=device)
    flops = 2 * macs
    print(f"[S0] Params: {params/1e6:.3f}M | MACs@32x32: {macs/1e9:.3f}G | FLOPs≈{flops/1e9:.3f}G")
    write_json(
        {
            "params_trainable": int(params),
            "macs_32x32": int(macs),
            "flops_32x32": int(flops),
            "notes": "FLOPs ≈ 2×MACs; counts big 1×1 only for streamed/gemm/mstream (concat via Conv2d).",
        },
        os.path.join(cfg.logdir, "params_flops.json"),
    )

    # Optional torch.compile for the main training model
    compiled_flag_main = False
    if hasattr(torch, "compile") and bool(cfg.compile):
        try:
            net = torch.compile(net, mode=cfg.compile_mode, fullgraph=bool(cfg.compile_fullgraph))
            compiled_flag_main = True
            print(f"[compile] torch.compile enabled (mode={cfg.compile_mode}, fullgraph={bool(cfg.compile_fullgraph)})")
        except Exception as e:
            print(f"[compile] skipped: {e}")

    # Hardware snapshot baseline
    snapshot_hardware(cfg.logdir, cfg, device)

    wall0 = time.time()
    epoch_seconds = []

    best_val_acc = -float("inf")
    best_epoch = -1
    best_path = os.path.join(cfg.logdir, "best_model.pth")
    best_meta = os.path.join(cfg.logdir, "best_model_meta.json")

    # CSV for epoch efficiency
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
    ]
    eff_path = os.path.join(cfg.logdir, "epoch_efficiency.csv")

    # Run-level aggregation for energy/util
    energy_total_j = 0.0
    total_images_for_energy = 0
    last_power_method = "none"
    util_gpu_sum = 0.0
    util_mem_sum = 0.0
    mem_mb_sum = 0.0
    util_epochs = 0

    # Training loop
    for ep in range(cfg.epochs):
        sampler = None
        if int(cfg.power_sample) == 1 and device.type == "cuda":
            sampler = GPUPowerSampler(interval=cfg.power_interval, device_index=cfg.power_device, method="auto")
            sampler.start()

        t0 = time.time()
        tr_l, tr_a, tr_data_s, tr_comp_s = train_epoch(
            net, tr_loader, crit, opt, scaler, cfg, device, amp_dtype_train
        )
        sched.step()

        if cfg.true_train_every and (ep % cfg.true_train_every == 0):
            tc_l, tc_a, _, _ = evaluate(net, true_tr_loader, crit, device, amp_dtype_eval)
        else:
            tc_l = float("nan")
            tc_a = float("nan")

        run_val = (ep % max(1, cfg.val_every) == 0)
        if run_val:
            va_l, va_a, _, _ = evaluate(net, val_loader, crit, device, amp_dtype_eval)
        else:
            va_l = float("nan")
            va_a = float("nan")

        if sampler is not None:
            sampler.stop()
            pstats = sampler.stats()
        else:
            pstats = {
                "method": "none",
                "samples": 0,
                "missing": 0,
                "mean_watts": None,
                "min_watts": None,
                "max_watts": None,
                "mean_util_gpu": None,
                "mean_util_mem": None,
                "mean_mem_used_mb": None,
                "energy_joules": 0.0,
            }

        epoch_sec = time.time() - t0
        epoch_seconds.append(epoch_sec)

        train_images = len(tr_loader.dataset)
        val_images = len(val_loader.dataset)
        true_tr_images = len(true_tr_loader.dataset)

        imgs_per_s_epoch = (train_images / epoch_sec) if epoch_sec > 0 else float("inf")

        energy_j = float(pstats.get("energy_joules", 0.0))
        energy_total_j += energy_j
        last_power_method = pstats.get("method", last_power_method)

        epoch_images_energy = train_images
        if cfg.true_train_every and (ep % cfg.true_train_every == 0):
            epoch_images_energy += true_tr_images
        if run_val:
            epoch_images_energy += val_images
        total_images_for_energy += epoch_images_energy

        energy_per_img = (energy_j / epoch_images_energy) if epoch_images_energy > 0 else None

        if pstats.get("mean_util_gpu") is not None:
            util_gpu_sum += pstats["mean_util_gpu"]
            util_mem_sum += (pstats.get("mean_util_mem") or 0.0)
            mem_mb_sum += (pstats.get("mean_mem_used_mb") or 0.0)
            util_epochs += 1

        lr_show = opt.param_groups[0]["lr"]
        print(
            f"Epoch {ep:03d} | Train L {tr_l:.3f} A {tr_a:.2f}% | "
            f"TrueTrain L {tc_l:.3f} A {tc_a:.2f}% | "
            f"Val L {va_l:.3f} A {va_a:.2f}% | LR {lr_show:.6f} | "
            f"epoch_sec {epoch_sec:.2f}s | {imgs_per_s_epoch:.1f} img/s | "
            f"data {tr_data_s:.2f}s ({(tr_data_s/epoch_sec)*100:.1f}%) | "
            f"compute {tr_comp_s:.2f}s ({(tr_comp_s/epoch_sec)*100:.1f}%) | "
            f"energy {energy_j:.1f} J (per image {energy_per_img:.4e} J over {epoch_images_energy} imgs)"
        )

        # Log curves
        save_learning_curve_csv(
            cfg.logdir,
            {
                "epoch": ep,
                "train_loss": tr_l,
                "train_acc": tr_a,
                "true_train_loss": tc_l,
                "true_train_acc": tc_a,
                "val_loss": va_l,
                "val_acc": va_a,
                "lr": lr_show,
                "epoch_seconds": epoch_sec,
                "wall_seconds": time.time() - wall0,
            },
        )

        # Log efficiency
        data_frac = (tr_data_s / epoch_sec) if epoch_sec > 0 else 0.0
        comp_frac = (tr_comp_s / epoch_sec) if epoch_sec > 0 else 0.0
        from chebgate.core.io import append_csv_row  # keep local to preserve module boundaries
        append_csv_row(
            eff_path,
            eff_header,
            {
                "epoch": ep,
                "epoch_sec": epoch_sec,
                "train_images": train_images,
                "imgs_per_s_epoch": imgs_per_s_epoch,
                "data_time_s": tr_data_s,
                "compute_time_s": tr_comp_s,
                "data_frac": data_frac,
                "compute_frac": comp_frac,
                "power_method": pstats.get("method", "none"),
                "power_samples": pstats.get("samples", 0),
                "power_missing": pstats.get("missing", 0),
                "mean_watts": pstats.get("mean_watts", None),
                "min_watts": pstats.get("min_watts", None),
                "max_watts": pstats.get("max_watts", None),
                "mean_util_gpu": pstats.get("mean_util_gpu", None),
                "mean_util_mem": pstats.get("mean_util_mem", None),
                "mean_mem_mb": pstats.get("mean_mem_used_mb", None),
                "energy_joules": energy_j,
                "images_for_energy": epoch_images_energy,
                "energy_per_img_j": energy_per_img,
                "amp_dtype": amp_dtype_name(amp_dtype_train),
                "compiled": int(compiled_flag_main),
                "realization": cfg.realization,
                "dataset": cfg.dataset,
                "bs": cfg.bs,
            },
        )

        # Best checkpoint on val
        if run_val and math.isfinite(va_a) and (va_a > best_val_acc):
            best_val_acc = va_a
            best_epoch = ep
            try:
                torch.save(state_dict_uncompiled(net), best_path)
                write_json(
                    {
                        "best_epoch": int(best_epoch),
                        "best_val_acc": float(best_val_acc),
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "lr_at_save": float(lr_show),
                    },
                    best_meta,
                )
                print(f"[best] New best Val Acc {best_val_acc:.2f}% at epoch {best_epoch} → saved {best_path}")
            except Exception as e:
                print(f"[best] Save failed: {e}")

    # Final test (load best Val)
    if os.path.isfile(best_path):
        sd = torch.load(best_path, map_location=device)
        # Use portable loader via model core (works for compiled/uncompiled)
        from chebgate.core.state_dict import load_state_dict_portable
        load_state_dict_portable(net, sd, strict=True)
        print(f"[best] Loaded best model from epoch {best_epoch} (Val Acc {best_val_acc:.2f}%)")
    else:
        print("[best] No best_model.pth found; evaluating current weights")

    te_l, te_a, _, _ = evaluate(net, te_loader, crit, device, amp_dtype_eval)
    print(f"\nFinal {cfg.dataset.upper()} Test Accuracy: {te_a:.2f}%")
    write_json({"test_loss": te_l, "test_acc": te_a}, os.path.join(cfg.logdir, f"{cfg.dataset}_test_metrics.json"))

    # Peak memory & latency (autocast-aware, using eval config)
    pm_mb = peak_mem_mb(net, shape=(cfg.bs, 3, 32, 32), device=device, amp_dtype=amp_dtype_eval)
    if pm_mb is not None:
        print(f"[T3-M] Peak memory during eval forward (bs={cfg.bs}): {pm_mb:.1f} MB")

    lat1 = latency_ms_samples(net, shape=(1, 3, 32, 32), iters=200, warmup=50, device=device, amp_dtype=amp_dtype_eval)
    lat128 = latency_ms_samples(
        net, shape=(128, 3, 32, 32), iters=200, warmup=50, device=device, amp_dtype=amp_dtype_eval
    )
    write_json(
        {"bs1": lat1, "bs128": lat128, f"peak_mem_eval_mb_bs{cfg.bs}": pm_mb},
        os.path.join(cfg.logdir, "latency_stats.json"),
    )
    print(
        f"[Latency] bs=1   {lat1['mean_ms']:.3f}±{lat1['std_ms']:.3f} ms "
        f"(median {lat1['median_ms']:.3f}, p10 {lat1['p10_ms']:.3f}, p90 {lat1['p90_ms']:.3f}) "
        f"| {lat1['imgs_per_s']:.1f} img/s"
    )
    print(
        f"[Latency] bs=128 {lat128['mean_ms']:.3f}±{lat128['std_ms']:.3f} ms "
        f"(median {lat128['median_ms']:.3f}, p10 {lat128['p10_ms']:.3f}, p90 {lat128['p90_ms']:.3f}) "
        f"| {lat128['imgs_per_s']:.1f} img/s"
    )

    # Optional latency sweep CSV for the currently chosen realization
    if int(cfg.latency_sweep) == 1:
        max_bs = int(cfg.latency_max_bs)
        bss = [1, 2, 4, 8, 16, 32, 64, 128] + ([256] if max_bs >= 256 else [])
        samples_dir = cfg.latency_samples_dir if cfg.latency_samples_dir else None
        latency_sweep_csv(
            net,
            cfg.logdir,
            device=device,
            amp_dtype=amp_dtype_eval,
            batch_sizes=tuple(bss),
            iters=200,
            warmup=50,
            tag=cfg.realization,
            samples_dir=samples_dir,
        )

    # Finalize env snapshot
    snapshot_hardware(cfg.logdir, cfg, device, epoch_seconds, pm_mb)
    print("[E1] Hardware/env snapshot saved:", os.path.join(cfg.logdir, "hardware_env.json"))

    # Gate stats + alpha dumps (using eval AMP config)
    collect_gate_stats(net, val_loader, device, cfg.logdir, amp_dtype=amp_dtype_eval)
    dump_order_scales(net, cfg.logdir)

    # Exactness checks (layer + network)
    with torch.no_grad():
        ex_layer = layer_exactness_check(net, device=device, batch=8, H=32, W=32)
        if ex_layer is not None:
            write_json(ex_layer, os.path.join(cfg.logdir, "exactness_fp32_layer.json"))
            print(
                "[exactness:layer] "
                f"S–C max {ex_layer['streamed_vs_concat']['max_abs']:.3e} (mean {ex_layer['streamed_vs_concat']['mean_abs']:.3e}) | "
                f"S–M max {ex_layer['streamed_vs_mstream']['max_abs']:.3e} (mean {ex_layer['streamed_vs_mstream']['mean_abs']:.3e}) | "
                f"S–G max {ex_layer['streamed_vs_gemm']['max_abs']:.3e} (mean {ex_layer['streamed_vs_gemm']['mean_abs']:.3e}) | "
                f"C–M max {ex_layer['concat_vs_mstream']['max_abs']:.3e} (mean {ex_layer['concat_vs_mstream']['mean_abs']:.3e}) | "
                f"C–G max {ex_layer['concat_vs_gemm']['max_abs']:.3e} (mean {ex_layer['concat_vs_gemm']['mean_abs']:.3e}) | "
                f"M–G max {ex_layer['mstream_vs_gemm']['max_abs']:.3e} (mean {ex_layer['mstream_vs_gemm']['mean_abs']:.3e})"
            )

        model_cfg: Dict[str, object] = dict(
            classes=classes,
            K=Ktup,
            depth=depth,
            widths=widths,
            drop_rate=cfg.drop_rate,
            lap=cfg.lambda_lap,
            gate_mode=cfg.gate_mode,
            stabilize_cheb=bool(cfg.stabilize_cheb),
        )
        ex_net = network_exactness_check(state_dict_uncompiled(net), model_cfg, device)
        write_json(ex_net, os.path.join(cfg.logdir, "exactness_fp32_network.json"))
        print(
            "[exactness:network] "
            f"C–S max {ex_net['C_vs_S']['max_abs']:.3e} (mean {ex_net['C_vs_S']['mean_abs']:.3e}) | "
            f"C–M max {ex_net['C_vs_M']['max_abs']:.3e} (mean {ex_net['C_vs_M']['mean_abs']:.3e}) | "
            f"C–G max {ex_net['C_vs_G']['max_abs']:.3e} (mean {ex_net['C_vs_G']['mean_abs']:.3e}) | "
            f"S–M max {ex_net['S_vs_M']['max_abs']:.3e} (mean {ex_net['S_vs_M']['mean_abs']:.3e}) | "
            f"S–G max {ex_net['S_vs_G']['max_abs']:.3e} (mean {ex_net['S_vs_G']['mean_abs']:.3e}) | "
            f"M–G max {ex_net['M_vs_G']['max_abs']:.3e} (mean {ex_net['M_vs_G']['mean_abs']:.3e})"
        )

    # Per-module MACs breakdown
    profile_macs_breakdown(
        _unwrap_compiled(net),
        input_size=(1, 3, 32, 32),
        device=device,
        save_csv=os.path.join(cfg.logdir, "F3_macs_breakdown.csv"),
    )

    # Fairness harness (unified; eager + compiled-if-available)
    if int(cfg.fair_all) == 1:
        try:
            realizations = ["concat", "streamed", "mstream", "gemm"]
            bss = [1, 2, 4, 8, 16, 32, 64, 128, 256]

            # Eager
            fairness_latency_all(
                state_dict_uncompiled(net),
                model_cfg,
                realizations,
                cfg.logdir,
                device,
                amp_dtype_eval,
                bss,
                compiled=False,
                compile_mode=cfg.compile_mode,
                fullgraph=bool(cfg.compile_fullgraph),
                tag="all_realizations_eager",
            )

            # Compiled if supported
            if can_run_compiled(device):
                fairness_latency_all(
                    state_dict_uncompiled(net),
                    model_cfg,
                    realizations,
                    cfg.logdir,
                    device,
                    amp_dtype_eval,
                    bss,
                    compiled=True,
                    compile_mode=cfg.compile_mode,
                    fullgraph=bool(cfg.compile_fullgraph),
                    tag="all_realizations_compiled",
                )
            else:
                print("[fair_all] compile not supported on this platform → eager-only unified sweep.")
        except Exception as e:
            print(f"[fair_all] unified sweep skipped: {e}")

    print("\n[DONE] All metrics & artifacts saved in:", cfg.logdir)


if __name__ == "__main__":
    main()
