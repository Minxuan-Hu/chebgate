#!/usr/bin/env python3
# Adds PARTIAL SWAP via --swap_beta in [0,1]:
#   g_use = (1-beta) * g_orig + beta * g_proto
#   - beta=1.0: original hard swap behavior
#   - beta=0.0: no intervention (base gates)
# For amp_preserve, we blend first, then re-normalize amplitude to match base per-sample amplitude.

import os
import re
import json
import argparse
import hashlib
import random
import pickle
import shutil
import zipfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.transforms import Bbox

import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset

import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


# -------------------------
# Determinism / strictness
# -------------------------
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# -------------------------
# Global plot style
# -------------------------
plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "lines.linewidth": 1.0,
})
COL_BASE = "C1"
COL_SWAP = "C2"
COL_AUX  = "C0"


# -------------------------
# Checkpoint helpers
# -------------------------
def safe_torch_load(path: Path, map_location="cpu", trusted: bool = True):
    """
    Fixes PyTorch>=2.6 weights_only behavior for checkpoints that include numpy objects.
    If checkpoint is trusted (your own), fallback to weights_only=False.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        msg = str(e)
        wo_failed = ("Weights only load failed" in msg) or isinstance(e, pickle.UnpicklingError)
        if wo_failed and trusted:
            return torch.load(path, map_location=map_location, weights_only=False)
        raise

def extract_state_dict(obj):
    # best_model may be pure state_dict; checkpoint may be dict with model/state_dict keys
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()) and any(k.endswith(".weight") for k in obj.keys()):
        return obj
    raise ValueError("Could not extract a state_dict from checkpoint.")

def strip_module_prefix(sd: dict):
    if any(k.startswith("module.") for k in sd.keys()):
        return {k[len("module."):]: v for k, v in sd.items()}
    if any(k.startswith("model.") for k in sd.keys()):
        return {k[len("model."):]: v for k, v in sd.items()}
    return sd

def infer_arch_from_sd(sd: dict):
    """
    Infers ChebResNet shape from state_dict keys.
    Assumes naming like:
      stem.0.weight
      l{stage}.{block}.c{1|2}.order_scales
      fc.{weight,bias}
    """
    classes = int(sd["fc.bias"].numel())

    def max_block(stage):
        pat = re.compile(rf"^l{stage}\.(\d+)\.")
        idx = []
        for k in sd.keys():
            m = pat.match(k)
            if m:
                idx.append(int(m.group(1)))
        return max(idx) if idx else 0

    d1 = max_block(1) + 1
    d2 = max_block(2) + 1
    d3 = max_block(3) + 1

    K1 = int(sd["l1.0.c1.order_scales"].numel()) - 1
    K2 = int(sd["l2.0.c1.order_scales"].numel()) - 1
    K3 = int(sd["l3.0.c1.order_scales"].numel()) - 1

    w1 = int(sd["stem.0.weight"].shape[0])
    w2 = int(sd["l2.0.c1.combine.weight"].shape[0])
    w3 = int(sd["fc.weight"].shape[1])

    return classes, (K1, K2, K3), (d1, d2, d3), (w1, w2, w3)

def expand_stage3_layers(spec: str, depth_stage3: int):
    s = (spec or "").strip().lower()
    if s in ("all", "*", "auto", "stage3_all", "all_stage3"):
        out = []
        for b in range(int(depth_stage3)):
            out.append(f"l3.{b}.c1")
            out.append(f"l3.{b}.c2")
        return out
    return [t.strip() for t in spec.split(",") if t.strip()]


# -------------------------
# Tiny-ImageNet utilities
# -------------------------
def ensure_tinyimagenet_ready(data_root: str, allow_download: bool = False, log_fn=None) -> str:
    ds_root = os.path.join(data_root, "tiny-imagenet-200")
    if os.path.isdir(ds_root):
        return ds_root

    if not allow_download:
        raise RuntimeError(
            f"tiny-imagenet-200 not found under {data_root}. "
            "Either place it at data_root/tiny-imagenet-200, or pass --allow_download "
            "(requires internet)."
        )

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
        print("[data] Extracting...", flush=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)

    try:
        os.remove(zip_path)
    except Exception:
        pass

    if not os.path.isdir(ds_root):
        raise RuntimeError(f"Download/extract finished but {ds_root} not found.")
    return ds_root

def ensure_val_reorganized(ds_root: str) -> None:
    marker = os.path.join(ds_root, "val", ".reorg_done")
    if os.path.isfile(marker):
        return

    vdir = os.path.join(ds_root, "val")
    imgs_dir = os.path.join(vdir, "images")
    ann = os.path.join(vdir, "val_annotations.txt")
    if (not os.path.isdir(imgs_dir)) or (not os.path.isfile(ann)):
        raise RuntimeError(f"Unexpected Tiny-ImageNet val structure under: {vdir}")

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

def stratified_split_indices(labels: np.ndarray, val_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels, dtype=np.int64)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        nv = int(round(len(idx_c) * float(val_frac)))
        nv = max(1, nv)
        val_idx.append(idx_c[:nv])
        train_idx.append(idx_c[nv:])
    train_idx = np.concatenate(train_idx) if train_idx else np.arange(labels.shape[0])
    val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


# -------------------------
# Dataset with local index
# -------------------------
class IndexedDataset(Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, i):
        x, y = self.base_ds[i]
        return x, y, i

def _seed_worker(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)

def sha16(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256(arr.view(np.uint8)).hexdigest()
    return h[:16]

def build_loader_from_dataset(ds, batch_size=128, num_workers=4, seed=0, pin_memory=False):
    ds = IndexedDataset(ds)
    g = torch.Generator()
    g.manual_seed(int(seed))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        drop_last=False,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
        persistent_workers=(num_workers > 0),
    )
    return loader

def build_tiny_fit_and_test_loaders(
    data_root: str,
    fit_split: str,
    val_frac: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_idx: Optional[np.ndarray],
    val_idx: Optional[np.ndarray],
    allow_download: bool,
    pin_memory: bool,
):
    ds_root = ensure_tinyimagenet_ready(data_root, allow_download=allow_download)
    ensure_val_reorganized(ds_root)

    train_root = os.path.join(ds_root, "train")
    val_images_root = os.path.join(ds_root, "val", "images")

    mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
    tf_eval = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_full = torchvision.datasets.ImageFolder(train_root, transform=tf_eval)
    test_ds = torchvision.datasets.ImageFolder(val_images_root, transform=tf_eval)

    if (train_idx is None) or (val_idx is None):
        labels = np.array(train_full.targets, dtype=np.int64)
        train_idx, val_idx = stratified_split_indices(labels, val_frac=val_frac, seed=seed)

    fit_split = fit_split.lower()
    if fit_split == "val":
        fit_ds = Subset(train_full, val_idx.tolist())
        fit_name = f"train/val_idx({len(val_idx)})"
        fit_hash = sha16(val_idx.astype(np.int64))
    elif fit_split == "train":
        fit_ds = Subset(train_full, train_idx.tolist())
        fit_name = f"train/train_idx({len(train_idx)})"
        fit_hash = sha16(train_idx.astype(np.int64))
    else:
        raise ValueError("fit_split must be 'val' or 'train'")

    fit_loader = build_loader_from_dataset(fit_ds, batch_size=batch_size, num_workers=num_workers, seed=seed, pin_memory=pin_memory)
    test_loader = build_loader_from_dataset(test_ds, batch_size=batch_size, num_workers=num_workers, seed=seed, pin_memory=pin_memory)

    split_info = {
        "dataset": "tinyimagenet",
        "fit_split": fit_split,
        "fit_name": fit_name,
        "val_frac": float(val_frac),
        "seed": int(seed),
        "fit_idx_sha16": fit_hash,
        "n_train_full": int(len(train_full)),
        "n_test_official_val": int(len(test_ds)),
        "n_fit": int(len(fit_loader.dataset)),
    }
    return fit_loader, test_loader, split_info


# -------------------------
# Metrics
# -------------------------
def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)

def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)

def nll_from_probs(probs: np.ndarray, y_true: np.ndarray) -> float:
    p = probs[np.arange(probs.shape[0]), y_true]
    return float((-np.log(np.clip(p, 1e-12, 1.0))).mean())

def brier_from_probs(probs: np.ndarray, y_true: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(probs.shape[0]), y_true] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())

def ece_score(maxprob: np.ndarray, correct: np.ndarray, n_bins=15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = maxprob.shape[0]
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (maxprob >= lo) & (maxprob < hi) if i < n_bins - 1 else (maxprob >= lo) & (maxprob <= hi)
        if not np.any(m):
            continue
        acc = correct[m].mean()
        conf = maxprob[m].mean()
        ece += (m.sum() / N) * abs(acc - conf)
    return float(ece)

def summarize_metrics(logits: np.ndarray, y_true: np.ndarray, n_bins=15):
    probs = softmax_np(logits)
    pred = probs.argmax(axis=1).astype(np.int64)
    mp = probs.max(axis=1).astype(np.float32)
    ent = entropy_from_probs(probs).astype(np.float32)
    ln = np.linalg.norm(logits, axis=1).astype(np.float32)
    correct = (pred == y_true)
    return {
        "acc": float(correct.mean()),
        "ece": float(ece_score(mp, correct, n_bins=n_bins)),
        "nll": float(nll_from_probs(probs, y_true)),
        "brier": float(brier_from_probs(probs, y_true)),
        "entropy_mean": float(ent.mean()),
        "maxprob_mean": float(mp.mean()),
        "logit_norm_mean": float(ln.mean()),
        "pred": pred,
        "maxprob": mp,
        "entropy": ent,
        "logit_norm": ln,
        "probs": probs,
        "_y_true": y_true,
    }


# -------------------------
# Global temperature scaling
# -------------------------
def nll_from_logits_scaled(logits: np.ndarray, y_true: np.ndarray, tau: float) -> float:
    tau = float(tau)
    assert tau > 0.0
    z = logits / tau
    zmax = z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z - zmax).sum(axis=1)) + zmax.squeeze(1)
    zy = z[np.arange(z.shape[0]), y_true]
    return float((lse - zy).mean())

def fit_temperature_global(
    logits_fit: np.ndarray,
    y_fit: np.ndarray,
    tau_min: float = 0.05,
    tau_max: float = 10.0,
    n_grid: int = 200,
    n_refine: int = 60,
    n_rounds: int = 2,
    refine_radius_log: float = 0.6,
):
    assert tau_min > 0 and tau_max > tau_min
    y_fit = np.asarray(y_fit, dtype=np.int64)

    logtaus = np.linspace(np.log(tau_min), np.log(tau_max), int(n_grid), dtype=np.float64)
    taus = np.exp(logtaus)
    nlls = np.array([nll_from_logits_scaled(logits_fit, y_fit, t) for t in taus], dtype=np.float64)

    best_i = int(nlls.argmin())
    best_logtau = float(logtaus[best_i])
    best_nll = float(nlls[best_i])

    rad = float(refine_radius_log)
    for _ in range(int(n_rounds)):
        lo = best_logtau - rad
        hi = best_logtau + rad
        logtaus = np.linspace(lo, hi, int(n_refine), dtype=np.float64)
        taus = np.exp(logtaus)
        nlls = np.array([nll_from_logits_scaled(logits_fit, y_fit, t) for t in taus], dtype=np.float64)
        best_i = int(nlls.argmin())
        best_logtau = float(logtaus[best_i])
        best_nll = float(nlls[best_i])
        rad *= 0.5

    tau_star = float(np.exp(best_logtau))
    return tau_star, best_nll


# -------------------------
# Device helpers (TPU if available)
# -------------------------
def get_device(device_arg: str):
    d = (device_arg or "").lower().strip()
    if d in ("xla", "tpu", "auto"):
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except Exception:
            pass
    if d.startswith("cuda") and torch.cuda.is_available():
        return torch.device(d)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_xla_device(dev) -> bool:
    return str(dev).startswith("xla")

def mark_step_if_xla(dev):
    if is_xla_device(dev):
        try:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        except Exception:
            pass


# -------------------------
# Forward passes
# -------------------------
@torch.no_grad()
def infer_logits(model, loader, device, n_classes: int):
    model.eval()
    N = len(loader.dataset)
    logits = np.zeros((N, n_classes), dtype=np.float32)

    use_non_blocking = (not is_xla_device(device)) and (str(device).startswith("cuda"))
    for xb, _, idxb in loader:
        xb = xb.to(device, non_blocking=use_non_blocking)
        z = model(xb).detach().float().cpu().numpy()
        logits[idxb.numpy().astype(np.int64)] = z
        mark_step_if_xla(device)
    return logits

@torch.no_grad()
def collect_g_per_layer(model, loader, layer_names, device):
    """
    Returns:
      g_store[layer] = np.ndarray [N, K_layer] gate outputs
      alpha_store[layer] = np.ndarray [K_layer] order_scales
    """
    model.eval()
    name_to_mod = dict(model.named_modules())
    N = len(loader.dataset)

    g_store = {}
    alpha_store = {}
    for ln in layer_names:
        if ln not in name_to_mod:
            raise KeyError(f"Layer '{ln}' not found in model.named_modules().")
        cheb = name_to_mod[ln]
        Klen = int(cheb.order_scales.numel())
        g_store[ln] = np.zeros((N, Klen), dtype=np.float32)
        alpha_store[ln] = cheb.order_scales.detach().cpu().numpy().astype(np.float32)

    batch_idx_holder = {"idx": None}
    hooks = []

    def make_hook(layer_name):
        def fn(mod, inp, out):
            idx = batch_idx_holder["idx"]
            if idx is None:
                return
            g = out
            if torch.is_tensor(g) and g.dim() == 4:
                g = g.squeeze(-1).squeeze(-1)  # [B,K]
            g_store[layer_name][idx] = g.detach().float().cpu().numpy()
        return fn

    for ln in layer_names:
        hooks.append(name_to_mod[ln].gate.register_forward_hook(make_hook(ln)))

    use_non_blocking = (not is_xla_device(device)) and (str(device).startswith("cuda"))
    for xb, _, idxb in loader:
        batch_idx_holder["idx"] = idxb.numpy().astype(np.int64)
        xb = xb.to(device, non_blocking=use_non_blocking)
        _ = model(xb)
        batch_idx_holder["idx"] = None
        mark_step_if_xla(device)

    for h in hooks:
        h.remove()

    return g_store, alpha_store


# -------------------------
# Gain/Tilt computation
# -------------------------
def compute_gain_tilt_from_g(g_store: dict, alpha_store: dict, layer_names):
    N = next(iter(g_store.values())).shape[0]
    gain_raw = np.zeros((N,), dtype=np.float32)
    tilt_raw = np.zeros((N,), dtype=np.float32)

    for ln in layer_names:
        g = g_store[ln]                         # [N,K]
        alpha = alpha_store[ln].reshape(1, -1)  # [1,K]
        abs_s = np.abs(alpha * g).astype(np.float32)  # [N,K]

        gain_raw += abs_s.sum(axis=1)

        Klen = abs_s.shape[1]
        k = np.arange(Klen, dtype=np.float32)
        k_mean = float(k.mean())
        k_var = float(((k - k_mean) ** 2).mean()) + 1e-12
        abs_mean = abs_s.mean(axis=1, keepdims=True)
        cov = ((k.reshape(1, -1) - k_mean) * (abs_s - abs_mean)).mean(axis=1)
        tilt_raw += (cov / k_var).astype(np.float32)

    gain_raw /= float(len(layer_names))
    tilt_raw /= float(len(layer_names))

    gain = (gain_raw - gain_raw.mean()) / (gain_raw.std() + 1e-12)
    tilt = (tilt_raw - tilt_raw.mean()) / (tilt_raw.std() + 1e-12)
    return gain.astype(np.float32), tilt.astype(np.float32), gain_raw.astype(np.float32), tilt_raw.astype(np.float32)

def apply_zscore(x: np.ndarray, mu: float, sd: float):
    return ((x - mu) / (sd + 1e-12)).astype(np.float32)


# -------------------------
# Swap mapping (structured)
# -------------------------
def make_swap_to_pair_extremes_by_mean_gain(gain_raw: np.ndarray, labels: np.ndarray, k: int):
    mean_gain = np.zeros((k,), dtype=np.float64)
    for c in range(k):
        m = (labels == c)
        mean_gain[c] = float(gain_raw[m].mean()) if np.any(m) else 0.0
    order = np.argsort(mean_gain)
    swap_to = np.arange(k, dtype=np.int64)
    for i in range(k // 2):
        a = int(order[i])
        b = int(order[k - 1 - i])
        swap_to[a] = b
        swap_to[b] = a
    return swap_to, mean_gain.astype(np.float32)


# -------------------------
# Swap inference via gate override hooks
# -------------------------
@torch.no_grad()
def run_gate_override_inference(
    model,
    loader,
    layer_names,
    labels,                    # [N] cluster id on THIS loader
    swap_to,                   # [k]
    mean_g_by_layer,           # dict ln -> torch [k,K] prototype gate values
    g_orig_by_layer=None,      # dict ln -> np [N,K] ORIGINAL gates on THIS loader (required for partial swap)
    swap_beta: float = 1.0,    # partial swap strength in [0,1]
    abs_alpha_by_layer=None,   # dict ln -> torch [1,K]
    amp_base_by_layer=None,    # dict ln -> np [N]  (sum |alpha| g_orig) per sample
    device=None,
    n_classes=200,
    mode="plain",              # "plain" or "amp_preserve"
    eps=1e-12,
):
    """
    If swap_beta==1.0: behaves like original hard swap (replaces g with prototype).
    If swap_beta in (0,1): partial swap: g_use = (1-beta)*g_orig + beta*g_proto (then clamp).
    For amp_preserve: after blending, re-scale g_use so that sum(|alpha| g_use) matches base amplitude.
    """
    beta = float(swap_beta)
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"swap_beta must be in [0,1], got {beta}")

    if beta < 1.0:
        if g_orig_by_layer is None:
            raise ValueError("Partial swap requires g_orig_by_layer (original gates) to blend with prototypes.")

    model.eval()
    name_to_mod = dict(model.named_modules())
    N = len(loader.dataset)
    logits_out = np.zeros((N, n_classes), dtype=np.float32)

    override_holder = {ln: None for ln in layer_names}
    hooks = []

    def make_override_hook(layer_name):
        def _hook(gate_module, inp, out):
            g_new = override_holder[layer_name]
            if g_new is None:
                return out
            g_new = g_new.to(dtype=out.dtype)
            if out.dim() == 4:
                return g_new.view(g_new.size(0), g_new.size(1), 1, 1)
            return g_new
        return _hook

    for ln in layer_names:
        hooks.append(name_to_mod[ln].gate.register_forward_hook(make_override_hook(ln)))

    amp_ratio_collect = {ln: [] for ln in layer_names}
    use_non_blocking = (not is_xla_device(device)) and (str(device).startswith("cuda"))

    for xb, _, idxb in loader:
        idx = idxb.numpy().astype(np.int64)
        xb = xb.to(device, non_blocking=use_non_blocking)

        src_c = labels[idx]
        tgt_c = swap_to[src_c]
        tgt_t = torch.from_numpy(tgt_c).to(device=device, dtype=torch.long)

        B = xb.shape[0]
        for ln in layer_names:
            # prototype gate for swapped-to cluster
            g_proto = mean_g_by_layer[ln].index_select(0, tgt_t)  # (B,K)

            # original gate (for partial swap)
            if beta < 1.0:
                g_orig = torch.from_numpy(g_orig_by_layer[ln][idx]).to(device=device, dtype=torch.float32)
                g_mix = torch.clamp((1.0 - beta) * g_orig + beta * g_proto, 0.0, 1.0)
            else:
                g_mix = g_proto

            if mode == "plain":
                g_use = g_mix

            elif mode == "amp_preserve":
                # keep amplitude comparable to base: A = sum_k |alpha_k| g_k
                assert abs_alpha_by_layer is not None and amp_base_by_layer is not None
                abs_alpha = abs_alpha_by_layer[ln]  # (1,K)
                A_base = torch.from_numpy(amp_base_by_layer[ln][idx]).to(device=device, dtype=torch.float32).view(B, 1)

                # amplitude of current mixed gate
                A_mix = (abs_alpha * g_mix).sum(dim=1, keepdim=True)
                scale = A_base / (A_mix + eps)
                g_scaled = torch.clamp(g_mix * scale, 0.0, 1.0)

                # diagnostic: how well we preserved amplitude after clamp
                A_scaled = (abs_alpha * g_scaled).sum(dim=1, keepdim=True)
                ratio = (A_scaled / (A_base + eps)).detach().cpu().numpy().reshape(-1).astype(np.float32)
                amp_ratio_collect[ln].append(ratio)

                g_use = g_scaled
            else:
                raise ValueError(f"Unknown mode={mode}")

            override_holder[ln] = g_use

        z = model(xb).detach().float().cpu().numpy()
        logits_out[idx] = z
        mark_step_if_xla(device)

        # clear holders to avoid accidental reuse
        for ln in layer_names:
            override_holder[ln] = None

    for h in hooks:
        h.remove()

    amp_diag = None
    if mode == "amp_preserve":
        amp_diag = {}
        for ln in layer_names:
            r = np.concatenate(amp_ratio_collect[ln], axis=0) if amp_ratio_collect[ln] else None
            amp_diag[ln] = None if r is None else {
                "mean": float(r.mean()),
                "p10": float(np.percentile(r, 10)),
                "p50": float(np.percentile(r, 50)),
                "p90": float(np.percentile(r, 90)),
                "min": float(r.min()),
                "max": float(r.max()),
            }

    return logits_out, amp_diag


# -------------------------
# Logit-norm matching
# -------------------------
def logit_norm_match(logits_src: np.ndarray, logits_ref: np.ndarray, eps=1e-12):
    src_ln = np.linalg.norm(logits_src, axis=1) + eps
    ref_ln = np.linalg.norm(logits_ref, axis=1)
    scale = (ref_ln / src_ln).astype(np.float32)
    return (logits_src * scale[:, None]).astype(np.float32)


# -------------------------
# Plotting
# -------------------------
def reliability_curve(maxprob: np.ndarray, correct: np.ndarray, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confs, accs = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (maxprob >= lo) & (maxprob < hi) if i < n_bins - 1 else (maxprob >= lo) & (maxprob <= hi)
        if not np.any(m):
            continue
        confs.append(float(maxprob[m].mean()))
        accs.append(float(correct[m].mean()))
    return np.array(confs, dtype=np.float32), np.array(accs, dtype=np.float32)

def plot_reliability_curve(ax, maxprob, correct, label, color, marker="o", linestyle="-", n_bins=15, alpha=1.0, markersize=3):
    confs, accs = reliability_curve(maxprob, correct, n_bins=n_bins)
    ax.plot(confs, accs, marker=marker, label=label, color=color, linestyle=linestyle, alpha=alpha, markersize=markersize)

def plot_class_x_cluster_heatmap(y_true, labels, k, n_classes, out_path, do_show=False):
    # P(cluster|class): row-normalized
    mat = np.zeros((n_classes, k), dtype=np.float32)
    cnt = np.zeros((n_classes, k), dtype=np.int64)

    for c in range(n_classes):
        m = (y_true == c)
        if not np.any(m):
            continue
        for j in range(k):
            v = np.sum(labels[m] == j)
            cnt[c, j] = int(v)
        mat[c] = cnt[c] / max(int(m.sum()), 1)

    fig, ax = plt.subplots(figsize=(3.6, 8.0), dpi=800)
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="magma")

    ax.set_ylabel("True class $y$")
    ax.set_xlabel(r"Order-mixing program $c$ (k-means id)")
    ax.set_xticks(np.arange(k))
    ax.set_frame_on(False)

    if n_classes > 20:
        step = 10
        yt = list(range(0, n_classes, step))
        ax.set_yticks(yt)
        ax.set_yticklabels([str(i) for i in yt])
    else:
        ax.set_yticks(np.arange(n_classes))
        ax.set_yticklabels([str(i) for i in range(n_classes)])

    if n_classes <= 30:
        for i in range(n_classes):
            for j in range(k):
                val = mat[i, j]
                txt_color = "white" if im.norm(val) < 0.55 else "black"
                outline = "black" if txt_color == "white" else "white"
                ax.text(
                    j, i, f"{val:.2f}\n({cnt[i,j]})",
                    ha="center", va="center",
                    color=txt_color,
                    fontsize=4.0,
                    path_effects=[pe.withStroke(linewidth=1.0, foreground=outline)]
                )

    cbar = fig.colorbar(im, ax=ax, aspect=50)
    cbar.set_label(r"Row-normalized $P(c \mid y)$")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    if do_show:
        plt.show()
    plt.close(fig)

def plot_fig2_reliability_base_vs_swap(base_metrics, swap_metrics, out_path, n_bins=15, do_show=False):
    fig, ax = plt.subplots(figsize=(3.5, 1.95), dpi=800)
    ax.plot([0, 1], [0, 1], "--", linewidth=1.0, alpha=0.8, color=COL_AUX)

    plot_reliability_curve(
        ax,
        base_metrics["maxprob"],
        (base_metrics["pred"] == base_metrics["_y_true"]),
        label=f"base (ECE={base_metrics['ece']:.3f})",
        color=COL_BASE,
        n_bins=n_bins,
    )
    plot_reliability_curve(
        ax,
        swap_metrics["maxprob"],
        (swap_metrics["pred"] == swap_metrics["_y_true"]),
        label=f"paired-extremes swap (ECE={swap_metrics['ece']:.3f})",
        color=COL_SWAP,
        n_bins=n_bins,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Confidence (bin-avg $\max_j p_\theta(j\mid x)$)")
    ax.set_ylabel(r"Accuracy (bin-avg $\mathbb{1}[\hat y=y]$)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    if do_show:
        plt.show()
    plt.close(fig)

def plot_fig3_temp_attribution(base_m, amp_m, amp_nm_m, out_prefix, n_bins=15, do_show=False):
    out_a = out_prefix + "_a_hist.png"
    fig, ax = plt.subplots(figsize=(2.10, 1.95), dpi=800)
    ln_base = base_m["logit_norm"]
    ln_amp  = amp_m["logit_norm"]
    ln_nm   = amp_nm_m["logit_norm"]

    ax.hist(ln_amp, bins=60, alpha=0.7, label="amp-preserve", color=COL_SWAP)
    ax.hist(ln_nm,  bins=60, alpha=0.2, label="logit-norm matched", color=COL_SWAP)
    ax.hist(ln_base, bins=60, histtype="step", linewidth=1.2, label="base", color=COL_BASE)

    ax.set_xlabel(r"$\|\mathbf{z}\|_2$")
    ax.set_ylabel("Count")
    ax.legend(loc="best", frameon=True, fontsize=5)

    fig.tight_layout()
    fig.savefig(out_a, bbox_inches="tight")
    fig.savefig(out_a.replace(".png", ".pdf"), bbox_inches="tight")
    if do_show:
        plt.show()
    plt.close(fig)

    out_b = out_prefix + "_b_reliability.png"
    fig, ax = plt.subplots(figsize=(2.90, 1.95), dpi=800)

    ax.plot([0, 1], [0, 1], "--", linewidth=1.0, alpha=0.8, color=COL_AUX)
    plot_reliability_curve(ax, base_m["maxprob"], (base_m["pred"] == base_m["_y_true"]),
                           label=f"base (ECE={base_m['ece']:.3f})", n_bins=n_bins,
                           color=COL_BASE, marker="o", linestyle="-")
    plot_reliability_curve(ax, amp_m["maxprob"], (amp_m["pred"] == amp_m["_y_true"]),
                           label=f"amp-preserve (ECE={amp_m['ece']:.3f})", n_bins=n_bins,
                           color=COL_SWAP, marker="o", linestyle="-")
    plot_reliability_curve(ax, amp_nm_m["maxprob"], (amp_nm_m["pred"] == amp_nm_m["_y_true"]),
                           label=f"norm-matched (ECE={amp_nm_m['ece']:.3f})", n_bins=n_bins,
                           color=COL_SWAP, marker="s", linestyle="--")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Confidence (bin-avg $\max_j p_\theta(j\mid x)$)")
    ax.set_ylabel(r"Accuracy (bin-avg $\mathbb{1}[\hat y=y]$)")
    ax.legend(loc="best", frameon=True, fontsize=5)

    fig.tight_layout()
    fig.savefig(out_b, bbox_inches="tight")
    fig.savefig(out_b.replace(".png", ".pdf"), bbox_inches="tight")
    if do_show:
        plt.show()
    plt.close(fig)

    out_c = out_prefix + "_c_metrics.png"
    col_headers = ["Base", "Amp-\npreserve", "Logit-\nnorm\nmatched"]
    row_labels = [r"Acc $\uparrow$",
                  r"ECE $\downarrow$",
                  r"NLL $\downarrow$",
                  r"Brier $\downarrow$",
                  r"Mean$\|z\|_2$"]

    data = [
        [base_m["acc"], base_m["ece"], base_m["nll"], base_m["brier"], base_m["logit_norm_mean"]],
        [amp_m["acc"],  amp_m["ece"],  amp_m["nll"],  amp_m["brier"],  amp_m["logit_norm_mean"]],
        [amp_nm_m["acc"], amp_nm_m["ece"], amp_nm_m["nll"], amp_nm_m["brier"], amp_nm_m["logit_norm_mean"]],
    ]
    vals = list(map(list, zip(*data)))
    cell_text = [[f"{v:.4f}" for v in row] for row in vals]

    fig_w, fig_h = 2.16, 1.8
    fs_body = 7.0
    fs_head = 6.3
    lw_topbot = 1.0
    lw_mid = 0.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    left, right = 0.01, 0.995
    top, bottom = 0.97, 0.055
    header_h = 0.26
    y_midrule = top - header_h
    y_header = (top + y_midrule) / 2

    n_rows = len(row_labels)
    body_h = y_midrule - bottom
    row_h = body_h / n_rows
    y_rows = y_midrule - row_h * (np.arange(n_rows) + 0.5)

    label_w = 0.28
    data_w = (right - left - label_w) / 3.0
    x_label = left + 0.005
    x_cols = [left + label_w + data_w * (i + 0.5) for i in range(3)]

    ax.hlines(top, left, right, colors="black", linewidth=lw_topbot)
    ax.hlines(y_midrule, left, right, colors="black", linewidth=lw_mid)
    ax.hlines(bottom, left, right, colors="black", linewidth=lw_topbot)

    for i, h in enumerate(col_headers):
        ax.text(x_cols[i], y_header, h, ha="center", va="center",
                fontsize=fs_head, linespacing=0.9, multialignment="center")

    for r, (lab, row) in enumerate(zip(row_labels, cell_text)):
        y = y_rows[r]
        ax.text(x_label, y, lab, ha="left", va="center", fontsize=fs_body)
        for c in range(3):
            ax.text(x_cols[c], y, row[c], ha="center", va="center", fontsize=fs_body)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    pad_y = 9 / 72
    bbox_y = Bbox.from_extents(tight.x0, tight.y0 - pad_y, tight.x1, tight.y1 + pad_y)

    fig.savefig(out_c, bbox_inches=bbox_y)
    fig.savefig(out_c.replace(".png", ".pdf"), bbox_inches=bbox_y)
    if do_show:
        plt.show()
    plt.close(fig)


# -------------------------
# Sanity suite (label leakage defense)
# -------------------------
def mean_per_class_purity(y_true: np.ndarray, labels: np.ndarray, n_classes: int, k: int) -> float:
    pur = []
    for c in range(n_classes):
        m = (y_true == c)
        if m.sum() == 0:
            continue
        cnt = np.bincount(labels[m], minlength=k).astype(np.float64)
        pur.append(cnt.max() / cnt.sum())
    return float(np.mean(pur)) if pur else 0.0

def permutation_pvalue(obs: float, null: np.ndarray) -> float:
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))

def run_sanity_suite(
    out_dir,
    y_true,
    labels,
    k,
    n_classes,
    swap_logits,
    amp_logits,
    amp_nm_logits,
    seed=0,
    n_bins=15,
    do_show=False,
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    y_perm = rng.permutation(y_true)

    figS_path = os.path.join(out_dir, "FigS_labelperm_class_x_cluster.png")
    plot_class_x_cluster_heatmap(y_true=y_perm, labels=labels, k=k, n_classes=n_classes, out_path=figS_path, do_show=do_show)
    print("[sanity] wrote", figS_path, "(y_true permuted; should NOT look pure/diagonal)")

    m_swap_true = summarize_metrics(swap_logits, y_true, n_bins=n_bins)
    m_swap_perm = summarize_metrics(swap_logits, y_perm, n_bins=n_bins)

    obs_nmi = normalized_mutual_info_score(y_true, labels)
    obs_pur = mean_per_class_purity(y_true, labels, n_classes=n_classes, k=k)

    B = 1000
    nmi_null = np.zeros(B, dtype=np.float32)
    pur_null = np.zeros(B, dtype=np.float32)
    for b in range(B):
        yp = rng.permutation(y_true)
        nmi_null[b] = normalized_mutual_info_score(yp, labels)
        pur_null[b] = mean_per_class_purity(yp, labels, n_classes=n_classes, k=k)

    p_nmi = permutation_pvalue(obs_nmi, nmi_null)
    p_pur = permutation_pvalue(obs_pur, pur_null)

    report = {
        "hash_labels": sha16(labels.astype(np.int64)),
        "hash_swap_logits": sha16(swap_logits.astype(np.float32)),
        "hash_amp_logits": sha16(amp_logits.astype(np.float32)),
        "hash_amp_nm_logits": sha16(amp_nm_logits.astype(np.float32)),
        "swap_metrics_true": {kk: float(m_swap_true[kk]) for kk in ["acc","ece","nll","brier","logit_norm_mean"]},
        "swap_metrics_perm": {kk: float(m_swap_perm[kk]) for kk in ["acc","ece","nll","brier","logit_norm_mean"]},
        "obs_nmi": float(obs_nmi),
        "p_nmi": float(p_nmi),
        "obs_mean_per_class_purity": float(obs_pur),
        "p_purity": float(p_pur),
        "note": "programs/prototypes fit without y_true; y_true only used for evaluation + heatmap rows."
    }

    rep_path = os.path.join(out_dir, "sanity_report.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print("[sanity] wrote", rep_path)

    print("[sanity] swap logits fixed; metrics change when labels permuted:")
    print("  swap(true): acc={:.4f} ece={:.4f} nll={:.4f}".format(
        report["swap_metrics_true"]["acc"], report["swap_metrics_true"]["ece"], report["swap_metrics_true"]["nll"]))
    print("  swap(perm): acc={:.4f} ece={:.4f} nll={:.4f}".format(
        report["swap_metrics_perm"]["acc"], report["swap_metrics_perm"]["ece"], report["swap_metrics_perm"]["nll"]))
    print("[sanity] NMI={:.4f} (p={:.3g}), mean purity={:.4f} (p={:.3g})".format(
        report["obs_nmi"], report["p_nmi"], report["obs_mean_per_class_purity"], report["p_purity"]))


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, default='',
                    help="Best model checkpoint (state_dict OR dict containing key 'model' or 'state_dict').")
    ap.add_argument("--train_ckpt", type=str, default="",
                    help="Optional training checkpoint that contains train_idx/val_idx and maybe model_cfg.")
    ap.add_argument("--trusted_ckpt", action="store_true", default=True,
                    help="Allow safe_torch_load fallback to weights_only=False.")

    ap.add_argument("--data_root", type=str, default="",
                    help="Folder containing tiny-imagenet-200/ (or where it will be downloaded if --allow_download).")
    ap.add_argument("--allow_download", action="store_true", default=False,
                    help="If tiny-imagenet-200 is missing, download it (requires internet).")

    ap.add_argument("--out_dir", type=str, default="./order_mixing_out")
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true", default=False)
    ap.add_argument("--device", type=str, default="auto")

    # analysis config
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--stage3_layers", type=str, default="ALL")
    ap.add_argument("--swap_to", type=str, default="")
    ap.add_argument("--fit_split", type=str, default="val", choices=["val", "train"])
    ap.add_argument("--val_frac", type=float, default=0.10)

    # partial swap strength
    ap.add_argument("--swap_beta", type=float, default=0.8,
                    help="Partial swap strength beta in [0,1]. 1.0=hard swap (original), 0.0=no intervention.")

    # model knobs
    ap.add_argument("--drop_rate", type=float, default=0.15)
    ap.add_argument("--lambda_lap", type=float, default=0.25)
    ap.add_argument("--realization", type=str, default="concat")
    ap.add_argument("--gate_mode", type=str, default="on")
    ap.add_argument("--stabilize_cheb", type=int, default=0)
    ap.add_argument("--channels_last", action="store_true", default=False)
    ap.add_argument("--auto_model_cfg", action="store_true", default=True,
                    help="If train_ckpt contains model_cfg, adopt those knobs for inference.")

    # permutation sweep
    ap.add_argument("--n_perm", type=int, default=20)
    ap.add_argument("--skip_perm", action="store_true", default=False)

    # outputs
    ap.add_argument("--no_show", action="store_true", default=False)
    ap.add_argument("--no_sanity", action="store_true", default=False)

    args, _ = ap.parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Validate swap_beta early
    if not (0.0 <= float(args.swap_beta) <= 1.0):
        raise ValueError(f"--swap_beta must be in [0,1], got {args.swap_beta}")

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args.device)

    print(f"[env] PJRT_DEVICE={os.environ.get('PJRT_DEVICE','')} | device={device}", flush=True)
    print(f"[env] torch={torch.__version__} torchvision={torchvision.__version__} sklearn={sklearn.__version__}", flush=True)
    print(f"[cfg] swap_beta={float(args.swap_beta):.3f} (1.0=hard swap, 0.0=no-op)", flush=True)

    # Import model from repo
    from model import ChebResNet

    # ---- Load best checkpoint
    ckpt_obj = safe_torch_load(Path(args.ckpt), map_location="cpu", trusted=args.trusted_ckpt)
    sd = strip_module_prefix(extract_state_dict(ckpt_obj))

    # ---- Optional: split indices + model_cfg from training checkpoint
    train_idx = val_idx = None
    model_cfg = None
    if args.train_ckpt.strip():
        tr_obj = safe_torch_load(Path(args.train_ckpt), map_location="cpu", trusted=args.trusted_ckpt)
        if isinstance(tr_obj, dict):
            train_idx = tr_obj.get("train_idx", None)
            val_idx = tr_obj.get("val_idx", None)
            if train_idx is not None:
                train_idx = np.array(train_idx, dtype=np.int64)
            if val_idx is not None:
                val_idx = np.array(val_idx, dtype=np.int64)
            print(f"[split] loaded from train_ckpt: train_idx={None if train_idx is None else len(train_idx)} "
                  f"val_idx={None if val_idx is None else len(val_idx)}", flush=True)

            model_cfg = tr_obj.get("model_cfg", None)
            if (model_cfg is not None) and args.auto_model_cfg:
                args.lambda_lap = float(model_cfg.get("lambda_lap", args.lambda_lap))
                args.realization = str(model_cfg.get("realization", args.realization))
                args.gate_mode = str(model_cfg.get("gate_mode", args.gate_mode))
                args.stabilize_cheb = int(model_cfg.get("stabilize_cheb", args.stabilize_cheb))
                args.drop_rate = float(model_cfg.get("drop_rate", args.drop_rate))
                print("[model_cfg] adopted from train_ckpt:", json.dumps({
                    "lambda_lap": args.lambda_lap,
                    "realization": args.realization,
                    "gate_mode": args.gate_mode,
                    "stabilize_cheb": args.stabilize_cheb,
                    "drop_rate": args.drop_rate,
                }, indent=2), flush=True)

    # ---- Infer arch from state_dict
    classes, K_stages, depth_stages, widths = infer_arch_from_sd(sd)
    print(f"[arch] classes={classes} K={K_stages} depth={depth_stages} widths={widths}", flush=True)

    # ---- Build model
    model = ChebResNet(
        classes=classes,
        K=K_stages,
        depth=depth_stages,
        widths=widths,
        drop_rate=float(args.drop_rate),
        lap=float(args.lambda_lap),
        realization=str(args.realization),
        gate_mode=str(args.gate_mode),
        stabilize_cheb=bool(int(args.stabilize_cheb)),
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    model = model.to(device).eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Layers (stage 3)
    layer_names = expand_stage3_layers(args.stage3_layers, depth_stage3=depth_stages[2])
    name_to_mod = dict(model.named_modules())
    not_found = [ln for ln in layer_names if ln not in name_to_mod]
    if not_found:
        print("[warn] some requested layers not found in model.named_modules():", not_found[:20], flush=True)
        layer_names = [ln for ln in layer_names if ln in name_to_mod]
    print("[cfg] layers used:", layer_names, flush=True)
    print("[cfg] num layers:", len(layer_names), flush=True)

    # ---- Build loaders
    fit_loader, test_loader, split_info = build_tiny_fit_and_test_loaders(
        data_root=args.data_root,
        fit_split=args.fit_split,
        val_frac=args.val_frac,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_idx=train_idx,
        val_idx=val_idx,
        allow_download=args.allow_download,
        pin_memory=args.pin_memory,
    )
    print("[split]", json.dumps(split_info, indent=2), flush=True)
    with open(os.path.join(args.out_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # =========================
    # (1) FIT: codebook + prototypes
    # =========================
    print("\n[FIT] collecting g(x) on fit split ...", flush=True)
    g_fit, alpha_store = collect_g_per_layer(model, fit_loader, layer_names, device=device)
    _, _, gain_raw_fit, tilt_raw_fit = compute_gain_tilt_from_g(g_fit, alpha_store, layer_names)

    muG, sdG = float(gain_raw_fit.mean()), float(gain_raw_fit.std() + 1e-12)
    muT, sdT = float(tilt_raw_fit.mean()), float(tilt_raw_fit.std() + 1e-12)

    gain_fit = apply_zscore(gain_raw_fit, muG, sdG)
    tilt_fit = apply_zscore(tilt_raw_fit, muT, sdT)
    X_fit = np.stack([gain_fit, tilt_fit], axis=1).astype(np.float32)

    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=20, init="k-means++", max_iter=300)
    labels_fit = km.fit_predict(X_fit).astype(np.int64)
    counts_fit = np.bincount(labels_fit, minlength=args.k)
    print("[FIT] cluster counts:", counts_fit.tolist(), flush=True)
    if np.any(counts_fit == 0):
        print("[warn] empty FIT clusters exist; prototypes for those clusters will be zeros.", flush=True)

    # swap mapping pi
    if args.swap_to.strip():
        swap_to = np.array([int(x) for x in args.swap_to.split(",")], dtype=np.int64)
        assert swap_to.shape[0] == args.k
        mean_gain = None
        print("[swap_to] manual:", swap_to.tolist(), flush=True)
    else:
        swap_to, mean_gain = make_swap_to_pair_extremes_by_mean_gain(gain_raw_fit, labels_fit, k=args.k)
        print("[swap_to] paired extremes by mean FIT gain:", swap_to.tolist(), flush=True)
        print("[swap_to] mean_gain(FIT):", mean_gain.tolist(), flush=True)

    # prototypes (mean gate per cluster, per layer)
    mean_g_by_layer = {}
    abs_alpha_by_layer = {}
    for ln in layer_names:
        cheb = name_to_mod[ln]
        alpha = cheb.order_scales.detach().cpu().numpy().astype(np.float32)  # [K]
        abs_alpha = np.abs(alpha).astype(np.float32)

        g = g_fit[ln]  # [N_fit,K]
        s = (alpha.reshape(1, -1) * g).astype(np.float32)  # [N_fit,K]

        ms = np.zeros((args.k, s.shape[1]), dtype=np.float32)
        for c in range(args.k):
            m = (labels_fit == c)
            ms[c] = s[m].mean(axis=0) if np.any(m) else 0.0

        denom = np.where(np.abs(alpha) < 1e-8, 1.0, alpha).astype(np.float32)
        mg = (ms / denom.reshape(1, -1)).astype(np.float32)
        mg = np.clip(mg, 0.0, 1.0)

        mean_g_by_layer[ln] = torch.from_numpy(mg).to(device=device, dtype=torch.float32)
        abs_alpha_by_layer[ln] = torch.from_numpy(abs_alpha.reshape(1, -1)).to(device=device, dtype=torch.float32)

    fit_art = {
        "fit_split": split_info["fit_split"],
        "muG": muG, "sdG": sdG, "muT": muT, "sdT": sdT,
        "swap_to": swap_to.tolist(),
        "swap_beta": float(args.swap_beta),
        "mean_gain_fit": mean_gain.tolist() if mean_gain is not None else None,
        "km_centers": km.cluster_centers_.astype(np.float32).tolist(),
        "cluster_counts_fit": counts_fit.tolist(),
    }
    with open(os.path.join(args.out_dir, "fit_artifacts.json"), "w") as f:
        json.dump(fit_art, f, indent=2)
    print("[FIT] wrote fit_artifacts.json", flush=True)

    # =========================
    # (2) TEST: base + swaps + temp scaling
    # =========================
    print("\n[TEST] base logits on official val/ ...", flush=True)
    base_logits = infer_logits(model, test_loader, device=device, n_classes=classes)

    # y_true (TEST) for evaluation only
    N_test = len(test_loader.dataset)
    y_true = np.zeros((N_test,), dtype=np.int64)
    for _, yb, idxb in test_loader:
        y_true[idxb.numpy().astype(np.int64)] = yb.numpy().astype(np.int64)
    print("[TEST] y_true loaded (eval only)", flush=True)

    print("[TEST] collecting g(x) for program assignment ...", flush=True)
    g_test, alpha_store_test = collect_g_per_layer(model, test_loader, layer_names, device=device)
    _, _, gain_raw_test, tilt_raw_test = compute_gain_tilt_from_g(g_test, alpha_store_test, layer_names)

    gain_test = apply_zscore(gain_raw_test, muG, sdG)
    tilt_test = apply_zscore(tilt_raw_test, muT, sdT)
    X_test = np.stack([gain_test, tilt_test], axis=1).astype(np.float32)
    labels_test = km.predict(X_test).astype(np.int64)

    counts_test = np.bincount(labels_test, minlength=args.k)
    print("[TEST] cluster counts:", counts_test.tolist(), flush=True)

    # amplitude on base gates (for amp-preserve)
    amp_base_by_layer = {}
    for ln in layer_names:
        cheb = name_to_mod[ln]
        alpha = cheb.order_scales.detach().cpu().numpy().astype(np.float32)
        abs_alpha = np.abs(alpha).astype(np.float32)
        g = g_test[ln]
        amp_base_by_layer[ln] = (abs_alpha.reshape(1, -1) * g).sum(axis=1).astype(np.float32)

    # metrics: base
    base_m = summarize_metrics(base_logits, y_true, n_bins=args.n_bins)
    print(f"[base] acc={base_m['acc']:.4f} ece={base_m['ece']:.4f} nll={base_m['nll']:.4f} "
          f"brier={base_m['brier']:.4f} mean||z||={base_m['logit_norm_mean']:.4f}", flush=True)

    # -------------------------
    # Global temperature scaling baseline (fit on FIT split)
    # -------------------------
    print("\n[temp] fitting global temperature on FIT split ...", flush=True)
    fit_logits = infer_logits(model, fit_loader, device=device, n_classes=classes)
    N_fit = len(fit_loader.dataset)
    y_fit = np.zeros((N_fit,), dtype=np.int64)
    for _, yb, idxb in fit_loader:
        y_fit[idxb.numpy().astype(np.int64)] = yb.numpy().astype(np.int64)

    tau_star, fit_nll = fit_temperature_global(
        fit_logits, y_fit,
        tau_min=0.05, tau_max=10.0,
        n_grid=200, n_refine=60, n_rounds=2, refine_radius_log=0.6
    )
    ts_logits = (base_logits / float(tau_star)).astype(np.float32)
    ts_m = summarize_metrics(ts_logits, y_true, n_bins=args.n_bins)
    print(f"[temp] tau*={tau_star:.6f} (fit NLL={fit_nll:.6f})", flush=True)
    print(f"[temp] TEST: acc={ts_m['acc']:.4f} ece={ts_m['ece']:.4f} nll={ts_m['nll']:.4f} "
          f"brier={ts_m['brier']:.4f} mean||z||={ts_m['logit_norm_mean']:.4f}", flush=True)

    temp_path = os.path.join(args.out_dir, "temp_scaling_global.json")
    with open(temp_path, "w") as f:
        json.dump({
            "fit_split": split_info["fit_split"],
            "tau_star": float(tau_star),
            "fit_nll": float(fit_nll),
            "test_metrics": {k: float(ts_m[k]) for k in ["acc","ece","nll","brier","logit_norm_mean"]},
            "n_bins": int(args.n_bins),
        }, f, indent=2)
    print("[temp] wrote", temp_path, flush=True)

    # structured swap (plain) with partial swap beta
    print(f"\n[swap] paired-extremes swap (plain, beta={float(args.swap_beta):.3f}) ...", flush=True)
    swap_logits, _ = run_gate_override_inference(
        model, test_loader, layer_names,
        labels=labels_test, swap_to=swap_to,
        mean_g_by_layer=mean_g_by_layer,
        g_orig_by_layer=g_test,
        swap_beta=float(args.swap_beta),
        device=device, n_classes=classes,
        mode="plain",
    )
    swap_m = summarize_metrics(swap_logits, y_true, n_bins=args.n_bins)
    print(f"[swap] acc={swap_m['acc']:.4f} ece={swap_m['ece']:.4f} nll={swap_m['nll']:.4f} "
          f"brier={swap_m['brier']:.4f} mean||z||={swap_m['logit_norm_mean']:.4f}", flush=True)

    # amp-preserve swap with partial swap beta
    print(f"[amp ] amp-preserve swap (beta={float(args.swap_beta):.3f}) ...", flush=True)
    amp_logits, amp_diag = run_gate_override_inference(
        model, test_loader, layer_names,
        labels=labels_test, swap_to=swap_to,
        mean_g_by_layer=mean_g_by_layer,
        g_orig_by_layer=g_test,
        swap_beta=float(args.swap_beta),
        abs_alpha_by_layer=abs_alpha_by_layer,
        amp_base_by_layer=amp_base_by_layer,
        device=device, n_classes=classes,
        mode="amp_preserve",
    )
    amp_m = summarize_metrics(amp_logits, y_true, n_bins=args.n_bins)
    print(f"[amp ] acc={amp_m['acc']:.4f} ece={amp_m['ece']:.4f} nll={amp_m['nll']:.4f} "
          f"brier={amp_m['brier']:.4f} mean||z||={amp_m['logit_norm_mean']:.4f}", flush=True)

    # logit-norm matching (temperature attribution)
    print("[attr] logit-norm matching (amp -> base norm) ...", flush=True)
    amp_nm_logits = logit_norm_match(amp_logits, base_logits)
    amp_nm_m = summarize_metrics(amp_nm_logits, y_true, n_bins=args.n_bins)
    print(f"[nm  ] acc={amp_nm_m['acc']:.4f} ece={amp_nm_m['ece']:.4f} nll={amp_nm_m['nll']:.4f} "
          f"brier={amp_nm_m['brier']:.4f} mean||z||={amp_nm_m['logit_norm_mean']:.4f}", flush=True)

    # ---- Figures
    do_show = (not args.no_show)

    fig1_path = os.path.join(args.out_dir, "Fig1_class_x_cluster.png")
    plot_class_x_cluster_heatmap(y_true=y_true, labels=labels_test, k=args.k, n_classes=classes, out_path=fig1_path, do_show=do_show)
    print("[ok] wrote Fig1_class_x_cluster.{png,pdf}", flush=True)

    fig2_path = os.path.join(args.out_dir, "Fig2_reliability_base_vs_swap.png")
    plot_fig2_reliability_base_vs_swap(base_m, swap_m, fig2_path, n_bins=args.n_bins, do_show=do_show)
    print("[ok] wrote Fig2_reliability_base_vs_swap.{png,pdf}", flush=True)

    fig3_prefix = os.path.join(args.out_dir, "Fig3_temperature_attribution")
    plot_fig3_temp_attribution(base_m, amp_m, amp_nm_m, fig3_prefix, n_bins=args.n_bins, do_show=do_show)
    print("[ok] wrote Fig3_temperature_attribution_{a,b,c}.{png,pdf}", flush=True)

    # ---- Table1: permutation sweep (structured swap only, plain; uses SAME beta)
    perm_csv = os.path.join(args.out_dir, "Table1_perm_sweep_detail.csv")
    perm_sum_csv = os.path.join(args.out_dir, "Table1_perm_sweep_summary.csv")
    if args.skip_perm:
        print("[perm] skipped", flush=True)
    else:
        print(f"[perm] permutation sweep n_perm={args.n_perm} (structured swap, plain; beta={float(args.swap_beta):.3f}) ...", flush=True)
        rng = np.random.default_rng(args.seed)
        rows = []
        for t in range(int(args.n_perm)):
            perm = rng.permutation(args.k).astype(np.int64)
            logits_t, _ = run_gate_override_inference(
                model, test_loader, layer_names,
                labels=labels_test, swap_to=perm,
                mean_g_by_layer=mean_g_by_layer,
                g_orig_by_layer=g_test,
                swap_beta=float(args.swap_beta),
                device=device, n_classes=classes,
                mode="plain",
            )
            m = summarize_metrics(logits_t, y_true, n_bins=args.n_bins)
            rows.append([t, m["acc"], m["ece"], m["nll"], m["brier"], m["entropy_mean"], m["logit_norm_mean"]])
            print(f"  [perm {t:02d}] acc={m['acc']:.4f} ece={m['ece']:.4f} nll={m['nll']:.4f} brier={m['brier']:.4f}", flush=True)

        rows = np.array(rows, dtype=np.float64)
        header = "perm_id,acc,ece,nll,brier,entropy_mean,logit_norm_mean"
        np.savetxt(perm_csv, rows, delimiter=",", header=header, comments="")
        print("[perm] wrote", perm_csv, flush=True)

        vals = rows[:, 1:]
        mean = vals.mean(axis=0)
        std = vals.std(axis=0)
        sum_header = "stat,acc,ece,nll,brier,entropy_mean,logit_norm_mean"
        sum_rows = np.vstack([
            np.concatenate([[0], mean]),
            np.concatenate([[1], std]),
        ])
        np.savetxt(perm_sum_csv, sum_rows, delimiter=",", header=sum_header, comments="")
        print("[perm] wrote", perm_sum_csv, flush=True)

    # ---- Save artifacts
    out_npz = os.path.join(args.out_dir, "tiny_order_mixing_artifacts.npz")
    np.savez(
        out_npz,
        y_true=y_true,
        labels_test=labels_test,
        swap_to=swap_to,
        swap_beta=np.array([float(args.swap_beta)], dtype=np.float32),
        layer_names=np.array(layer_names, dtype=object),

        fit_muG=np.array([muG], dtype=np.float32),
        fit_sdG=np.array([sdG], dtype=np.float32),
        fit_muT=np.array([muT], dtype=np.float32),
        fit_sdT=np.array([sdT], dtype=np.float32),

        gain_raw_test=gain_raw_test,
        tilt_raw_test=tilt_raw_test,

        base_logits=base_logits,
        temp_scaled_logits=ts_logits,
        swap_logits=swap_logits,
        amp_logits=amp_logits,
        amp_normmatched_logits=amp_nm_logits,

        base_metrics=json.dumps({k: v for k, v in base_m.items() if not isinstance(v, np.ndarray)}),
        temp_scaled_metrics=json.dumps({k: float(ts_m[k]) for k in ["acc","ece","nll","brier","logit_norm_mean"]}),
        swap_metrics=json.dumps({k: v for k, v in swap_m.items() if not isinstance(v, np.ndarray)}),
        amp_metrics=json.dumps({k: v for k, v in amp_m.items() if not isinstance(v, np.ndarray)}),
        amp_normmatched_metrics=json.dumps({k: v for k, v in amp_nm_m.items() if not isinstance(v, np.ndarray)}),

        tau_star=np.array([tau_star], dtype=np.float32),
        amp_preserve_diag=json.dumps(amp_diag if amp_diag is not None else {}),
        split_info=json.dumps(split_info),
        model_knobs=json.dumps({
            "lambda_lap": float(args.lambda_lap),
            "realization": str(args.realization),
            "gate_mode": str(args.gate_mode),
            "stabilize_cheb": int(args.stabilize_cheb),
            "drop_rate": float(args.drop_rate),
            "channels_last": bool(args.channels_last),
        }),
    )
    print("[ok] wrote", out_npz, flush=True)

    # ---- Sanity suite
    if not args.no_sanity:
        run_sanity_suite(
            out_dir=args.out_dir,
            y_true=y_true,
            labels=labels_test,
            k=args.k,
            n_classes=classes,
            swap_logits=swap_logits,
            amp_logits=amp_logits,
            amp_nm_logits=amp_nm_logits,
            seed=args.seed,
            n_bins=args.n_bins,
            do_show=do_show,
        )

    # ---- Concise main-text numbers
    print("\n=== TINY-IMAGENET ORDER-MIXING KEY NUMBERS (TEST=official val/; FIT=train split) ===", flush=True)
    print(f"swap_beta={float(args.swap_beta):.3f}", flush=True)
    print(f"base: acc={base_m['acc']:.4f} ECE={base_m['ece']:.4f} NLL={base_m['nll']:.4f} Brier={base_m['brier']:.4f} mean||z||={base_m['logit_norm_mean']:.4f}", flush=True)
    print(f"temp scaling (global; fit={split_info['fit_split']}): tau*={tau_star:.6f} | acc={ts_m['acc']:.4f} ECE={ts_m['ece']:.4f} NLL={ts_m['nll']:.4f} Brier={ts_m['brier']:.4f} mean||z||={ts_m['logit_norm_mean']:.4f}", flush=True)
    print(f"swap: acc={swap_m['acc']:.4f} ECE={swap_m['ece']:.4f} NLL={swap_m['nll']:.4f} Brier={swap_m['brier']:.4f} mean||z||={swap_m['logit_norm_mean']:.4f}", flush=True)
    print(f"amp : acc={amp_m['acc']:.4f} ECE={amp_m['ece']:.4f} NLL={amp_m['nll']:.4f} Brier={amp_m['brier']:.4f} mean||z||={amp_m['logit_norm_mean']:.4f}", flush=True)
    print(f"nm  : acc={amp_nm_m['acc']:.4f} ECE={amp_nm_m['ece']:.4f} NLL={amp_nm_m['nll']:.4f} Brier={amp_nm_m['brier']:.4f} mean||z||={amp_nm_m['logit_norm_mean']:.4f}", flush=True)
    print("====================================================================================\n", flush=True)


if __name__ == "__main__":
    main()
