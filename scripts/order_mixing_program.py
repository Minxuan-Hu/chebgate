# Fig1: class×cluster heatmap P(cluster|class)
# Fig2: reliability diagram base vs structured swap
# Fig3: temperature attribution
#     (a) logit-norm hist
#     (b) reliability (3 curves)
#     (c) compact metrics table

import os
import re
import json
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.transforms import Bbox

import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from chebgate.model import ChebResNet, ChebConv2d

# -------------------------
# Determinism / strictness
# -------------------------
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


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
COL_BASE = "C1"   # orange-ish
COL_SWAP = "C2"   # green-ish
COL_AUX  = "C0"   # blue-ish (diag)

# -------------------------
# Checkpoint helpers
# -------------------------
def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def extract_state_dict(obj):
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
    classes = int(sd["fc.bias"].numel())
    w1 = int(sd["stem.0.weight"].shape[0])
    w2 = int(sd["l2.0.c1.combine.weight"].shape[0])
    w3 = int(sd["fc.weight"].shape[1])

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

    return classes, (K1, K2, K3), (d1, d2, d3), (w1, w2, w3)

def expand_stage3_layers(spec: str, depth_stage3: int):
    """
    spec:
      - 'ALL' / '*' / 'AUTO' => all l3.{b}.c1 and l3.{b}.c2 for b=0..depth_stage3-1
      - otherwise => parse comma-separated explicit list
    """
    s = (spec or "").strip().lower()
    if s in ("all", "*", "auto", "stage3_all", "all_stage3"):
        out = []
        for b in range(int(depth_stage3)):
            out.append(f"l3.{b}.c1")
            out.append(f"l3.{b}.c2")
        return out
    return [t.strip() for t in spec.split(",") if t.strip()]


# -------------------------
# Dataset with global index + deterministic loading
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

def build_test_loader(data_root="./data", batch_size=128, num_workers=4, seed=0):
    tfms = [
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
    base = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=T.Compose(tfms)
    )
    ds = IndexedDataset(base)

    g = torch.Generator()
    g.manual_seed(int(seed))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
        persistent_workers=(num_workers > 0),
    )
    return loader


# -------------------------
# Small helpers
# -------------------------
def sha16(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256(arr.view(np.uint8)).hexdigest()
    return h[:16]

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

    out = {
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
    }
    return out


# -------------------------
# Forward passes
# -------------------------
@torch.no_grad()
def infer_logits(model, loader, device="cuda", n_classes=10):
    model.eval()
    N = len(loader.dataset)
    logits = np.zeros((N, n_classes), dtype=np.float32)
    for xb, _, idxb in loader:
        xb = xb.to(device, non_blocking=True)
        z = model(xb).detach().float().cpu().numpy()
        logits[idxb.numpy().astype(np.int64)] = z
    return logits

@torch.no_grad()
def collect_g_per_layer(model, loader, layer_names, device="cuda"):
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

    for xb, _, idxb in loader:
        batch_idx_holder["idx"] = idxb.numpy().astype(np.int64)
        xb = xb.to(device, non_blocking=True)
        _ = model(xb)
        batch_idx_holder["idx"] = None

    for h in hooks:
        h.remove()

    return g_store, alpha_store


# -------------------------
# Gain/Tilt computation
# -------------------------
def compute_gain_tilt_from_g(g_store: dict, alpha_store: dict, layer_names):
    """
    Raw per-sample:
      gain_raw(x) = mean_over_layers sum_k |alpha_k * g_k(x)|
      tilt_raw(x) = mean_over_layers slope of abs(alpha*g) vs order index k
    Then z-score each to form (gain, tilt) for clustering.
    """
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


# -------------------------
# Swap mapping (structured)
# -------------------------
def make_swap_to_pair_extremes_by_mean_gain(gain: np.ndarray, labels: np.ndarray, k: int):
    mean_gain = np.zeros((k,), dtype=np.float64)
    for c in range(k):
        m = (labels == c)
        mean_gain[c] = float(gain[m].mean()) if np.any(m) else 0.0
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
    labels,                    # [N]
    swap_to,                   # [k]
    mean_g_by_layer,           # dict ln -> torch [k,K]
    abs_alpha_by_layer=None,   # dict ln -> torch [1,K]
    amp_base_by_layer=None,    # dict ln -> np [N]
    device="cuda",
    n_classes=10,
    mode="plain",              # "plain" or "amp_preserve"
    eps=1e-12,
):
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

    for xb, _, idxb in loader:
        idx = idxb.numpy().astype(np.int64)
        xb = xb.to(device, non_blocking=True)

        src_c = labels[idx]
        tgt_c = swap_to[src_c]
        tgt_t = torch.from_numpy(tgt_c).to(device=device, dtype=torch.long)

        B = xb.shape[0]
        for ln in layer_names:
            g_proto = mean_g_by_layer[ln].index_select(0, tgt_t)  # (B,K)

            if mode == "plain":
                g_use = g_proto

            elif mode == "amp_preserve":
                assert abs_alpha_by_layer is not None and amp_base_by_layer is not None
                abs_alpha = abs_alpha_by_layer[ln]  # (1,K)
                A_base = torch.from_numpy(amp_base_by_layer[ln][idx]).to(device=device, dtype=torch.float32).view(B, 1)
                A_proto = (abs_alpha * g_proto).sum(dim=1, keepdim=True)
                scale = A_base / (A_proto + eps)
                g_scaled = torch.clamp(g_proto * scale, 0.0, 1.0)

                A_scaled = (abs_alpha * g_scaled).sum(dim=1, keepdim=True)
                ratio = (A_scaled / (A_base + eps)).detach().cpu().numpy().reshape(-1).astype(np.float32)
                amp_ratio_collect[ln].append(ratio)

                g_use = g_scaled
            else:
                raise ValueError(f"Unknown mode={mode}")

            override_holder[ln] = g_use

        z = model(xb).detach().float().cpu().numpy()
        logits_out[idx] = z

    for h in hooks:
        h.remove()

    amp_diag = None
    if mode == "amp_preserve":
        amp_diag = {}
        for ln in layer_names:
            r = np.concatenate(amp_ratio_collect[ln], axis=0) if amp_ratio_collect[ln] else None
            if r is None:
                amp_diag[ln] = None
            else:
                amp_diag[ln] = {
                    "mean": float(r.mean()),
                    "p10": float(np.percentile(r, 10)),
                    "p50": float(np.percentile(r, 50)),
                    "p90": float(np.percentile(r, 90)),
                    "min": float(r.min()),
                    "max": float(r.max()),
                }

    return logits_out, amp_diag


# -------------------------
# Logit-norm matching (temperature attribution)
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

def plot_class_x_cluster_heatmap(y_true, labels, k, n_classes, out_path):
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

    fig, ax = plt.subplots(figsize=(3.5, 2.0), dpi=600)
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="magma")

    cifar10_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    if n_classes == 10:
        ax.set_yticks(list(range(10)))
        ax.set_yticklabels(cifar10_names)
    ax.set_ylabel("True class $y$ (CIFAR-10)")

    ax.set_xlabel(r"Order-mixing program $c$ (k-means id)")
    ax.set_xticks(np.arange(k))
    ax.set_frame_on(False)

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
    plt.grid(False)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_fig2_reliability_base_vs_swap(base_metrics, swap_metrics, out_path, n_bins=15):
    fig, ax = plt.subplots(figsize=(3.5, 1.8), dpi=600)
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
    plt.show()
    plt.close(fig)

def plot_fig3_temp_attribution(base_m, amp_m, amp_nm_m, out_prefix, n_bins=15):
    L_BASE = "base"
    L_AMP  = "amp-preserve"
    L_NM   = "logit-norm matched"

    # (a) histogram
    out_a = out_prefix + "_a_hist.png"
    fig, ax = plt.subplots(figsize=(2.36, 1.8), dpi=600)
    ln_base = base_m["logit_norm"]
    ln_amp  = amp_m["logit_norm"]
    ln_nm   = amp_nm_m["logit_norm"]

    ax.hist(ln_amp, bins=60, alpha=0.7, label=L_AMP, color=COL_SWAP)
    ax.hist(ln_nm,  bins=60, alpha=0.2, label=L_NM, color=COL_SWAP)
    ax.hist(ln_base, bins=60, histtype="step", linewidth=1.2, label=L_BASE, color=COL_BASE)

    ax.set_xlabel(r"$\|\mathbf{z}\|_2$")
    ax.set_ylabel("Count")
    ax.legend(loc="best", frameon=True, fontsize=5)

    fig.tight_layout()
    fig.savefig(out_a, bbox_inches="tight")
    fig.savefig(out_a.replace(".png", ".pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # (b) reliability 3 curves
    out_b = out_prefix + "_b_reliability.png"
    fig, ax = plt.subplots(figsize=(2.86, 1.8), dpi=600)

    ax.plot([0, 1], [0, 1], "--", linewidth=1.0, alpha=0.8, color=COL_AUX)
    plot_reliability_curve(ax, base_m["maxprob"], (base_m["pred"] == base_m["_y_true"]),
                           label=f"{L_BASE} (ECE={base_m['ece']:.3f})", n_bins=n_bins,
                           color=COL_BASE, marker="o", linestyle="-")
    plot_reliability_curve(ax, amp_m["maxprob"], (amp_m["pred"] == amp_m["_y_true"]),
                           label=f"{L_AMP} (ECE={amp_m['ece']:.3f})", n_bins=n_bins,
                           color=COL_SWAP, marker="o", linestyle="-")
    plot_reliability_curve(ax, amp_nm_m["maxprob"], (amp_nm_m["pred"] == amp_nm_m["_y_true"]),
                           label=f"{L_NM} (ECE={amp_nm_m['ece']:.3f})", n_bins=n_bins,
                           color=COL_SWAP, marker="s", linestyle="--")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Confidence (bin-avg $\max_j p_\theta(j\mid x)$)")
    ax.set_ylabel(r"Accuracy (bin-avg $\mathbb{1}[\hat y=y]$)")
    ax.legend(loc="best", frameon=True, fontsize=5)

    fig.tight_layout()
    fig.savefig(out_b, bbox_inches="tight")
    fig.savefig(out_b.replace(".png", ".pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # (c) compact metrics table (export names unchanged)
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

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    fig_w, fig_h = 2.15, 1.8
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

    # keep your exact bbox logic (do not change appearance)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    pad_y = 9 / 72
    bbox_y = Bbox.from_extents(tight.x0, tight.y0 - pad_y, tight.x1, tight.y1 + pad_y)

    fig.savefig(out_c, bbox_inches=bbox_y)
    fig.savefig(out_c.replace(".png", ".pdf"), bbox_inches=bbox_y)
    plt.show()
    plt.close(fig)


# -------------------------
# Sanity suite (label leakage defense)
# -------------------------
def run_sanity_suite(out_dir, y_true, labels, k, n_classes,
                     swap_logits, amp_logits, amp_nm_logits,
                     seed=0, n_bins=15):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    y_perm = rng.permutation(y_true)

    figS_path = os.path.join(out_dir, "FigS_labelperm_class_x_cluster.png")
    plot_class_x_cluster_heatmap(y_true=y_perm, labels=labels, k=k, n_classes=n_classes, out_path=figS_path)
    print("[sanity] wrote", figS_path, "(y_true permuted; should NOT be pure/diagonal)")

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
        "note": "labels/logits computed without y_true; y_true only used for evaluation + heatmap rows."
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
# Main pipeline
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./maintext_out")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # analysis config
    ap.add_argument("--k", type=int, default=6, help="num clusters")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_bins", type=int, default=15, help="num bins for ECE / reliability diagrams")
    ap.add_argument("--stage3_layers", type=str, default="ALL",
                    help="Comma-separated layer list (e.g. 'l3.0.c1,l3.0.c2') or 'ALL' to use all stage-3 layers.")
    ap.add_argument("--npz", type=str, default="", help="optional precomputed npz containing gain/tilt (y_true optional)")
    ap.add_argument("--swap_to", type=str, default="", help="optional mapping like '3,2,5,4,0,1'")
    ap.add_argument("--n_perm", type=int, default=20, help="permutation sweep count for Table1")

    ap.add_argument("--skip_perm", action="store_true", default=False, help="skip permutation sweep")

    ap.add_argument("--sanity", action="store_true", default=True, help="run label-leakage sanity suite")

    # model kwargs
    ap.add_argument("--drop_rate", type=float, default=0.1)
    ap.add_argument("--lambda_lap", type=float, default=0.25)
    ap.add_argument("--realization", type=str, default="streamed")
    ap.add_argument("--gate_mode", type=str, default="on")
    ap.add_argument("--stabilize_cheb", type=int, default=0)

    args, _ = ap.parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[env] torch={torch.__version__} torchvision={torchvision.__version__} sklearn={sklearn.__version__}")

    # ---- load model
    ckpt_obj = safe_torch_load(Path(args.ckpt), map_location="cpu")
    sd = strip_module_prefix(extract_state_dict(ckpt_obj))
    classes, K_stages, depth_stages, widths = infer_arch_from_sd(sd)
    print(f"[arch] classes={classes} K={K_stages} depth={depth_stages} widths={widths}")
    
    model = ChebResNet(
        classes=classes,
        K=K_stages,
        depth=depth_stages,
        widths=widths,
        drop_rate=args.drop_rate,
        lap=args.lambda_lap,
        realization=args.realization,
        gate_mode=args.gate_mode,
        stabilize_cheb=bool(args.stabilize_cheb),
    )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}  (gate_sum/sqsum/count are OK unexpected)")

    device = args.device
    model = model.to(device).eval()

    layer_names = expand_stage3_layers(args.stage3_layers, depth_stage3=depth_stages[2])

    # Filter to existing modules
    name_to_mod = dict(model.named_modules())
    not_found = [ln for ln in layer_names if ln not in name_to_mod]
    if not_found:
        print("[warn] some requested layers not found in model.named_modules():", not_found[:20])
        layer_names = [ln for ln in layer_names if ln in name_to_mod]

    print("[cfg] layers:", layer_names)
    print("[cfg] num layers used:", len(layer_names))

    # ---- loader
    loader = build_test_loader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    N = len(loader.dataset)
    idx = np.arange(N, dtype=np.int64)

    # ---- optional NPZ (gain/tilt only; y_true deliberately optional)
    gain, tilt = None, None
    if args.npz.strip():
        D = np.load(args.npz, allow_pickle=True)
        gain = D["gain"].astype(np.float32) if "gain" in D else None
        tilt = D["tilt"].astype(np.float32) if "tilt" in D else None
        print("[npz] loaded (gain,tilt) =", (gain is not None and tilt is not None))

    # ---- compute base logits (no y_true)
    base_logits = infer_logits(model, loader, device=device, n_classes=classes)
    print("[logits] computed base logits; y_true not loaded yet")

    # ---- collect per-sample g(x) and compute labels (still no y_true)
    print("[gate] collecting per-sample g(x) ...")
    g_store, alpha_store = collect_g_per_layer(model, loader, layer_names, device=device)

    if gain is None or tilt is None:
        gain, tilt, gain_raw, tilt_raw = compute_gain_tilt_from_g(g_store, alpha_store, layer_names)
        np.savez(os.path.join(args.out_dir, "computed_gain_tilt.npz"),
                 gain=gain, tilt=tilt, gain_raw=gain_raw, tilt_raw=tilt_raw)
        print("[gain/tilt] computed internally (z-scored). wrote computed_gain_tilt.npz")

    X = np.stack([gain, tilt], axis=1)
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=20)
    labels = km.fit_predict(X).astype(np.int64)

    # ---- choose swap_to (still no y_true)
    if args.swap_to.strip():
        swap_to = np.array([int(x) for x in args.swap_to.split(",")], dtype=np.int64)
        assert swap_to.shape[0] == args.k
        mean_gain = None
        print("[swap_to] manual:", swap_to.tolist())
    else:
        swap_to, mean_gain = make_swap_to_pair_extremes_by_mean_gain(gain, labels, k=args.k)
        print("[swap_to] paired extremes by mean gain:", swap_to.tolist())
        print("[swap_to] mean_gain:", mean_gain.tolist())

    # ---- build mean_g prototypes per cluster per layer, and per-layer A_base(x)
    mean_g_by_layer = {}
    abs_alpha_by_layer = {}
    amp_base_by_layer = {}

    for ln in layer_names:
        cheb = name_to_mod[ln]
        alpha = cheb.order_scales.detach().cpu().numpy().astype(np.float32)  # [K]
        g = g_store[ln]  # [N,K]
        abs_alpha = np.abs(alpha).astype(np.float32)

        amp_base = (abs_alpha.reshape(1, -1) * g).sum(axis=1).astype(np.float32)
        amp_base_by_layer[ln] = amp_base

        s = (alpha.reshape(1, -1) * g).astype(np.float32)
        ms = np.zeros((args.k, s.shape[1]), dtype=np.float32)
        for c in range(args.k):
            m = (labels == c)
            if np.any(m):
                ms[c] = s[m].mean(axis=0)
            else:
                ms[c] = 0.0

        denom = np.where(np.abs(alpha) < 1e-8, 1.0, alpha).astype(np.float32)
        mg = (ms / denom.reshape(1, -1)).astype(np.float32)
        mg = np.clip(mg, 0.0, 1.0)

        mean_g_by_layer[ln] = torch.from_numpy(mg).to(device=device, dtype=torch.float32)
        abs_alpha_by_layer[ln] = torch.from_numpy(abs_alpha.reshape(1, -1)).to(device=device, dtype=torch.float32)

    # ---- NOW load y_true (evaluation only)
    y_true = np.zeros((N,), dtype=np.int64)
    for _, yb, idxb in loader:
        y_true[idxb.numpy().astype(np.int64)] = yb.numpy().astype(np.int64)
    print("[labels] programs fixed; y_true loaded only for evaluation/plots")

    # ---- baseline metrics
    base_m = summarize_metrics(base_logits, y_true, n_bins=args.n_bins)
    base_m["_y_true"] = y_true
    print(f"[base] acc={base_m['acc']:.4f} ent={base_m['entropy_mean']:.4f} nll={base_m['nll']:.4f} "
          f"brier={base_m['brier']:.4f} ece={base_m['ece']:.4f} | ln={base_m['logit_norm_mean']:.4f}")

    # ---- structured swap (plain)
    print("[swap] running structured swap (plain) ...")
    swap_logits, _ = run_gate_override_inference(
        model, loader, layer_names,
        labels=labels, swap_to=swap_to,
        mean_g_by_layer=mean_g_by_layer,
        device=device, n_classes=classes,
        mode="plain",
    )
    swap_m = summarize_metrics(swap_logits, y_true, n_bins=args.n_bins)
    swap_m["_y_true"] = y_true
    print(f"[swap ] acc={swap_m['acc']:.4f} ent={swap_m['entropy_mean']:.4f} nll={swap_m['nll']:.4f} "
          f"brier={swap_m['brier']:.4f} ece={swap_m['ece']:.4f} | ln={swap_m['logit_norm_mean']:.4f}")

    # ---- amp-preserve swap
    print("[amp ] running amp-preserve swap ...")
    amp_logits, amp_diag = run_gate_override_inference(
        model, loader, layer_names,
        labels=labels, swap_to=swap_to,
        mean_g_by_layer=mean_g_by_layer,
        abs_alpha_by_layer=abs_alpha_by_layer,
        amp_base_by_layer=amp_base_by_layer,
        device=device, n_classes=classes,
        mode="amp_preserve",
    )
    amp_m = summarize_metrics(amp_logits, y_true, n_bins=args.n_bins)
    amp_m["_y_true"] = y_true
    print(f"[amp ] acc={amp_m['acc']:.4f} ent={amp_m['entropy_mean']:.4f} nll={amp_m['nll']:.4f} "
          f"brier={amp_m['brier']:.4f} ece={amp_m['ece']:.4f} | ln={amp_m['logit_norm_mean']:.4f}")

    # ---- logit-norm matching
    print("[attr] logit-norm matching: amp -> match base norm ...")
    amp_nm_logits = logit_norm_match(amp_logits, base_logits)
    amp_nm_m = summarize_metrics(amp_nm_logits, y_true, n_bins=args.n_bins)
    amp_nm_m["_y_true"] = y_true
    print(f"[amp norm-matched] acc={amp_nm_m['acc']:.4f} ent={amp_nm_m['entropy_mean']:.4f} nll={amp_nm_m['nll']:.4f} "
          f"brier={amp_nm_m['brier']:.4f} ece={amp_nm_m['ece']:.4f} | ln={amp_nm_m['logit_norm_mean']:.4f}")

    # ---- write figures
    fig1_path = os.path.join(args.out_dir, "Fig1_class_x_cluster.png")
    plot_class_x_cluster_heatmap(y_true=y_true, labels=labels, k=args.k, n_classes=classes, out_path=fig1_path)
    print("[ok] wrote", fig1_path, "(.pdf too)")

    fig2_path = os.path.join(args.out_dir, "Fig2_reliability_base_vs_swap.png")
    plot_fig2_reliability_base_vs_swap(base_m, swap_m, fig2_path, n_bins=args.n_bins)
    print("[ok] wrote", fig2_path, "(.pdf too)")

    fig3_prefix = "Fig3_temperature_attribution"
    fig3_path = os.path.join(args.out_dir, fig3_prefix)
    plot_fig3_temp_attribution(base_m, amp_m, amp_nm_m, fig3_path, n_bins=args.n_bins)
    print("[ok]", fig3_prefix + "_a_hist.{png,pdf}")
    print("[ok]", fig3_prefix + "_b_reliability.{png,pdf}")
    print("[ok]", fig3_prefix + "_c_metrics.{png,pdf}")

    # ---- Table1: permutation robustness (structured swap only)
    perm_csv = os.path.join(args.out_dir, "Table1_perm_sweep_detail.csv")
    perm_sum_csv = os.path.join(args.out_dir, "Table1_perm_sweep_summary.csv")

    if args.skip_perm:
        print("[perm] skipped")
    else:
        print(f"[perm] permutation sweep n_perm={args.n_perm} (structured swap, plain) ...")
        rng = np.random.default_rng(args.seed)
        rows = []
        for t in range(args.n_perm):
            perm = rng.permutation(args.k).astype(np.int64)
            logits_t, _ = run_gate_override_inference(
                model, loader, layer_names,
                labels=labels, swap_to=perm,
                mean_g_by_layer=mean_g_by_layer,
                device=device, n_classes=classes,
                mode="plain",
            )
            m = summarize_metrics(logits_t, y_true, n_bins=args.n_bins)
            rows.append([t, m["acc"], m["ece"], m["nll"], m["brier"], m["entropy_mean"], m["logit_norm_mean"]])
            print(f"  [perm {t:02d}] acc={m['acc']:.4f} ece={m['ece']:.4f} nll={m['nll']:.4f} brier={m['brier']:.4f}")

        rows = np.array(rows, dtype=np.float64)
        header = "perm_id,acc,ece,nll,brier,entropy_mean,logit_norm_mean"
        np.savetxt(perm_csv, rows, delimiter=",", header=header, comments="")
        print("[perm] wrote", perm_csv)

        vals = rows[:, 1:]
        mean = vals.mean(axis=0)
        std = vals.std(axis=0)
        sum_header = "stat,acc,ece,nll,brier,entropy_mean,logit_norm_mean"
        sum_rows = np.vstack([
            np.concatenate([[0], mean]),
            np.concatenate([[1], std]),
        ])
        np.savetxt(perm_sum_csv, sum_rows, delimiter=",", header=sum_header, comments="")
        print("[perm] wrote", perm_sum_csv)

    # ---- save artifacts
    out_npz = os.path.join(args.out_dir, "maintext_artifacts.npz")
    np.savez(
        out_npz,
        idx=idx,
        y_true=y_true,
        gain=gain,
        tilt=tilt,
        labels=labels,
        swap_to=swap_to,
        layer_names=np.array(layer_names, dtype=object),

        base_logits=base_logits,
        swap_logits=swap_logits,
        amp_logits=amp_logits,
        amp_normmatched_logits=amp_nm_logits,

        base_metrics=json.dumps({k: v for k, v in base_m.items() if not isinstance(v, np.ndarray)}),
        swap_metrics=json.dumps({k: v for k, v in swap_m.items() if not isinstance(v, np.ndarray)}),
        amp_metrics=json.dumps({k: v for k, v in amp_m.items() if not isinstance(v, np.ndarray)}),
        amp_normmatched_metrics=json.dumps({k: v for k, v in amp_nm_m.items() if not isinstance(v, np.ndarray)}),

        amp_preserve_diag=json.dumps(amp_diag if amp_diag is not None else {}),
    )
    print("[ok] wrote", out_npz)

    # ---- concise main-text numbers
    print("\n=== MAIN-TEXT KEY NUMBERS ===")
    print(f"base: acc={base_m['acc']:.4f} ECE={base_m['ece']:.4f} NLL={base_m['nll']:.4f} Brier={base_m['brier']:.4f} mean||z||={base_m['logit_norm_mean']:.4f}")
    print(f"swap: acc={swap_m['acc']:.4f} ECE={swap_m['ece']:.4f} NLL={swap_m['nll']:.4f} Brier={swap_m['brier']:.4f} mean||z||={swap_m['logit_norm_mean']:.4f}")
    print(f"amp : acc={amp_m['acc']:.4f} ECE={amp_m['ece']:.4f} NLL={amp_m['nll']:.4f} Brier={amp_m['brier']:.4f} mean||z||={amp_m['logit_norm_mean']:.4f}")
    print(f"amp norm-matched: acc={amp_nm_m['acc']:.4f} ECE={amp_nm_m['ece']:.4f} NLL={amp_nm_m['nll']:.4f} Brier={amp_nm_m['brier']:.4f} mean||z||={amp_nm_m['logit_norm_mean']:.4f}")
    print("=============================\n")

    if args.sanity:
        run_sanity_suite(
            out_dir=args.out_dir,
            y_true=y_true,
            labels=labels,
            k=args.k,
            n_classes=classes,
            swap_logits=swap_logits,
            amp_logits=amp_logits,
            amp_nm_logits=amp_nm_logits,
            seed=args.seed,
            n_bins=args.n_bins,
        )

if __name__ == "__main__":
    main()
