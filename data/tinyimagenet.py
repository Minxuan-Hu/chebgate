import os
import time
import shutil
import zipfile
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit


TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINYIMAGENET_DIRNAME = "tiny-imagenet-200"


# ----------------------------
# Simple filesystem locking
# ----------------------------

def _acquire_lock(lock_path: str, timeout_s: int = 3600, poll_s: float = 1.0) -> bool:
    """
    Acquire an exclusive lock by atomically creating a lock file.
    Returns True if acquired, False if timed out.
    """
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            return True
        except FileExistsError:
            if (time.time() - t0) > timeout_s:
                return False
            time.sleep(poll_s)


def _release_lock(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def _wait_for_marker(marker_path: str, timeout_s: int = 3600, poll_s: float = 1.0) -> bool:
    t0 = time.time()
    while True:
        if os.path.exists(marker_path):
            return True
        if (time.time() - t0) > timeout_s:
            return False
        time.sleep(poll_s)


def _atomic_write_marker(marker_path: str, text: str = "ok\n") -> None:
    tmp = marker_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, marker_path)


# ----------------------------
# Paths / layout
# ----------------------------

@dataclass(frozen=True)
class TinyImageNetPaths:
    data_root: str
    tiny_root: str
    train_root: str
    val_root: str
    val_images_root: str
    val_ann_path: str
    marker_download: str
    marker_extract: str
    marker_repack: str
    lock_download: str
    lock_extract: str
    lock_repack: str


def get_tinyimagenet_paths(data_root: str) -> TinyImageNetPaths:
    data_root = os.path.abspath(data_root)
    tiny_root = os.path.join(data_root, TINYIMAGENET_DIRNAME)
    train_root = os.path.join(tiny_root, "train")
    val_root = os.path.join(tiny_root, "val")
    val_images_root = os.path.join(val_root, "images")
    val_ann_path = os.path.join(val_root, "val_annotations.txt")

    # marker files (idempotence + mp safety)
    marker_download = os.path.join(tiny_root, ".chebgate_download_ok")
    marker_extract = os.path.join(tiny_root, ".chebgate_extract_ok")
    marker_repack = os.path.join(val_root, ".chebgate_val_repacked_ok")

    # locks in data_root so they exist even if tiny_root doesn't yet
    lock_download = os.path.join(data_root, ".lock_chebgate_tiny_download")
    lock_extract = os.path.join(data_root, ".lock_chebgate_tiny_extract")
    lock_repack = os.path.join(data_root, ".lock_chebgate_tiny_repack")

    return TinyImageNetPaths(
        data_root=data_root,
        tiny_root=tiny_root,
        train_root=train_root,
        val_root=val_root,
        val_images_root=val_images_root,
        val_ann_path=val_ann_path,
        marker_download=marker_download,
        marker_extract=marker_extract,
        marker_repack=marker_repack,
        lock_download=lock_download,
        lock_extract=lock_extract,
        lock_repack=lock_repack,
    )


# ----------------------------
# Download / extract / repack
# ----------------------------

def ensure_tinyimagenet_downloaded(
    data_root: str,
    url: str = TINYIMAGENET_URL,
    timeout_s: int = 3600,
) -> str:
    """
    Ensures tiny-imagenet-200.zip is downloaded into data_root.
    Returns zip_path.
    Multi-process safe via lock + marker.
    """
    paths = get_tinyimagenet_paths(data_root)
    os.makedirs(paths.data_root, exist_ok=True)

    zip_path = os.path.join(paths.data_root, "tiny-imagenet-200.zip")
    # If already extracted and marked, we don't need the zip.
    if os.path.exists(paths.marker_download):
        return zip_path

    # Acquire lock to ensure only one process downloads.
    got = _acquire_lock(paths.lock_download, timeout_s=timeout_s)
    if not got:
        # Someone else should be downloading; wait for marker.
        ok = _wait_for_marker(paths.marker_download, timeout_s=timeout_s)
        if not ok:
            raise RuntimeError("Timed out waiting for TinyImageNet download marker.")
        return zip_path

    try:
        # Re-check after lock acquisition
        if os.path.exists(paths.marker_download):
            return zip_path

        # Download to temp then rename (avoid partial zip being treated as complete)
        tmp_zip = zip_path + ".tmp"
        if os.path.exists(tmp_zip):
            try:
                os.remove(tmp_zip)
            except OSError:
                pass

        urllib.request.urlretrieve(url, tmp_zip)
        os.replace(tmp_zip, zip_path)

        # Marker goes in tiny_root only after extract succeeds; but still mark download here
        # so other processes won't re-download if extract is happening next.
        os.makedirs(paths.tiny_root, exist_ok=True)
        _atomic_write_marker(paths.marker_download)
        return zip_path
    finally:
        _release_lock(paths.lock_download)


def ensure_tinyimagenet_extracted(
    data_root: str,
    cleanup_zip: bool = True,
    timeout_s: int = 3600,
) -> str:
    """
    Ensures tiny-imagenet-200/ exists under data_root and looks valid.
    Returns tiny_root.
    Multi-process safe via lock + marker.
    """
    paths = get_tinyimagenet_paths(data_root)
    os.makedirs(paths.data_root, exist_ok=True)

    # If already extracted & marked, return.
    if os.path.exists(paths.marker_extract) and os.path.isdir(paths.train_root) and os.path.isdir(paths.val_root):
        return paths.tiny_root

    # Ensure zip exists (download if needed)
    zip_path = ensure_tinyimagenet_downloaded(data_root, timeout_s=timeout_s)

    got = _acquire_lock(paths.lock_extract, timeout_s=timeout_s)
    if not got:
        ok = _wait_for_marker(paths.marker_extract, timeout_s=timeout_s)
        if not ok:
            raise RuntimeError("Timed out waiting for TinyImageNet extract marker.")
        return paths.tiny_root

    try:
        # Re-check after lock acquisition
        if os.path.exists(paths.marker_extract) and os.path.isdir(paths.train_root) and os.path.isdir(paths.val_root):
            return paths.tiny_root

        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"TinyImageNet zip not found at: {zip_path}")

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(paths.data_root)

        # Validate minimum structure
        if not os.path.isdir(paths.train_root):
            raise RuntimeError(f"Extraction completed but train/ missing: {paths.train_root}")
        if not os.path.isfile(paths.val_ann_path):
            raise RuntimeError(f"Extraction completed but val_annotations.txt missing: {paths.val_ann_path}")

        _atomic_write_marker(paths.marker_extract)

        if cleanup_zip:
            try:
                os.remove(zip_path)
            except OSError:
                pass

        return paths.tiny_root
    finally:
        _release_lock(paths.lock_extract)


def ensure_tinyimagenet_val_repacked(
    data_root: str,
    timeout_s: int = 3600,
) -> str:
    """
    Repack tiny-imagenet-200/val/images into ImageFolder-friendly structure:
        val/images/<wnid>/*.JPEG
    using val/val_annotations.txt.

    Returns val_images_root (the folder you pass to ImageFolder for "test").
    Multi-process safe via lock + marker.
    """
    paths = get_tinyimagenet_paths(data_root)

    # Ensure extracted
    ensure_tinyimagenet_extracted(data_root, timeout_s=timeout_s)

    # If already marked repacked, done.
    if os.path.exists(paths.marker_repack):
        return paths.val_images_root

    got = _acquire_lock(paths.lock_repack, timeout_s=timeout_s)
    if not got:
        ok = _wait_for_marker(paths.marker_repack, timeout_s=timeout_s)
        if not ok:
            raise RuntimeError("Timed out waiting for TinyImageNet val repack marker.")
        return paths.val_images_root

    try:
        if os.path.exists(paths.marker_repack):
            return paths.val_images_root

        if not os.path.isdir(paths.val_images_root):
            raise RuntimeError(f"val/images missing: {paths.val_images_root}")
        if not os.path.isfile(paths.val_ann_path):
            raise RuntimeError(f"val_annotations.txt missing: {paths.val_ann_path}")

        # If val/images already contains class subfolders with images, we can mark and exit.
        # We detect by checking if there exists any subdirectory that contains at least one file.
        has_class_dirs = False
        for name in os.listdir(paths.val_images_root):
            p = os.path.join(paths.val_images_root, name)
            if os.path.isdir(p):
                has_class_dirs = True
                break
        if has_class_dirs:
            _atomic_write_marker(paths.marker_repack)
            return paths.val_images_root

        # Repack: move each image into its <wnid> folder
        with open(paths.val_ann_path, "r") as f:
            for line in f:
                # Format: <img>\t<wnid>\t<x>\t<y>\t<w>\t<h>
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_name, wnid = parts[0], parts[1]
                src = os.path.join(paths.val_images_root, img_name)
                dst_dir = os.path.join(paths.val_images_root, wnid)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, img_name)

                # If already moved, skip; if src exists, move.
                if os.path.exists(dst):
                    continue
                if os.path.exists(src):
                    shutil.move(src, dst)

        _atomic_write_marker(paths.marker_repack)
        return paths.val_images_root
    finally:
        _release_lock(paths.lock_repack)


def ensure_tinyimagenet_ready(
    data_root: str,
    url: str = TINYIMAGENET_URL,
    cleanup_zip: bool = True,
    timeout_s: int = 3600,
) -> TinyImageNetPaths:
    """
    Convenience: download + extract + repack val.
    Returns paths object.
    """
    paths = get_tinyimagenet_paths(data_root)
    ensure_tinyimagenet_downloaded(data_root, url=url, timeout_s=timeout_s)
    ensure_tinyimagenet_extracted(data_root, cleanup_zip=cleanup_zip, timeout_s=timeout_s)
    ensure_tinyimagenet_val_repacked(data_root, timeout_s=timeout_s)
    return paths


# ----------------------------
# Split helpers
# ----------------------------

def get_tinyimagenet_train_labels(data_root: str) -> List[int]:
    """
    Returns class indices for every sample in tiny-imagenet-200/train.
    This is used to build deterministic stratified train/val indices.
    """
    paths = get_tinyimagenet_paths(data_root)
    if not os.path.isdir(paths.train_root):
        ensure_tinyimagenet_extracted(data_root)

    ds = ImageFolder(paths.train_root, transform=None)
    # torchvision ImageFolder typically exposes .targets; fallback to samples
    if hasattr(ds, "targets") and ds.targets is not None:
        return list(ds.targets)
    return [y for _, y in ds.samples]


def make_stratified_split_indices(
    labels: List[int],
    seed: int,
    val_frac: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic stratified split indices for train/val.
    Returns (train_idx, val_idx) as numpy arrays.
    """
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0,1). Got {val_frac}")
    y = np.asarray(labels, dtype=np.int64)
    n = len(y)
    if n == 0:
        raise ValueError("Empty label list.")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=int(seed))
    # X is dummy; stratify uses y.
    tr_idx, val_idx = next(sss.split(np.zeros(n, dtype=np.int8), y))
    return np.asarray(tr_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


# ----------------------------
# Dataset builders
# ----------------------------

def build_tinyimagenet_datasets(
    data_root: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    tf_train,
    tf_eval,
    use_val_as_test: bool = True,
):
    """
    Builds (train_ds, val_ds, test_ds) with transforms:
      - train_ds uses tf_train over train split indices
      - val_ds uses tf_eval over val split indices (from train folder)
      - test_ds uses tf_eval over repacked val/images (val-as-test), if use_val_as_test

    Returns: (train_ds, val_ds, test_ds)
    """
    paths = ensure_tinyimagenet_ready(data_root)

    # Build two ImageFolders over train/ with different transforms,
    # then Subset using the same class ordering.
    train_full = ImageFolder(paths.train_root, transform=tf_train)
    eval_full = ImageFolder(paths.train_root, transform=tf_eval)

    train_ds = Subset(train_full, list(map(int, train_idx)))
    val_ds = Subset(eval_full, list(map(int, val_idx)))

    if use_val_as_test:
        test_root = paths.val_images_root  # repacked: val/images/<wnid>/*.JPEG
        test_ds = ImageFolder(test_root, transform=tf_eval)
    else:
        # If you ever decide to use a different test folder later, you can extend here.
        raise NotImplementedError("use_val_as_test=False is not implemented in this minimal data layer.")

    return train_ds, val_ds, test_ds
