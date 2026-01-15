from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torchvision.transforms as T


def tinyimagenet_default_mean_std() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2302, 0.2265, 0.2262)
    return mean, std


def _maybe_randaugment(n: int, m: int):
    # Torchvision compatibility: RandAugment may not exist in older builds.
    try:
        return T.RandAugment(num_ops=n, magnitude=m)
    except Exception:
        # Fallback: ImageNet AutoAugment if available; else identity.
        try:
            return T.AutoAugment(T.AutoAugmentPolicy.IMAGENET)
        except Exception:
            return T.Lambda(lambda x: x)


def build_tinyimagenet_transforms(
    train: bool,
    image_size: int = 64,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    randaugment_n: int = 3,
    randaugment_m: int = 9,
    crop_padding: int = 8,
    hflip_p: float = 0.5,
    random_erasing_p: float = 0.25,
    random_erasing_scale: Tuple[float, float] = (0.02, 0.20),
):
    """
    Returns torchvision transform pipeline for TinyImageNet.

    - Train: RandAugment -> RandomCrop -> HFlip -> ToTensor -> Normalize -> RandomErasing
    - Eval : ToTensor -> Normalize
    """
    if mean is None or std is None:
        mean0, std0 = tinyimagenet_default_mean_std()
        mean = mean0 if mean is None else mean
        std = std0 if std is None else std

    if train:
        return T.Compose(
            [
                _maybe_randaugment(randaugment_n, randaugment_m),
                T.RandomCrop(image_size, padding=crop_padding, padding_mode="reflect"),
                T.RandomHorizontalFlip(p=hflip_p),
                T.ToTensor(),
                T.Normalize(mean, std),
                T.RandomErasing(p=random_erasing_p, scale=random_erasing_scale),
            ]
        )
    else:
        return T.Compose([T.ToTensor(), T.Normalize(mean, std)])
