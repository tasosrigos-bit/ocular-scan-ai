"""Deterministic preprocessing + train-only augmentation.

The same deterministic path (ROI crop -> resize -> to-RGB -> normalize) is used
for every split including the OPTOPOL external set. Augmentation is applied only
when ``train=True``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def retina_roi(img: Image.Image) -> Image.Image:
    """Crop to the bright retinal band (row-energy threshold). Falls back to no-op."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    row_energy = g.mean(axis=1)
    thr = row_energy.mean() + 0.15 * row_energy.std()
    rows = np.where(row_energy > thr)[0]
    if rows.size < 8:
        return img
    pad = 12
    top = max(0, int(rows[0]) - pad)
    bot = min(g.shape[0], int(rows[-1]) + pad)
    return img.crop((0, top, img.width, bot))


def build_transforms(cfg: Any, train: bool):
    """Return a torchvision transform pipeline built from the preprocess config."""
    import torch
    from torchvision import transforms as T

    size = int(cfg["image_size"])
    steps: list[Any] = []

    if cfg.get("roi_crop", True):
        steps.append(T.Lambda(retina_roi))

    if train:
        aug = cfg.get("train_augmentation", {}) or {}
        steps.append(T.Resize((int(size * 1.14), int(size * 1.14))))
        steps.append(T.RandomCrop(size))
        if aug.get("horizontal_flip"):
            steps.append(T.RandomHorizontalFlip(float(aug["horizontal_flip"])))
        if aug.get("rotate_deg"):
            steps.append(T.RandomRotation(float(aug["rotate_deg"])))
        if aug.get("brightness_contrast"):
            bc = float(aug["brightness_contrast"])
            steps.append(T.ColorJitter(brightness=bc, contrast=bc))
    else:
        steps.append(T.Resize((size, size)))

    if cfg.get("to_rgb", True):
        steps.append(T.Grayscale(num_output_channels=3))

    steps.append(T.ToTensor())

    if train and (cfg.get("train_augmentation") or {}).get("gaussian_noise_std"):
        std = float(cfg["train_augmentation"]["gaussian_noise_std"])
        steps.append(T.Lambda(lambda t: (t + std * torch.randn_like(t)).clamp(0, 1)))

    steps.append(T.Normalize(mean=list(cfg["normalize_mean"]), std=list(cfg["normalize_std"])))
    return T.Compose(steps)
