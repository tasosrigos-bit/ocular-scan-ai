"""Render a Grad-CAM heatmap on top of the (model's view of the) scan.

The background is the *denormalised input tensor* — i.e. exactly what the network
saw after ROI crop + resize + normalise — so the heatmap lines up pixel-for-pixel
with what produced the prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def denormalize(t, mean, std) -> np.ndarray:
    """(3,H,W) normalised tensor -> HxWx3 float image in [0, 1]."""
    import torch

    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    x = (t.detach().cpu() * s + m).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def colorize(cam: np.ndarray, cmap: str = "jet") -> np.ndarray:
    """HxW map in [0,1] -> HxWx3 RGB float in [0,1] using a matplotlib colormap."""
    from matplotlib import colormaps

    cam = np.clip(cam.astype(np.float32), 0.0, 1.0)
    return colormaps[cmap](cam)[..., :3]


def save_overlay(
    rgb01: np.ndarray,
    cam: np.ndarray,
    dest: str | Path,
    alpha: float = 0.45,
    cmap: str = "jet",
    side_by_side: bool = True,
) -> Path:
    """Blend heatmap onto the image and save a PNG. When ``side_by_side`` the
    output is [ original | overlay ] so you can see the raw scan too."""
    from PIL import Image

    heat = colorize(cam, cmap)
    blended = (1.0 - alpha) * rgb01 + alpha * heat
    blended = np.clip(blended, 0.0, 1.0)

    if side_by_side:
        gap = np.ones((rgb01.shape[0], 4, 3), dtype=np.float32)
        canvas = np.concatenate([rgb01, gap, blended], axis=1)
    else:
        canvas = blended

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((canvas * 255).astype(np.uint8)).save(dest)
    return dest
