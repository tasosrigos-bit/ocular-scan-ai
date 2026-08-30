"""Image quality control — flags rows in the manifest (does not delete files).

Rows with quality_flag != "ok" are kept in the manifest but excluded from
training (see DataModule). STUB: wire in real checks before first training run.
"""

from __future__ import annotations

import numpy as np

QUALITY_FLAGS = (
    "ok", "low_contrast", "blur", "artifact", "wrong_modality", "phi_suspected",
)


def assess_image(arr: np.ndarray) -> str:
    """Return a quality_flag for a single grayscale image array (H, W), 0-255.

    TODO: replace heuristics with validated thresholds / a small QC model.
    """
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)

    contrast = float(arr.std())
    if contrast < 12.0:
        return "low_contrast"

    # variance-of-Laplacian blur proxy without scipy
    gy, gx = np.gradient(arr.astype(np.float32))
    sharpness = float((gx**2 + gy**2).mean())
    if sharpness < 5.0:
        return "blur"

    return "ok"


def check_phi_corners(arr: np.ndarray) -> bool:
    """Heuristic: bright text-like pixels in image corners -> possible burned-in PHI.

    Returns True if the image should be manually reviewed before use.
    """
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    h, w = arr.shape
    ch, cw = max(1, h // 8), max(1, w // 4)
    corners = [
        arr[:ch, :cw], arr[:ch, -cw:], arr[-ch:, :cw], arr[-ch:, -cw:],
    ]
    return any(float((c > 220).mean()) > 0.05 for c in corners)
