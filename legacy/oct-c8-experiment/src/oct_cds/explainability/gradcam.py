"""Grad-CAM for the OCT classifier.

Needs the ``explain`` extra:  pip install -e '.[explain]'
"""

from __future__ import annotations

import numpy as np


def _unwrap(model):
    """OCTClassifier stores the real backbone as ``.net`` — Grad-CAM wants that."""
    return getattr(model, "net", model)


def default_target_layer(model):
    """Best-effort last spatial layer for the supported backbones."""
    net = _unwrap(model)
    name = type(net).__name__.lower()
    if "densenet" in name:
        return net.features.norm5           # timm densenet: final BN before pooling
    if "resnet" in name:
        return net.layer4[-1]
    if "efficientnet" in name:
        return net.conv_head
    # fallback: last module whose weight is a 4-D conv kernel
    last = None
    for m in net.modules():
        if hasattr(m, "weight") and getattr(m.weight, "ndim", 0) == 4:
            last = m
    return last


class GradCAMRunner:
    """Reusable Grad-CAM over one target layer. Create once, call ``cam`` per image."""

    def __init__(self, model, target_layer=None):
        try:
            from pytorch_grad_cam import GradCAM
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Grad-CAM needs: pip install -e '.[explain]'") from exc

        self._net = _unwrap(model)
        # Grad-CAM backprops to the target-layer activations; make sure the graph
        # is not dead because the backbone was frozen during training.
        for p in self._net.parameters():
            p.requires_grad_(True)
        self._net.eval()
        layer = target_layer or default_target_layer(model)
        self._cam = GradCAM(model=self._net, target_layers=[layer])

    def cam(self, image_tensor, target_class: int | None = None) -> np.ndarray:
        """image_tensor: (3,H,W) or (1,3,H,W). Returns an HxW map in [0, 1]."""
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        batch = image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor
        targets = (
            [ClassifierOutputTarget(int(target_class))] if target_class is not None else None
        )
        grayscale = self._cam(input_tensor=batch, targets=targets)[0]
        return np.asarray(grayscale, dtype=np.float32)

    def close(self):
        for attr in ("activations_and_grads",):
            try:
                getattr(self._cam, attr).release()
            except Exception:  # noqa: BLE001
                pass


def gradcam_heatmap(model, image_tensor, target_class: int | None = None) -> np.ndarray:
    """One-shot convenience wrapper (creates and tears down a runner)."""
    runner = GradCAMRunner(model)
    try:
        return runner.cam(image_tensor, target_class)
    finally:
        runner.close()
