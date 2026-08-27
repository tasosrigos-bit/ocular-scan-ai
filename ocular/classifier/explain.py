"""Grad-CAM explainability for the ocular classifier.

This module produces a class activation map for a single B-scan, so that the
region the model relies on can be shown on top of the scan. The question it is
meant to answer is whether the model attends to the retinal pathology or to an
artefact of the acquiring device.

The public entry point is :func:`explain_scan`, which preprocesses a file, runs
Grad-CAM and returns the scan together with its activation map. The individual
pieces are exposed so that a caller can attach the hooks to a different layer or
combine the map with the scan in a different way.

Notes
-----
Grad-CAM weights the activations of one convolutional layer by the gradient of a
class logit with respect to those activations, then keeps the positive part and
normalises it to ``[0, 1]``. For a ConvNeXt-Tiny built by
:func:`ocular.model.build_model`, the last stage is reached at
``model.features[-1]`` and is a sensible layer to attach the hooks to.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from torch import nn

from ocular import config
from ocular.classifier.data import IMAGENET_MEAN, IMAGENET_STD, PreConfig
from ocular.classifier.train import get_device


def target_layer(model: nn.Module) -> nn.Module:
    """Return the convolutional layer Grad-CAM attaches its hooks to.

    The layer should be the last one that still keeps spatial structure, so that
    its activations can be mapped back onto the scan. The location differs by
    backbone family: the ConvNeXt, EfficientNet and custom models built by
    :func:`ocular.model.build_model` expose it as ``model.features[-1]``, while
    ResNet exposes it as ``model.layer4[-1]``. Attention-only backbones (e.g.
    ViT) have no such layer and are rejected with a clear error rather than
    silently reading the wrong module.

    Parameters
    ----------
    model : torch.nn.Module
        A model built by :func:`ocular.model.build_model`.

    Returns
    -------
    torch.nn.Module
        The layer whose activations and gradients Grad-CAM will read.

    Raises
    ------
    ValueError
        If the model exposes neither ``features`` nor ``layer4``.
    """
    if hasattr(model, "features"):
        return model.features[-1]
    if hasattr(model, "layer4"):  # ResNet-style backbones
        return model.layer4[-1]
    raise ValueError(
        f"Grad-CAM target layer is not known for {type(model).__name__}; it "
        "exposes neither `features` nor `layer4`."
    )


def gradcam(
    model: nn.Module,
    image: torch.Tensor,
    target: int | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute a Grad-CAM class activation map for one image.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model, in evaluation mode.
    image : torch.Tensor
        One preprocessed and normalised image of shape ``(3, H, W)`` or
        ``(1, 3, H, W)``, on any device.
    target : int, optional
        Index of the class whose map is produced. When ``None`` the predicted
        class is used.
    device : torch.device, optional
        Device to run on. Defaults to :func:`ocular.train.get_device`.

    Returns
    -------
    numpy.ndarray
        The activation map of shape ``(H, W)`` with values in ``[0, 1]``,
        upsampled to the input resolution.

    Notes
    -----
    A sketch of the steps. Register a forward hook on :func:`target_layer` to
    capture its activations and a backward hook to capture their gradients. Run a
    forward pass, take the logit of ``target``, and back-propagate it. Weight each
    activation channel by the mean of its gradient, sum over channels, keep the
    positive part with a ReLU, upsample to ``(H, W)`` and rescale to ``[0, 1]``.
    """
    device = device or get_device()
    model = model.to(device)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    layer = target_layer(model)
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        activations["value"] = out

    def backward_hook(module: nn.Module, grad_in: tuple, grad_out: tuple) -> None:
        gradients["value"] = grad_out[0]

    fh = layer.register_forward_hook(forward_hook)
    bh = layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image)
        if target is None:
            target = int(logits.argmax(1).item())
        logits[0, target].backward()

        act = activations["value"][0]
        grad = gradients["value"][0]
        weights = grad.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * act).sum(0))
    finally:
        fh.remove()
        bh.remove()

    cam = cam.detach().cpu().numpy()
    cam -= cam.min()
    peak = cam.max()
    if peak > 0:
        cam /= peak

    h, w = image.shape[-2], image.shape[-1]
    resized = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h))
    return np.asarray(resized, dtype=np.float32) / 255.0


def overlay(scan: np.ndarray, cam: np.ndarray, alpha: float = 0.4, cmap: str = "jet") -> np.ndarray:
    """Blend a class activation map over a grey-scale B-scan.

    Parameters
    ----------
    scan : numpy.ndarray
        Grey-scale scan of shape ``(H, W)`` with values in ``[0, 1]``.
    cam : numpy.ndarray
        Activation map with values in ``[0, 1]``. It is resized to the scan shape
        if the two differ.
    alpha : float, optional
        Weight of the heat map in the blend, between 0 and 1.
    cmap : str, optional
        Name of the matplotlib colour map used for the heat map.

    Returns
    -------
    numpy.ndarray
        Colour image of shape ``(H, W, 3)`` with values in ``[0, 1]``.
    """
    if cam.shape != scan.shape:
        cam = np.asarray(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((scan.shape[1], scan.shape[0])),
            dtype=np.float32,
        ) / 255.0
    heat = colormaps[cmap](cam)[..., :3]
    base = np.stack([scan] * 3, axis=-1)
    return (1.0 - alpha) * base + alpha * heat


def explain_scan(
    model: nn.Module,
    path: str | Path,
    cfg: PreConfig,
    target: int | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess one file, compute its Grad-CAM map, and return both.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model.
    path : str or pathlib.Path
        Path to the B-scan image file.
    cfg : PreConfig
        The preprocessing configuration, matched to the trained model.
    target : int, optional
        Class index passed to :func:`gradcam`.
    device : torch.device, optional
        Device to run on. Defaults to :func:`ocular.train.get_device`.

    Returns
    -------
    tuple of numpy.ndarray
        The preprocessed grey-scale scan of shape ``(out_h, out_w)`` in
        ``[0, 1]``, and its activation map of the same shape in ``[0, 1]``.

    Notes
    -----
    A sketch of the steps. Preprocess the file with ``cfg.apply`` to a grey-scale
    scan in ``[0, 1]``. Turn it into the three-channel, ImageNet-normalised tensor
    the model expects, matching :class:`ocular.data.OCTDataset`. Call
    :func:`gradcam` on that tensor. Return the plain scan for display and the map.
    """
    device = device or get_device()
    scan = cfg.apply(path)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = torch.from_numpy(scan).float().unsqueeze(0).repeat(3, 1, 1)
    tensor = (tensor - mean) / std

    cam = gradcam(model, tensor, target=target, device=device)
    return scan, cam
