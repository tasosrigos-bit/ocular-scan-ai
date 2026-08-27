"""Model definitions for the ocular project.

The project compares a small number of backbones under one interface. Every
architecture is built with a fresh classification head of the project's four
classes by :func:`build_model`, which accepts a short architecture name.

The pretrained backbones are drawn from ``torchvision`` and cover the families
compared in notebook 03. :class:`CustomCNN` is a small network trained
from scratch, and serves as the no-pretraining anchor of the pretraining
comparison rather than as a competitor for best accuracy.
"""
from __future__ import annotations

import torch
import torchvision.models as tvm
from torch import nn

from ocular import config

#: Architecture names accepted by :func:`build_model`.
ARCHS = (
    "resnet18",
    "resnet50",
    "resnet101",
    "efficientnet_b0",
    "efficientnet_b3",
    "convnext_tiny",
    "vit_b_16",
    "swin_t",
    "densenet121",
    "custom",
)


class CustomCNN(nn.Module):
    """A small convolutional network trained from scratch.

    The network is four convolutional blocks followed by global average pooling
    and a linear head, giving roughly one to three million parameters. It is the
    no-pretraining anchor of the pretraining comparison.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes.
    in_channels : int, optional
        Number of input channels. The data pipeline serves three channels, so
        the default matches the pretrained backbones.
    """

    def __init__(self, num_classes: int = len(config.CLASSES), in_channels: int = 3) -> None:
        super().__init__()

        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def build_model(
    arch: str = "resnet50",
    num_classes: int = len(config.CLASSES),
    pretrained: bool = True,
) -> nn.Module:
    """Build a backbone with a fresh classification head.

    Parameters
    ----------
    arch : str, optional
        Architecture name, one of :data:`ARCHS`.
    num_classes : int, optional
        Number of output classes.
    pretrained : bool, optional
        Whether to initialise the backbone from its default pretrained weights.
        Ignored for the custom network, which is always trained from scratch.

    Returns
    -------
    torch.nn.Module
        The model with a classification head of ``num_classes`` outputs.

    Raises
    ------
    ValueError
        If ``arch`` is not in :data:`ARCHS`.
    """
    if arch == "custom":
        return CustomCNN(num_classes=num_classes)
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}, expected one of {ARCHS}")

    weights = "DEFAULT" if pretrained else None
    model = getattr(tvm, arch)(weights=weights)

    if arch.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch.startswith("efficientnet") or arch == "convnext_tiny":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif arch == "swin_t":
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif arch == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model.

    Returns
    -------
    int
        The count of parameters with ``requires_grad`` set.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
