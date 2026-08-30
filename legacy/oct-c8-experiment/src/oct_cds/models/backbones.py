"""Backbone factory. All models come from ``timm`` so densenet121 / resnet50 /
efficientnet_b3 are drop-in swappable via ``model=<name>`` on the CLI."""

from __future__ import annotations

from typing import Any

SUPPORTED = {"densenet121", "resnet50", "efficientnet_b3"}


def build_backbone(cfg: Any):
    """Create a timm model with a fresh ``num_classes`` head.

    cfg keys: timm_name, pretrained, in_chans, num_classes, drop_rate
    """
    import timm

    name = cfg["timm_name"]
    model = timm.create_model(
        name,
        pretrained=bool(cfg.get("pretrained", True)),
        num_classes=int(cfg["num_classes"]),
        in_chans=int(cfg.get("in_chans", 3)),
        drop_rate=float(cfg.get("drop_rate", 0.0)),
    )
    return model


def split_param_groups(model, base_lr: float, head_lr_mult: float) -> list[dict]:
    """Two param groups: classifier head (fast) and everything else (base)."""
    try:
        head_params = set(map(id, model.get_classifier().parameters()))
    except AttributeError:  # pragma: no cover
        head_params = set()
    head, body = [], []
    for p in model.parameters():
        (head if id(p) in head_params else body).append(p)
    groups = [{"params": body, "lr": base_lr}]
    if head:
        groups.append({"params": head, "lr": base_lr * head_lr_mult})
    return groups


def set_backbone_frozen(model, frozen: bool) -> None:
    head_params = set()
    try:
        head_params = set(map(id, model.get_classifier().parameters()))
    except AttributeError:  # pragma: no cover
        pass
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad = not frozen
