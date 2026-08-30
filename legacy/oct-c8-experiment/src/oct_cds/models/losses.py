from __future__ import annotations

from typing import Any

import numpy as np


def class_weights_from_manifest(manifest_csv: str, num_classes: int):
    import pandas as pd
    import torch

    counts = np.ones(num_classes)
    df = pd.read_csv(manifest_csv)
    vc = df[df["quality_flag"] == "ok"]["label_id"].value_counts()
    for k, v in vc.items():
        counts[int(k)] = v
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def build_loss(cfg: Any, class_weights=None):
    """cfg keys: name (cross_entropy|focal), label_smoothing, class_weighted."""
    import torch
    import torch.nn as nn

    name = cfg.get("name", "cross_entropy")
    smoothing = float(cfg.get("label_smoothing", 0.0))
    weight = class_weights if cfg.get("class_weighted") else None

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)

    if name == "focal":
        gamma = float(cfg.get("gamma", 2.0))

        class FocalLoss(nn.Module):
            def forward(self, logits, target):
                ce = nn.functional.cross_entropy(
                    logits, target, weight=weight, reduction="none",
                    label_smoothing=smoothing,
                )
                pt = torch.exp(-ce)
                return ((1 - pt) ** gamma * ce).mean()

        return FocalLoss()

    raise ValueError(f"unknown loss {name!r}")
