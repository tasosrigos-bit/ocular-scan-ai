"""Evaluation metrics for the ocular project.

A model is scored by running it over a loader and comparing predictions with
labels. The reported metrics are overall accuracy, macro-averaged F1, and the
recall of every class. Macro-averaged F1 gives each class equal weight, so poor
performance on a rare class such as DRUSEN is not hidden by the majority classes,
and the per-class recall makes that performance explicit.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from ocular import config
from ocular.classifier.train import get_device


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the true and predicted labels for a loader.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model.
    loader : torch.utils.data.DataLoader
        Loader over the evaluation set.
    device : torch.device, optional
        Device to run on. Defaults to :func:`ocular.train.get_device`.

    Returns
    -------
    tuple of numpy.ndarray
        The true labels and the predicted labels, as integer arrays.
    """
    device = device or get_device()
    model = model.to(device)
    model.eval()
    trues, preds = [], []
    for xb, yb in loader:
        pred = model(xb.to(device)).argmax(1).cpu()
        trues.append(yb)
        preds.append(pred)
    return torch.cat(trues).numpy(), torch.cat(preds).numpy()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy, macro-F1, and per-class recall.

    Parameters
    ----------
    y_true : numpy.ndarray
        True integer labels.
    y_pred : numpy.ndarray
        Predicted integer labels.

    Returns
    -------
    dict of str to float
        Keys are ``accuracy``, ``macro_f1``, and ``recall_<CLASS>`` for every
        class in ``config.CLASSES``. Classes absent from ``y_true`` report a
        recall of zero rather than raising.
    """
    labels = list(range(len(config.CLASSES)))
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for cls, r in zip(config.CLASSES, recalls):
        out[f"recall_{cls}"] = float(r)
    return out


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device | None = None
) -> dict[str, float]:
    """Run a model over a loader and return its metrics.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model.
    loader : torch.utils.data.DataLoader
        Loader over the evaluation set.
    device : torch.device, optional
        Device to run on.

    Returns
    -------
    dict of str to float
        The metrics returned by :func:`metrics`.
    """
    y_true, y_pred = predict(model, loader, device)
    return metrics(y_true, y_pred)
