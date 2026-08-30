"""Metrics for internal test and the OPTOPOL external set.

Headline numbers: per-class sensitivity/specificity, macro-F1, AUROC, AUPRC,
quadratic-weighted kappa, plus expected calibration error. Bootstrap CIs via
``bootstrap_ci``.
"""

from __future__ import annotations

import numpy as np


def _one_hot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1
    return out


def sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray, k: int):
    sens, spec = {}, {}
    for c in range(k):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        tn = int(np.sum((y_pred != c) & (y_true != c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        sens[c] = tp / (tp + fn) if (tp + fn) else float("nan")
        spec[c] = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def confusion_matrix_dict(y_true, y_pred, class_names: list[str]) -> dict:
    """Row = true class, column = predicted class. Also row-normalised (recall)."""
    from sklearn.metrics import confusion_matrix

    k = len(class_names)
    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=list(range(k)))
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    return {
        "labels": list(class_names),
        "matrix": cm.tolist(),
        "row_normalized": np.round(cm / row_sums, 4).tolist(),
    }


def classification_report_dict(
    y_true, y_pred, probs=None, class_names: list[str] | None = None
) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        cohen_kappa_score,
        f1_score,
        precision_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    k = len(class_names) if class_names else (
        int(max(y_true.max(), y_pred.max())) + 1 if len(y_true) else 0
    )
    names = class_names or [str(i) for i in range(k)]

    sens, spec = sensitivity_specificity(y_true, y_pred, k)
    f1_per = f1_score(y_true, y_pred, labels=list(range(k)), average=None, zero_division=0)
    prec_per = precision_score(y_true, y_pred, labels=list(range(k)), average=None, zero_division=0)

    report = {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "balanced_accuracy": float(np.nanmean([sens[c] for c in range(k)])),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
        )
        if len(np.unique(y_true)) > 1
        else float("nan"),
        "per_class": {
            names[c]: {
                "sensitivity": sens[c],       # recall / TPR
                "specificity": spec[c],       # TNR
                "precision": float(prec_per[c]),
                "f1": float(f1_per[c]),
                "support": int(np.sum(y_true == c)),
            }
            for c in range(k)
        },
    }

    if probs is not None and len(y_true):
        probs = np.asarray(probs)
        oh = _one_hot(y_true, probs.shape[1])
        present = oh.sum(axis=0) > 0
        try:
            report["auroc_macro"] = float(
                roc_auc_score(oh[:, present], probs[:, present], average="macro")
            )
            report["auprc_macro"] = float(
                average_precision_score(oh[:, present], probs[:, present], average="macro")
            )
        except ValueError:
            report["auroc_macro"] = float("nan")
            report["auprc_macro"] = float("nan")

        for c in range(k):
            entry = report["per_class"][names[c]]
            if oh[:, c].any() and not oh[:, c].all():
                entry["auroc"] = float(roc_auc_score(oh[:, c], probs[:, c]))
                entry["auprc"] = float(average_precision_score(oh[:, c], probs[:, c]))
            else:
                entry["auroc"] = float("nan")
                entry["auprc"] = float("nan")

        report["ece"] = expected_calibration_error(probs, y_true)

    return report


def bootstrap_ci(metric_fn, y_true, y_pred, n: int = 2000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(np.mean(stats)), "ci_low": float(lo), "ci_high": float(hi)}
