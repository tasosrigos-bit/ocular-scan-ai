"""Aggregate per-image CDS recommendation rows into a summary.

Each row is a dict with at least: ``true``, ``pred``, ``correct``,
``abstained``, ``ood_rejected``, ``deferred_to_specialist``, ``urgency``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def _breakdown(subset: list[dict]) -> dict[str, Any]:
    deferred = [r for r in subset if r["deferred_to_specialist"]]
    confident = [r for r in subset if not r["deferred_to_specialist"]]
    return {
        "n": len(subset),
        "deferred_to_specialist": len(deferred),
        "confident_call": len(confident),
        "confident_call_urgencies": dict(Counter(r["urgency"] for r in confident)),
    }


def summarize_recommendations(rows: list[dict], split: str = "") -> dict[str, Any]:
    n = len(rows)
    mis = [r for r in rows if not r["correct"]]
    cor = [r for r in rows if r["correct"]]
    return {
        "split": split,
        "n_images": n,
        "model_accuracy": round(len(cor) / n, 4) if n else None,
        "urgency_distribution": dict(Counter(r["urgency"] for r in rows)),
        "abstention_rate": round(sum(r["abstained"] for r in rows) / n, 4) if n else None,
        "ood_reject_rate": round(sum(r["ood_rejected"] for r in rows) / n, 4) if n else None,
        # the headline question: on the images the model got WRONG, did CDS
        # defer to a specialist (good) or assert a confident wrong call (bad)?
        "on_misclassified": _breakdown(mis),
        "on_correct": _breakdown(cor),
        "misclassified_by_true_class": {
            k: _breakdown([r for r in mis if r["true"] == k])
            for k in sorted({r["true"] for r in mis})
        },
    }
