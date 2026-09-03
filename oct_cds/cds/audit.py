"""Append-only audit log. Every recommendation is written here with its inputs
and the model / rules versions that produced it."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from oct_cds.cds.schema import CaseInput, ModelResult, Recommendation

_DEFAULT_LOG = Path("outputs/cds_audit.jsonl")


def log_recommendation(
    case: CaseInput,
    model_result: ModelResult,
    rec: Recommendation,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "image_path": case.image_path,
        "eye": case.eye,
        "probs": model_result.probs,
        "ood_score": model_result.ood_score,
        "model_version": model_result.model_version,
        "calibrator_version": model_result.calibrator_version,
        "rules_version": rec.rules_version,
        "predicted_class": rec.predicted_class,
        "urgency": rec.urgency.value,
        "abstained": rec.abstained,
        "ood_rejected": rec.ood_rejected,
    }
    dest = Path(log_path) if log_path else _DEFAULT_LOG
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry
