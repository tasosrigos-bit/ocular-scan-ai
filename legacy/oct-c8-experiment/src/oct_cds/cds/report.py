"""Assemble a human-readable CDS report from a Recommendation.

Structured dict + plain-text rendering. ``llm_narrator`` (optional) may only
restyle ``narrative`` — never change urgency, class, or refs.
"""

from __future__ import annotations

from typing import Any

from oct_cds.cds.schema import CaseInput, ModelResult, Recommendation


def build_report(
    rec: Recommendation, model_result: ModelResult, case: CaseInput
) -> dict[str, Any]:
    return {
        "case": {
            "image_path": case.image_path,
            "eye": case.eye,
            "acquisition_device": case.acquisition_device,
        },
        "impression": {
            "predicted_class": rec.predicted_class,
            "group": rec.predicted_group,
            "confidence": rec.confidence,
            "margin": rec.margin,
            "abstained": rec.abstained,
            "ood_rejected": rec.ood_rejected,
        },
        "differential": rec.differential,
        "triage": {
            "urgency": rec.urgency.value,
            "recommendation": rec.recommendation_text,
        },
        "guideline_refs": rec.guideline_refs,
        "provenance": {
            "rules_version": rec.rules_version,
            "model_version": rec.model_version,
            "calibrator_version": model_result.calibrator_version,
            "temperature": model_result.temperature,
            "created_at": rec.created_at.isoformat(),
        },
        "disclaimer": rec.disclaimer,
        "narrative": render_text(rec, case),
    }


def render_text(rec: Recommendation, case: CaseInput) -> str:
    lines = [
        "AI CLINICAL DECISION SUPPORT — OCT MACULA",
        f"Image: {case.image_path}  Eye: {case.eye}",
        "",
    ]
    if rec.ood_rejected:
        lines.append("IMPRESSION: input rejected (out-of-distribution / not a macular OCT).")
    elif rec.abstained:
        lines.append("IMPRESSION: insufficient model confidence — no automated class asserted.")
    else:
        lines.append(f"IMPRESSION: {rec.predicted_class} (confidence {rec.confidence:.0%}).")
    ddx = ", ".join(
        "{} {:.0%}".format(d["class"], d["probability"]) for d in rec.differential
    )
    lines += [
        "",
        f"DIFFERENTIAL: {ddx}",
        f"TRIAGE: {rec.urgency.value.upper()} — {rec.recommendation_text}",
    ]
    if rec.guideline_refs:
        lines += ["", "REFERENCES:"] + [f"  - {r}" for r in rec.guideline_refs]
    lines += ["", rec.disclaimer]
    return "\n".join(lines)
