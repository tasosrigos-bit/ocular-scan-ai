"""Assemble a human-readable CDS report from a Recommendation.

Structured dict + plain-text rendering. The optional Part 2 ``narrator`` may only
add a grounded ``narrative_rag`` (and its citations) — ``impression`` and
``triage`` always come verbatim from the Recommendation, and the plain
``narrative`` is always Part 1's templated text.
"""

from __future__ import annotations

from typing import Any

from oct_cds.cds.schema import CaseInput, ModelResult, Recommendation


def build_report(
    rec: Recommendation,
    model_result: ModelResult,
    case: CaseInput,
    narrator: Any | None = None,
) -> dict[str, Any]:
    report = {
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

    if narrator is not None:
        report["narrator_meta"] = _attach_rag_narrative(report, narrator, rec, case)

    return report


def _attach_rag_narrative(report, narrator, rec, case) -> dict[str, Any]:
    """Part 2. Never mutates impression/triage/narrative — only adds fields."""
    result = narrator.narrate(rec, case)
    if not result.rag_used:
        return {"rag_used": False, "reason": result.reason}

    gn = result.narrative
    meta = {
        "rag_used": True,
        "verified": gn.verified,
        "fallback_used": gn.fallback_used,
        "flags": gn.flags,
        "model": gn.model,
        "kb_version": gn.kb_version,
        "retrieved_ids": gn.retrieved_ids,
        "raw_text": gn.raw_text,          # the LLM's unedited output (for inspection)
    }
    if gn.verified and not gn.fallback_used:
        report["narrative_rag"] = gn.text
        report["citations"] = gn.citations
    return meta


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
