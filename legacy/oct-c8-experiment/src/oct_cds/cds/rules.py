"""Deterministic CDS rule engine.

Input : ModelResult (calibrated probs) + CaseInput
Output: Recommendation

The decision logic lives ENTIRELY here and in ``configs/cds/rules_v1.yaml``.
No model, no LLM, decides urgency. ``cds/llm_narrator.py`` may only rephrase an
already-decided Recommendation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from oct_cds.common.paths import REPO_ROOT
from oct_cds.data.label_map import LabelMap, load_label_map
from oct_cds.cds.guideline_refs import refs_for
from oct_cds.cds.schema import CaseInput, ModelResult, Recommendation, Urgency

_DEFAULT_RULES = REPO_ROOT / "configs" / "cds" / "rules_v1.yaml"


def load_rules(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else _DEFAULT_RULES
    return yaml.safe_load(p.read_text(encoding="utf-8"))


class CDSRuleEngine:
    def __init__(self, rules: dict[str, Any] | None = None, label_map: LabelMap | None = None):
        self.rules = rules or load_rules()
        self.lm = label_map or load_label_map()

    # -- helpers ----------------------------------------------------
    def _urgency_for(self, class_key: str) -> Urgency:
        overrides = self.rules.get("urgency_overrides", {}) or {}
        if class_key in overrides:
            return Urgency(overrides[class_key])
        group = self.lm.group(class_key)
        by_group = self.rules.get("urgency_by_group", {})
        return Urgency(by_group.get(group, "routine"))

    def _escalate_context(self, urgency: Urgency, case: CaseInput, class_key: str) -> Urgency:
        """Small, explicit context bumps. Extend deliberately."""
        order = [Urgency.none, Urgency.routine, Urgency.soon, Urgency.urgent, Urgency.emergent]
        idx = order.index(urgency)
        sudden = any(
            s in {"sudden vision loss", "acute distortion", "new scotoma"}
            for s in (case.symptoms or [])
        )
        if sudden and class_key in {"CNV", "Macular Hole", "DME"}:
            idx = min(idx + 1, len(order) - 1)
        return order[idx]

    # -- main entry ----------------------------------------------
    def evaluate(self, model_result: ModelResult, case: CaseInput) -> Recommendation:
        r = self.rules
        disclaimer = (r.get("report", {}) or {}).get("disclaimer", "").strip()
        differential = [
            {"class": k, "probability": round(v, 4)}
            for k, v in model_result.sorted_items()[:3]
        ]

        # 1) OOD gate
        ood = model_result.ood_score
        if ood is not None and ood >= float(r.get("ood_reject_score", 1.0)):
            return Recommendation(
                predicted_class=None, predicted_group=None,
                confidence=0.0, margin=0.0, abstained=True, ood_rejected=True,
                urgency=Urgency(r["abstain_action"]["urgency"]),
                recommendation_text=(
                    "Input does not appear to be an in-distribution macular OCT "
                    "B-scan; no automated impression produced. "
                    + r["abstain_action"]["recommendation"].strip()
                ),
                differential=differential,
                rules_version=int(r.get("version", 1)),
                model_version=model_result.model_version,
                disclaimer=disclaimer,
            )

        # 2) confidence + margin
        items = model_result.sorted_items()
        (top_key, top_p) = items[0]
        second_p = items[1][1] if len(items) > 1 else 0.0
        margin = top_p - second_p
        abstain = top_p < float(r.get("min_confidence", 0.0)) or margin < float(
            r.get("min_margin", 0.0)
        )

        if abstain:
            return Recommendation(
                predicted_class=None,
                predicted_group=None,
                confidence=round(top_p, 4),
                margin=round(margin, 4),
                abstained=True,
                ood_rejected=False,
                urgency=Urgency(r["abstain_action"]["urgency"]),
                recommendation_text=r["abstain_action"]["recommendation"].strip(),
                guideline_refs=refs_for(top_key),
                differential=differential,
                rules_version=int(r.get("version", 1)),
                model_version=model_result.model_version,
                disclaimer=disclaimer,
            )

        # 3) confident impression -> urgency
        urgency = self._urgency_for(top_key)
        urgency = self._escalate_context(urgency, case, top_key)
        group = self.lm.group(top_key)

        text = _impression_text(top_key, top_p, urgency)
        return Recommendation(
            predicted_class=top_key,
            predicted_group=group,
            confidence=round(top_p, 4),
            margin=round(margin, 4),
            abstained=False,
            ood_rejected=False,
            urgency=urgency,
            recommendation_text=text,
            guideline_refs=refs_for(top_key) if (r.get("report", {}) or {}).get("cite_guidelines") else [],
            differential=differential,
            rules_version=int(r.get("version", 1)),
            model_version=model_result.model_version,
            disclaimer=disclaimer,
        )


_URGENCY_PHRASE = {
    Urgency.none: "No referral indicated on this scan; routine screening interval.",
    Urgency.routine: "Routine referral / monitoring appropriate.",
    Urgency.soon: "Refer for specialist review within ~1 week.",
    Urgency.urgent: "Expedite specialist referral (24-72h).",
    Urgency.emergent: "Same-day ophthalmology assessment.",
}


def _impression_text(class_key: str, prob: float, urgency: Urgency) -> str:
    return (
        f"OCT features most consistent with {class_key} "
        f"(calibrated probability {prob:.0%}). {_URGENCY_PHRASE[urgency]}"
    )
