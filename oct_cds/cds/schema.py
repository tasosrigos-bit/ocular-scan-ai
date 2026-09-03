"""Typed I/O for the clinical decision support layer."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Urgency(str, Enum):
    none = "none"
    routine = "routine"       # < 1 month
    soon = "soon"             # < 1 week
    urgent = "urgent"         # < 24-72h specialist
    emergent = "emergent"     # same day


class CaseInput(BaseModel):
    """What the clinician / PACS supplies for one B-scan."""

    image_path: str
    eye: str = "unknown"                       # OD | OS | unknown
    patient_age: int | None = None
    diabetic: bool | None = None
    symptoms: list[str] = Field(default_factory=list)
    visual_acuity_logmar: float | None = None
    acquisition_device: str | None = None


class ModelResult(BaseModel):
    """Output of the (calibrated) classifier for one B-scan."""

    probs: dict[str, float]                    # class_key -> calibrated probability
    logits: dict[str, float] | None = None
    temperature: float = 1.0
    ood_score: float | None = None             # higher = more out-of-distribution
    model_version: str = "unknown"
    calibrator_version: str = "unknown"

    @field_validator("probs")
    @classmethod
    def _probs_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        s = sum(v.values())
        if not (0.98 <= s <= 1.02):
            raise ValueError(f"probs must sum to ~1.0, got {s:.3f}")
        return v

    def top(self) -> tuple[str, float]:
        k = max(self.probs, key=self.probs.get)
        return k, self.probs[k]

    def sorted_items(self) -> list[tuple[str, float]]:
        return sorted(self.probs.items(), key=lambda kv: kv[1], reverse=True)


class Recommendation(BaseModel):
    predicted_class: str | None                # None when the model abstains
    predicted_group: str | None
    confidence: float
    margin: float
    abstained: bool
    ood_rejected: bool
    urgency: Urgency
    recommendation_text: str
    guideline_refs: list[str] = Field(default_factory=list)
    differential: list[dict] = Field(default_factory=list)
    rules_version: int = 1
    model_version: str = "unknown"
    disclaimer: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
