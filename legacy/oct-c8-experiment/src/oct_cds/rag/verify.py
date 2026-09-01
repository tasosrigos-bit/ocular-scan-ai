"""Enforced guardrail: check a generated narrative against the fixed decision.

Hard failures (-> caller must fall back to Part 1's templated narrative):
  * a citation that does not resolve to a retrieved passage id
  * no citation at all
  * the narrative asserts a *different* predicted class as the finding
  * the narrative downgrades the triage relative to the fixed urgency
    (e.g. "no referral needed" when the decision is urgent) — the dangerous
    direction, matching Part 1's confident-wrong finding

`strict=True` treats every issue as a hard failure. `strict=False` still fails
on unresolved citations / missing citations, but lets softer issues through as
recorded flags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from oct_cds.cds.schema import Recommendation, Urgency

_CITES = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9#\-_ ]*)\]")


def _norm_cite(raw: str) -> str:
    return raw.strip().lower().replace(" ", "-")

# urgency tiers, low -> high
_TIER = {Urgency.none: 0, Urgency.routine: 1, Urgency.soon: 2, Urgency.urgent: 3, Urgency.emergent: 4}

# phrases that assert a LOW-urgency triage for this patient
_DOWNGRADE_PHRASES = [
    r"no referral (is )?(needed|required|indicated)",
    r"does not (require|need|warrant) (a )?referral",
    r"routine (follow[- ]?up|monitoring) (is|would be) (sufficient|appropriate|adequate) for this",
    r"can be safely (monitored|observed|discharged)",
    r"no (urgent |further )?action (is )?(needed|required)",
    r"reassur\w+ and discharge",
]

_CLASS_KEYS = ["AMD", "CNV", "CSR", "DME", "DR", "Drusen", "Macular Hole", "Normal"]
_FINDING_PATTERNS = [
    r"consistent with {c}\b",
    r"findings? (show|indicate|represent|suggest) {c}\b",
    r"diagnosis of {c}\b",
    r"this is {c}\b",
]


@dataclass
class VerifyResult:
    ok: bool
    flags: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)

    @property
    def hard_fail(self) -> bool:
        return not self.ok


def verify_narrative(
    text: str,
    rec: Recommendation,
    retrieved_ids: set[str],
    *,
    strict: bool = True,
) -> VerifyResult:
    issues: list[str] = []
    hard: list[str] = []
    low = text.lower()

    cited = [_norm_cite(c) for c in _CITES.findall(text)]
    unresolved = sorted({c for c in cited if c not in retrieved_ids})
    if unresolved:
        hard.append(f"citations not in retrieved set: {unresolved}")
    if not cited:
        hard.append("no citations")

    # class contradiction (only when a class was actually asserted)
    if rec.predicted_class and not rec.abstained and not rec.ood_rejected:
        for c in _CLASS_KEYS:
            if c == rec.predicted_class:
                continue
            rx = "|".join(pat.format(c=re.escape(c.lower())) for pat in _FINDING_PATTERNS)
            if re.search(rx, low):
                hard.append(f"asserts a different finding ({c}) than the fixed impression "
                            f"({rec.predicted_class})")
                break

    # triage downgrade
    if _TIER.get(rec.urgency, 1) >= _TIER[Urgency.soon]:
        for pat in _DOWNGRADE_PHRASES:
            if re.search(pat, low):
                issues.append(f"narrative downgrades triage (fixed urgency: {rec.urgency.value})")
                break

    if strict:
        hard += issues
        issues = []

    return VerifyResult(ok=not hard, flags=hard + issues, cited_ids=sorted(set(cited)))
