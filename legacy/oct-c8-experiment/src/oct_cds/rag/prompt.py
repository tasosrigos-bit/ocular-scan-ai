"""Prompt construction for the narrator.

The system prompt is the first line of the "LLM never decides" guardrail
(`verify.py` is the enforced second line). The citation format is spelled out
with a worked example because small instruct models tend to produce good,
grounded content but skip the marker.
"""

from __future__ import annotations

from oct_cds.cds.schema import CaseInput, Recommendation
from oct_cds.rag.schema import RetrievedPassage

SYSTEM = """\
You write a short narrative that explains a retinal OCT finding to a clinician,
grounded ONLY in the reference passages you are given.

============ CITATION FORMAT — the rule most often missed ============
Every sentence that states a clinical fact MUST end with the id of the passage
it came from, in square brackets, e.g.  [amd#overview].
- Use ids exactly as written in the REFERENCE PASSAGES list.
- A sentence may carry more than one: "... can progress to central vision loss
  [amd#overview][amd#clinical-significance]."
- A response containing no [ ] markers is rejected automatically.

WORKED EXAMPLE
Given these passages:
  [xx#overview]  (Example Disease - Overview)
  Example disease is a build-up of fluid under the retina. It usually affects
  one eye and often resolves on its own.
  [xx#model-behavior-note]  (Example Disease - Model Behavior Note)
  On external validation this classifier reached 71% sensitivity for this class
  (n=7) and twice confused it with normal.
CORRECT narrative:
  "The OCT is classified as example disease, a build-up of subretinal fluid that
  usually affects one eye and often resolves without treatment [xx#overview]. In
  this project's external validation the classifier reached 71% sensitivity for
  this class on 7 scans and twice confused it with normal [xx#model-behavior-note],
  so an automated call here carries real uncertainty."
NOT acceptable (no citations):
  "The OCT shows example disease, which usually resolves on its own. The model is
  71% sensitive for this class."
(The ids above - xx#overview etc. - are only for this example. In your answer use
the real ids from the REFERENCE PASSAGES list, e.g. cnv#overview.)

============ THE DECISION IS FIXED ============
The impression, differential, and triage urgency below were decided by a separate
rule-based system. Do not change, question, hedge, re-rank, or override them. Do
not state your own urgency or referral timeframe for this patient.

============ CONTENT ============
- Use ONLY the reference passages. Do not add facts from your own knowledge.
- "Model Behavior Note" passages describe how this classifier performed in PAST
  validation - context about model reliability, NOT statements about this
  patient. Frame them that way.
- 120-180 words, plain clinical prose, no headings, no bullet lists.
- If the impression is uncertain / abstained, explain what the differential could
  represent and why specialist review is reasonable - still cited, still no
  urgency call.
"""


def _differential_line(rec: Recommendation) -> str:
    return ", ".join(
        f"{d['class']} {float(d['probability']):.0%}" for d in (rec.differential or [])
    ) or "n/a"


def build_user_prompt(
    rec: Recommendation, case: CaseInput, retrieved: list[RetrievedPassage]
) -> str:
    fixed = [
        "FIXED DECISION (do not change):",
        f"  impression: {rec.predicted_class or 'uncertain / abstained'}",
        f"  abstained: {str(rec.abstained).lower()}",
        f"  ood_rejected: {str(rec.ood_rejected).lower()}",
        f"  model confidence: {rec.confidence:.0%}",
        f"  differential: {_differential_line(rec)}",
        f"  triage urgency (fixed, shown to clinician separately): {rec.urgency.value}",
        f"  eye: {case.eye}   device: {case.acquisition_device or 'unknown'}",
    ]
    passages = ["REFERENCE PASSAGES (cite by the id in brackets):"]
    for rp in retrieved:
        p = rp.passage
        passages.append(f"\n[{p.id}]  ({p.cite_label()})\n{p.text}")

    task = (
        "\nTASK: write the narrative now. Follow the CITATION FORMAT exactly - "
        "end every factual sentence with a [passage#id] from the list above. "
        "A narrative with no [ ] markers will be rejected and discarded."
    )
    return "\n".join(fixed) + "\n\n" + "\n".join(passages) + "\n" + task
