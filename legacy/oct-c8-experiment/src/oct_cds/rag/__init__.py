"""Part 2 — retrieval-augmented **narrative** generation.

The LLM never makes or influences the diagnostic or urgency decision. That stays
entirely with Part 1's `CDSRuleEngine` (`src/oct_cds/cds/`). This package takes
an already-decided `Recommendation`, retrieves passages from the curated
`knowledge_base/`, and generates a grounded, cited explanation for a clinician —
with an automated check (`verify.py`) that the narrative does not contradict the
decision. On any failure it falls back to Part 1's templated narrative.
"""

from oct_cds.rag.narrator import Narrator, NarratorResult
from oct_cds.rag.schema import GroundedNarrative, Passage, RetrievedPassage

__all__ = [
    "Narrator",
    "NarratorResult",
    "GroundedNarrative",
    "Passage",
    "RetrievedPassage",
]
