"""Per-class reference pointers attached to a CDS recommendation.

Intentionally empty. The previous placeholders cited AAO Preferred Practice
Patterns, whose terms restrict reuse (including "in an artificial intelligence
program") — not something that belongs in this repo.

Part 2 (`src/oct_cds/rag/`, `knowledge_base/`) builds a curated corpus from
NEI (public domain) and CC-BY open-access sources. Once that lands, `refs_for()`
will resolve to the `sources` of the knowledge-base entry whose `covers` list
includes the class, so Part 1 and Part 2 cite the same vetted material.

Until then `refs_for()` returns `[]` for every class and the CDS report simply
omits the references block.
"""

from __future__ import annotations

# class_key -> list of citation strings. Empty by design; see module docstring.
GUIDELINE_REFS: dict[str, list[str]] = {
    "AMD": [],
    "CNV": [],
    "CSR": [],
    "DME": [],
    "DR": [],
    "Drusen": [],
    "Macular Hole": [],
    "Normal": [],
}


def refs_for(class_key: str) -> list[str]:
    return list(GUIDELINE_REFS.get(class_key, []))
