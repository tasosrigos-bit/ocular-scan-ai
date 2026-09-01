"""Decision-aware retrieval.

Given an already-decided `Recommendation`, assemble the passages the narrator is
allowed to use:

1. **deterministic** — every passage of the entry that `covers` the predicted class;
2. **deterministic** — the "Model Behavior Note" passage for the predicted class
   and for each differential class (this project's own validation evidence);
3. **semantic** — top-k passages for a query built from the differential.

Retrieval never re-decides anything: the predicted class comes straight from
Part 1's rule engine and is used only as a lookup key.
"""

from __future__ import annotations

from oct_cds.cds.schema import Recommendation
from oct_cds.rag.ingest import KnowledgeBase, load_knowledge_base
from oct_cds.rag.schema import RetrievedPassage


def _query_text(rec: Recommendation) -> str:
    diff = ", ".join(
        f"{d['class']} {float(d['probability']):.0%}" for d in (rec.differential or [])
    )
    cls = rec.predicted_class or "uncertain finding"
    return (
        f"Explain a macular OCT classified as {cls}. "
        f"Differential: {diff or 'n/a'}. "
        f"What is this condition, how does it look on OCT, why does it matter, "
        f"and how reliable is an automated {cls} prediction?"
    )


def retrieve_for(
    rec: Recommendation,
    kb: KnowledgeBase | None = None,
    index=None,
    embedder=None,
    *,
    k_semantic: int = 4,
) -> list[RetrievedPassage]:
    kb = kb or load_knowledge_base()
    pred = rec.predicted_class
    diff_classes = [d["class"] for d in (rec.differential or [])]
    picked: dict[str, RetrievedPassage] = {}

    def _add(p, score: float, why: str) -> None:
        if p.id not in picked:
            picked[p.id] = RetrievedPassage(passage=p, score=score, why=why)

    # 1. the predicted class's designated entry (all sections)
    if pred and (entry_id := kb.entry_for_class(pred)):
        for p in kb.passages_for_entry(entry_id):
            _add(p, 1.0, "predicted-class entry")

    # 2. model-behavior notes for predicted + differential classes
    for cls in [pred, *diff_classes]:
        if not cls:
            continue
        entry_id = kb.entry_for_class(cls)
        if not entry_id:
            continue
        for p in kb.passages_for_entry(entry_id):
            if p.is_model_behavior_note:
                _add(p, 1.0, "model-behavior note")

    # 3. semantic top-k (optional — needs an index + embedder)
    if index is not None and embedder is not None:
        qvec = embedder.encode_query(_query_text(rec))
        for pid, score in index.search(qvec, k_semantic):
            p = kb.by_id(pid)
            if p is not None:
                _add(p, score, "semantic")

    # order: predicted-class entry, then model-behavior notes, then semantic
    order = {"predicted-class entry": 0, "model-behavior note": 1, "semantic": 2}
    return sorted(picked.values(), key=lambda rp: (order[rp.why], -rp.score, rp.passage.id))
