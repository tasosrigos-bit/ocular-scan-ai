"""Retrieval metrics against each question's known source, with no language model.

Every question in the full-corpus set was written from one passage of one article,
and keeps both. Retrieval can therefore be scored directly, by whether it finds that
article and that passage, which is fast, deterministic and free, so every retrieval
configuration in the retrieval experiments can be compared on the same footing.

Three levels of relevance are scored. A retrieved passage is **from the source** when
it comes from the article the question was written from. It is **the evidence**
when, in addition, it shares at least half its text with the seed passage, measured
on overlapping five-word sequences in either direction, so a short chunk lying
inside the seed passage counts, and so does a long chunk that contains it. That
makes the evidence level fair across chunking strategies, whose passages are cut
differently from the seed. A passage is **an answer** passage when it is one of the
chunks, in the searched collection's chunking, that contain the sentences the
question's answer was copied from; these gold chunks are mapped once, when the
question set is built, and this level is scored only when they are given.

All levels are lower bounds on retrieval quality. The questions are general, so
other articles may answer them too, and a passage from another article that
answers the question still counts as a miss. The bias is the same for every
configuration, so comparisons between them stay fair.

For each question and a cut-off ``k`` the metrics are Hit@1 and Hit@k (a relevant
passage at rank one, or anywhere in the top ``k``), the reciprocal rank of the first
relevant passage (MRR, zero when there is none) and nDCG@k with binary gains, which
also rewards several passages from the source.
"""
from __future__ import annotations

import math
import re

SHINGLE = 5
EVIDENCE_OVERLAP = 0.5


def _shingles(text: str, n: int = SHINGLE) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i:i + n]) for i in range(max(len(words) - n + 1, 1))}


def is_evidence(passage: str, evidence: str, threshold: float = EVIDENCE_OVERLAP) -> bool:
    """Whether a retrieved passage shares at least ``threshold`` of its text with the evidence.

    The overlap is the number of shared five-word sequences divided by the number in
    the shorter of the two texts, so containment in either direction counts.
    """
    a, b = _shingles(passage), _shingles(evidence)
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= threshold


def relevance(hits, source_doc_id: str, evidence: str) -> tuple[list[int], list[int]]:
    """Binary relevance of each retrieved passage, at the article and evidence levels."""
    article = [int(h.chunk.doc_id == source_doc_id) for h in hits]
    found = [int(a and is_evidence(h.chunk.text, evidence)) for a, h in zip(article, hits)]
    return article, found


def _metrics(rel: list[int], k: int) -> dict[str, float]:
    rel = rel[:k]
    first = next((i for i, r in enumerate(rel) if r), None)
    ideal = sum(1 / math.log2(i + 2) for i in range(sum(rel)))
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    return {
        "hit_at_1": float(bool(rel[:1] and rel[0])),
        f"hit_at_{k}": float(first is not None),
        "mrr": 0.0 if first is None else 1 / (first + 1),
        f"ndcg_at_{k}": dcg / ideal if ideal else 0.0,
    }


def score(hits, source_doc_id: str, evidence: str, k: int = 5, gold: set | None = None) -> dict[str, float]:
    """All metrics for one question's retrieved passages, prefixed by level.

    ``gold`` holds the ``(doc_id, text)`` of the chunks, in the searched collection's
    chunking, that contain the question's answer sentences. When given, an
    ``answer`` level is scored too: a retrieved passage is relevant when it is one of
    them. This level is fair across chunking strategies, since every strategy has
    chunks that contain the sentences.
    """
    article, found = relevance(hits, source_doc_id, evidence)
    out = {
        **{f"article_{m}": v for m, v in _metrics(article, k).items()},
        **{f"evidence_{m}": v for m, v in _metrics(found, k).items()},
    }
    if gold is not None:
        answer = [int((h.chunk.doc_id, h.chunk.text) in gold) for h in hits]
        out.update({f"answer_{m}": v for m, v in _metrics(answer, k).items()})
    return out
