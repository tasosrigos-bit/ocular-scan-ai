"""Generate a grounded answer from retrieved context.

This is the core of the RAG system, shared by two callers: the evaluation, which
runs it across many configurations to compare them, and the deployed application,
which runs it with the single configuration the evaluation selected. Because both
use this one function, what the evaluation measures is exactly what the app ships.

The step is single-turn: given a question and an index, it retrieves the most
relevant passages (optionally reranked), builds a prompt that instructs the model
to answer **only** from those passages and to cite them, and returns the answer
together with the contexts it was given. The contexts travel with the answer
because both callers need them - the app to show citations, the evaluation to
judge faithfulness and context precision.

The system prompt encodes the clinical posture agreed for this tool: the user is a
clinician, so the assistant gives sourced decision support rather than refusing,
but it must ground every claim in the retrieved passages, cite them, and say
plainly when the context does not contain the answer instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ocular.rag import llm, rerank, retrieve
from ocular.rag.index import Index
from ocular.rag.retrieve import Hit

_SYSTEM = (
    "You are a clinical decision-support assistant for ophthalmologists. Answer the "
    "question using ONLY the numbered context passages provided. Cite the passages you "
    "rely on by their number, like [1] or [2]. If the passages do not contain enough "
    "information to answer, say so plainly rather than guessing or drawing on outside "
    "knowledge. You support the clinician's judgement and do not replace it; do not give "
    "definitive diagnoses or individual treatment directives."
)


@dataclass
class RAGResult:
    """A grounded answer with the context it was built from.

    Attributes
    ----------
    query : str
        The question that was answered.
    answer : str
        The generated answer.
    contexts : list of Hit
        The passages given to the model, in the order they were numbered in the
        prompt. Used for citation in the app and for reference-free evaluation.
    """

    query: str
    answer: str
    contexts: list[Hit] = field(default_factory=list)


def _format_context(hits: list[Hit]) -> str:
    """Number the retrieved passages and label each with its source."""
    blocks = []
    for i, h in enumerate(hits, 1):
        source = f"{h.chunk.title} - {h.chunk.section}" if h.chunk.section else h.chunk.title
        blocks.append(f"[{i}] ({source})\n{h.chunk.text}")
    return "\n\n".join(blocks)


def answer(
    query: str,
    idx: Index,
    *,
    mode: str = "hybrid",
    k: int = 5,
    candidates: int = 50,
    rerank_model: str | None = None,
    model: str = llm.MODEL,
) -> RAGResult:
    """Answer a question from an index, grounded in retrieved passages.

    Parameters
    ----------
    query : str
        The clinician's question.
    idx : Index
        The index to retrieve from.
    mode : {"vector", "bm25", "hybrid"}, optional
        First-stage retrieval mode. Defaults to ``"hybrid"``.
    k : int, optional
        Number of passages to give the model. Defaults to 5.
    candidates : int, optional
        Pool size retrieved before reranking. Only used when ``rerank_model`` is
        set. Defaults to 50.
    rerank_model : str, optional
        A cross-encoder id to rerank with. ``None`` skips reranking and retrieves
        ``k`` directly.
    model : str, optional
        Generation model. Defaults to :data:`ocular.rag.llm.MODEL`.

    Returns
    -------
    RAGResult
        The answer and the passages it was grounded in.
    """
    if rerank_model:
        hits = retrieve.retrieve(idx, query, k=candidates, mode=mode, candidates=candidates)
        hits = rerank.rerank(query, hits, model_name=rerank_model, k=k)
    else:
        hits = retrieve.retrieve(idx, query, k=k, mode=mode, candidates=candidates)

    prompt = f"Context:\n{_format_context(hits)}\n\nQuestion: {query}"
    text = llm.generate(prompt, model=model, temperature=0.0, system=_SYSTEM)
    return RAGResult(query=query, answer=text, contexts=hits)
