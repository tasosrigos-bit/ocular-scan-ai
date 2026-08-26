"""Reference-free evaluation of the RAG system.

We measure a configuration without any gold answers or relevance labels, using a
language model as the judge. Three metrics, the standard reference-free set
(popularised by RAGAS), are scored:

- **faithfulness** - is every claim in the answer supported by the retrieved
  passages? This is the hallucination check, and it matters most for a clinical
  tool.
- **answer relevance** - does the answer actually address the question?
- **context precision** - what fraction of the retrieved passages are relevant to
  the question? This is a label-free proxy for retrieval quality, standing in for
  recall@k / MRR, which would need gold relevance labels.

All three are produced in a single judge call per question, which keeps the free
tier's rate limits manageable across a sweep. The judge is held at temperature 0
and pinned to one model so scores are comparable across configurations. Two
honest limitations follow from this design: the judge is the sole authority, so
its biases become the numbers (mitigate with a fixed judge and a small human
spot-check), and a faithful answer built on wrongly-retrieved passages can still
be wrong - faithfulness is grounding, not factual correctness.
"""
from __future__ import annotations

import time
from typing import Callable

from pydantic import BaseModel

from ocular.rag import generate, llm
from ocular.rag.generate import RAGResult
from ocular.rag.index import Index
from ocular.rag.questions import Question
from ocular.rag.retrieve import Hit

_JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator of a retrieval-augmented answer produced "
    "by a clinical ophthalmology assistant. Judge only what you are given."
)

_JUDGE_PROMPT = """Score each metric from 0.0 to 1.0 and give a one-sentence reason.

- faithfulness: is EVERY factual claim in the answer supported by the context passages?
  1.0 = fully supported; 0.0 = unsupported or contradicted. An answer that correctly says
  the context does not contain the information is faithful (1.0).
- answer_relevance: does the answer directly and completely address the question?
  1.0 = fully on point; 0.0 = off-topic or evasive.
- context_precision: what fraction of the context passages are relevant to the question?
  1.0 = all relevant; 0.0 = none relevant.

QUESTION:
{query}

CONTEXT PASSAGES:
{context}

ANSWER:
{answer}"""


class Judgement(BaseModel):
    """The judge's scores for one answer."""

    faithfulness: float
    answer_relevance: float
    context_precision: float
    reasoning: str


def judge(query: str, answer: str, contexts: list[Hit], model: str = llm.MODEL) -> Judgement:
    """Score one answer on the three reference-free metrics with a single LLM call.

    Parameters
    ----------
    query : str
        The question that was asked.
    answer : str
        The answer produced by the RAG system.
    contexts : list of Hit
        The passages the answer was generated from.
    model : str, optional
        The judge model. Held fixed across a sweep for comparability.

    Returns
    -------
    Judgement
        The three scores and a short rationale.
    """
    context = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(contexts, 1))
    prompt = _JUDGE_PROMPT.format(query=query, context=context, answer=answer)
    return llm.generate_structured(
        prompt, Judgement, temperature=0.0, model=model, system=_JUDGE_SYSTEM
    )


def evaluate(
    questions: list[Question],
    idx: Index,
    *,
    mode: str = "hybrid",
    k: int = 5,
    rerank_model: str | None = None,
    gen_model: str = llm.MODEL,
    judge_model: str = llm.MODEL,
    limit: int | None = None,
    answer_fn: Callable[..., RAGResult] = generate.answer,
) -> dict[str, float]:
    """Run one configuration over the question set and average its metrics.

    For each question the system answers it (:func:`ocular.rag.generate.answer`)
    and the judge scores the answer. The per-question scores are averaged into the
    configuration's result, alongside the mean per-question latency.

    Parameters
    ----------
    questions : list of Question
        The evaluation questions.
    idx : Index
        The index this configuration retrieves from.
    mode, k, rerank_model, gen_model
        The configuration, passed through to :func:`ocular.rag.generate.answer`.
    judge_model : str, optional
        The judge model, held fixed across the sweep.
    limit : int, optional
        Evaluate only the first ``limit`` questions (useful for a quick check).

    Returns
    -------
    dict of str to float
        Mean ``faithfulness``, ``answer_relevance``, ``context_precision``, the
        question count ``n``, and mean ``latency_s`` per question.
    """
    from tqdm import tqdm

    qs = questions[:limit] if limit else questions
    faith: list[float] = []
    relevance: list[float] = []
    precision: list[float] = []
    latencies: list[float] = []

    for q in tqdm(qs, desc="evaluating"):
        t0 = time.time()
        result = answer_fn(
            q.question, idx, mode=mode, k=k, rerank_model=rerank_model, model=gen_model
        )
        latencies.append(time.time() - t0)
        verdict = judge(q.question, result.answer, result.contexts, model=judge_model)
        faith.append(verdict.faithfulness)
        relevance.append(verdict.answer_relevance)
        precision.append(verdict.context_precision)

    n = len(qs)
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0
    return {
        "n": n,
        "faithfulness": mean(faith),
        "answer_relevance": mean(relevance),
        "context_precision": mean(precision),
        "latency_s": mean(latencies),
    }
