"""Basic and agentic RAG pipelines.

The two ways of answering compared in the pipeline comparison:

- **basic** - one retrieval, then one generation. This is
  :func:`ocular.rag.generate.answer`, wrapped here so both pipelines share a name
  and signature.
- **agentic** - a LangGraph ReAct agent that is given the ``search_corpus`` tool
  and decides for itself when to search, may reformulate the query and search
  again, and then writes the answer. It can gather evidence over several steps
  rather than a single fixed retrieval.

Both return the same :class:`~ocular.rag.generate.RAGResult` - an answer plus the
passages it was built from - so the evaluation and the deployed application treat
them interchangeably. For the agentic pipeline the passages are collected from
every ``search_corpus`` call the agent made during the run.
"""
from __future__ import annotations

import os
import threading
from typing import TypedDict

from pydantic import BaseModel

from ocular import config  # noqa: F401 - loads .env (GEMINI_API_KEY)
from ocular.rag import generate, llm, rerank, tools
from ocular.rag.generate import RAGResult
from ocular.rag.index import Index
from ocular.rag.retrieve import Hit

_AGENT_SYSTEM = (
    "You are a clinical decision-support assistant for ophthalmologists. Use the "
    "search_corpus tool to find evidence in the ophthalmology literature before you "
    "answer; if the first results are insufficient, search again with a refined query. "
    "Answer using ONLY the retrieved passages and cite them by number, like [1] or [2]. "
    "If your searches do not surface relevant evidence, say so plainly rather than "
    "guessing or drawing on outside knowledge. You support the clinician's judgement and "
    "do not replace it; do not give definitive diagnoses or individual treatment directives."
)


def basic_answer(
    query: str,
    idx: Index,
    *,
    mode: str = "hybrid",
    k: int = 5,
    rerank_model: str | None = rerank.DEFAULT_RERANKER,
    model: str = llm.MODEL,
) -> RAGResult:
    """Basic RAG: a single retrieval followed by one generation."""
    return generate.answer(query, idx, mode=mode, k=k, rerank_model=rerank_model, model=model)


def agentic_answer(
    query: str,
    idx: Index,
    *,
    mode: str = "hybrid",
    k: int = 5,
    rerank_model: str | None = rerank.DEFAULT_RERANKER,
    model: str = llm.MODEL,
) -> RAGResult:
    """Agentic RAG: a LangGraph agent decides how to search, then answers.

    The agent is given a ``search_corpus`` tool bound to ``idx`` and the retrieval
    configuration. A collector records every passage returned across the agent's
    searches; those become the result's contexts.

    Parameters mirror :func:`basic_answer`.

    Returns
    -------
    RAGResult
        The agent's final answer and the passages it retrieved.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent

    collected: list[Hit] = []
    tool = tools.make_search_corpus(
        idx, mode=mode, k=k, rerank_model=rerank_model, collector=collected
    )
    chat = ChatGoogleGenerativeAI(
        model=model, google_api_key=os.getenv("GEMINI_API_KEY"), max_retries=5
    )
    agent = create_react_agent(chat, [tool], prompt=_AGENT_SYSTEM)
    out = agent.invoke({"messages": [{"role": "user", "content": query}]})

    # Gemini may return content as a list of blocks rather than a plain string.
    content = out["messages"][-1].content
    answer = content if isinstance(content, str) else "".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )
    return RAGResult(query=query, answer=answer, contexts=collected)


# --- Search-only agent for the basic vs agentic comparison -------------------
#
# In the comparison only the searching differs. The agent decides what to search
# for and when to stop, but it neither picks the final passages nor writes the
# answer: every passage it finds goes into one pool, the pool is reranked against
# the original question, and the top five are answered from by the same step as
# the basic pipeline. Its first search is the original question, so its pool
# always holds the basic pipeline's five passages and the comparison asks only
# whether searching again finds better ones.

MAX_SEARCHES = 5
_SEARCH_LOCK = threading.Lock()

_DECIDE = """You are gathering evidence from the ophthalmology literature to answer a clinician's
question. You cannot answer it yourself; you decide whether the passages found so far are enough,
and if not, what to search for next.

QUESTION:
{question}

SEARCHES SO FAR:
{queries}

PASSAGES FOUND SO FAR:
{passages}

Set sufficient=true if the passages state the answer to the question. Otherwise set
sufficient=false and write next_query, a new search that is likely to find the missing
information. A good next query differs from every earlier search: expand abbreviations
(DME -> diabetic macular oedema), use synonyms or more specific clinical terms, focus on
the part of the question the passages do not yet answer, or ask about one part of a
two-part question. Give a one-sentence reason first."""


class SearchDecision(BaseModel):
    """The agent's choice after each search."""

    reason: str
    sufficient: bool
    next_query: str = ""


class AgenticSearch(BaseModel):
    """What the agent searched for, everything it found, and the five it hands on."""

    model_config = {"arbitrary_types_allowed": True}

    queries: list[str]
    reasons: list[str]
    pool: list[Hit]
    top: list[Hit]


class _SearchState(TypedDict):
    queries: list[str]
    reasons: list[str]
    pool: list[Hit]
    next_query: str


def _key(h: Hit) -> tuple[str, str]:
    return h.chunk.doc_id, h.chunk.text


def search_once(store, query: str, *, mode: str = "hybrid", k: int = 5, candidates: int = 50,
                rerank_model: str = rerank.DEFAULT_RERANKER) -> list[Hit]:
    """One search as in the basic pipeline: retrieve candidates, rerank, keep the top ``k``."""
    from ocular.rag import retrieve

    # The embedder and the reranker share one device, so searches from several
    # threads run one at a time while the model calls around them overlap.
    with _SEARCH_LOCK:
        found = retrieve.retrieve(store, query, k=candidates, mode=mode, candidates=candidates)
        return rerank.rerank(query, found, model_name=rerank_model, k=k)


def agentic_search(
    question: str,
    store,
    *,
    mode: str = "hybrid",
    k: int = 5,
    candidates: int = 50,
    rerank_model: str = rerank.DEFAULT_RERANKER,
    max_searches: int = MAX_SEARCHES,
    model: str = llm.MODEL,
) -> AgenticSearch:
    """Search with an agent that may reformulate, then keep the best ``k`` of all it found.

    A LangGraph graph of two nodes: ``search`` runs one search (the first is the
    original question) and adds its top ``k`` to the pool; ``decide`` shows the
    model the question, the searches so far and the pooled passages, and the model
    either stops or writes the next query. The loop ends when the model stops,
    repeats an earlier query, or ``max_searches`` searches have run. The pool is
    then reranked against the original question and its top ``k`` returned.
    """
    from langgraph.graph import END, START, StateGraph

    def search(state: _SearchState) -> dict:
        query = state["next_query"]
        seen = {_key(h) for h in state["pool"]}
        new = [h for h in search_once(store, query, mode=mode, k=k, candidates=candidates,
                                      rerank_model=rerank_model) if _key(h) not in seen]
        return {"queries": [*state["queries"], query], "pool": [*state["pool"], *new]}

    def decide(state: _SearchState) -> dict:
        passages = "\n\n".join(f"[{i}] ({h.chunk.title})\n{h.chunk.text}" for i, h in enumerate(state["pool"], 1))
        queries = "\n".join(f"{i}. {q}" for i, q in enumerate(state["queries"], 1))
        d = llm.generate_structured(_DECIDE.format(question=question, queries=queries, passages=passages),
                                    SearchDecision, model=model)
        repeated = d.next_query.strip().lower() in {q.strip().lower() for q in state["queries"]}
        stop = d.sufficient or not d.next_query.strip() or repeated
        return {"reasons": [*state["reasons"], d.reason], "next_query": "" if stop else d.next_query.strip()}

    def route(state: _SearchState) -> str:
        if not state["next_query"] or len(state["queries"]) >= max_searches:
            return END
        return "search"

    graph = StateGraph(_SearchState)
    graph.add_node("search", search)
    graph.add_node("decide", decide)
    graph.add_edge(START, "search")
    # After the last allowed search there is nothing left to decide.
    graph.add_conditional_edges("search", lambda s: END if len(s["queries"]) >= max_searches else "decide")
    graph.add_conditional_edges("decide", route)
    out = graph.compile().invoke({"queries": [], "reasons": [], "pool": [], "next_query": question})

    with _SEARCH_LOCK:
        top = rerank.rerank(question, out["pool"], model_name=rerank_model, k=k)
    return AgenticSearch(queries=out["queries"], reasons=out["reasons"], pool=out["pool"], top=top)
