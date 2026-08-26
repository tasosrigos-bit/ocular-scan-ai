"""Basic and agentic RAG pipelines.

The two ways of answering compared in Phase B:

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
