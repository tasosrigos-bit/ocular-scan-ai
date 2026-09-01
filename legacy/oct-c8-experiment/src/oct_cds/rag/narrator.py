"""Orchestrator: an already-decided `Recommendation` -> a verified, cited
narrative (or a clean fall back to Part 1's templated text).

The narrator NEVER touches the decision. It reads `Recommendation` fields, pulls
passages from the knowledge base, asks the LLM to explain, verifies the result,
and returns it. On `Normal`, abstain, or OOD-reject there is no LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass

from oct_cds.common.logging import get_logger
from oct_cds.cds.schema import CaseInput, Recommendation
from oct_cds.rag.cache import NarrativeCache, cache_key
from oct_cds.rag.ingest import KnowledgeBase, load_knowledge_base
from oct_cds.rag.llm import LLMBackend, StubBackend
from oct_cds.rag.prompt import SYSTEM, build_user_prompt
from oct_cds.rag.retrieve import retrieve_for
from oct_cds.rag.schema import GroundedNarrative
from oct_cds.rag.verify import verify_narrative

log = get_logger(__name__)

_SKIP_CLASSES = {"Normal"}


@dataclass
class NarratorResult:
    rag_used: bool                       # False => caller uses Part 1's render_text
    narrative: GroundedNarrative | None
    reason: str = ""                     # why RAG was skipped, if it was


class Narrator:
    def __init__(
        self,
        backend: LLMBackend | None = None,
        kb: KnowledgeBase | None = None,
        index=None,
        embedder=None,
        *,
        strict_verify: bool = True,
        cache: NarrativeCache | None = None,
        max_tokens: int = 400,
        temperature: float = 0.0,
        retry_uncited: int = 1,
    ):
        self.backend = backend or StubBackend()
        self.kb = kb or load_knowledge_base()
        self.index = index
        self.embedder = embedder
        self.strict_verify = strict_verify
        self.cache = cache if cache is not None else NarrativeCache()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retry_uncited = int(retry_uncited)

    # -- main entry ------------------------------------------------
    def narrate(self, rec: Recommendation, case: CaseInput) -> NarratorResult:
        if rec.ood_rejected:
            return NarratorResult(False, None, "input rejected as out-of-distribution")
        if rec.abstained or rec.predicted_class is None:
            return NarratorResult(False, None, "model abstained — no class to ground on")
        if rec.predicted_class in _SKIP_CLASSES:
            return NarratorResult(False, None, f"{rec.predicted_class}: no pathology narrative")

        retrieved = retrieve_for(rec, self.kb, self.index, self.embedder)
        if not retrieved:
            return NarratorResult(False, None, "no knowledge-base passages retrieved")
        retrieved_ids = [rp.passage.id for rp in retrieved]
        model_id = getattr(self.backend, "model_id", self.backend.name)
        kb_version = max(p.kb_version for p in self.kb.passages)

        key = cache_key(rec, kb_version=kb_version, model=model_id, retrieved_ids=retrieved_ids)
        cached = self.cache.get(key)
        if cached is not None:
            return NarratorResult(True, GroundedNarrative(**cached))

        user = build_user_prompt(rec, case, retrieved)
        raw, vr, attempts = self._generate_verified(user, rec, set(retrieved_ids))
        flags = list(vr.flags)
        if attempts > 1:
            flags.append(f"took {attempts} attempts")

        if vr.hard_fail:
            log.warning("narrative rejected after %d attempt(s) (%s) -> Part 1 fallback",
                        attempts, vr.flags)
            gn = GroundedNarrative(
                text="", raw_text=raw.strip(), verified=False, flags=flags,
                fallback_used=True, model=model_id, kb_version=kb_version,
                retrieved_ids=retrieved_ids,
            )
            self.cache.put(key, gn.model_dump())
            return NarratorResult(True, gn)

        gn = GroundedNarrative(
            text=raw.strip(),
            raw_text=raw.strip(),
            citations=self._citations(vr.cited_ids),
            verified=True,
            flags=flags,
            fallback_used=False,
            model=model_id,
            kb_version=kb_version,
            retrieved_ids=retrieved_ids,
        )
        self.cache.put(key, gn.model_dump())
        return NarratorResult(True, gn)

    # -- helpers --------------------------------------------------
    _CITE_FLAG = ("no citations", "citations not in retrieved set")

    def _generate_verified(self, user: str, rec, rid_set: set):
        """Generate, verify, and retry up to `retry_uncited` times when the ONLY
        problem is missing / bad citations (small models get the content right
        but skip the marker)."""
        msg = user
        raw = vr = None
        for attempt in range(1, self.retry_uncited + 2):
            raw = self.backend.generate(
                SYSTEM, msg, max_tokens=self.max_tokens, temperature=self.temperature
            )
            vr = verify_narrative(raw, rec, rid_set, strict=self.strict_verify)
            if not vr.hard_fail:
                return raw, vr, attempt
            if not all(any(k in f for k in self._CITE_FLAG) for f in vr.flags):
                return raw, vr, attempt          # a non-citation failure — don't retry
            if attempt <= self.retry_uncited:
                msg = user + (
                    f"\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: {vr.flags}. "
                    "Rewrite the narrative. Put a [passage#id] from the REFERENCE "
                    "PASSAGES list at the end of every factual sentence. Do not "
                    "invent ids."
                )
        return raw, vr, self.retry_uncited + 1

    def _citations(self, ids: list[str]) -> list[dict]:
        out = []
        for pid in ids:
            p = self.kb.by_id(pid)
            if p is None:
                continue
            out.append({
                "id": pid,
                "label": p.cite_label(),
                "sources": p.sources,
                "snippet": p.text[:240] + ("…" if len(p.text) > 240 else ""),
            })
        return out
