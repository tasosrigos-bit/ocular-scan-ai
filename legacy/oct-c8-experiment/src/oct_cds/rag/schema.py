"""Typed objects for the RAG narrator."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Passage(BaseModel):
    """One `##` section of one knowledge-base entry."""

    id: str                      # "amd#overview"
    entry_id: str                # "amd"  (filename stem)
    entry_title: str             # "Age-Related Macular Degeneration and Drusen"
    heading: str                 # "Overview"
    text: str
    covers: list[str]            # Part 1 class keys this entry is the reference for
    sources: list[str] = Field(default_factory=list)
    kb_version: int = 1

    @property
    def is_model_behavior_note(self) -> bool:
        return self.heading.strip().lower() == "model behavior note"

    def cite_label(self) -> str:
        return f"{self.entry_title} — {self.heading}"


class RetrievedPassage(BaseModel):
    passage: Passage
    score: float                 # similarity (1.0 for deterministic hits)
    why: str                     # "predicted-class entry" | "model-behavior note" | "semantic"


class GroundedNarrative(BaseModel):
    """The verified output the report embeds."""

    text: str                    # "" when rejected (see raw_text)
    raw_text: str = ""           # the LLM's unedited output — kept for inspection
    citations: list[dict] = Field(default_factory=list)   # [{id, label, sources, snippet}]
    verified: bool = False
    flags: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    model: str = ""
    kb_version: int = 1
    retrieved_ids: list[str] = Field(default_factory=list)
