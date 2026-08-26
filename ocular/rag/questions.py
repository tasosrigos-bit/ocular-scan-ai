"""Generate a synthetic evaluation question set from the corpus.

The reference-free evaluation (faithfulness, answer relevance, context precision)
needs no gold answers or relevance labels, but it still needs a set of realistic
questions to run the system on. This module produces that set with the language
model: it samples passages from the corpus and asks the model to write, for each,
one question a clinician might plausibly ask that the passage answers.

Two choices guard against the classic pitfall of synthetic questions, *leakage* -
questions that merely echo their source passage and so are trivially retrievable:

- The model is told to **paraphrase**, not to copy the passage wording.
- Passages are sampled **one per article**, so the set spans the corpus rather
  than clustering on a few documents.

A concise reference answer is generated alongside each question. It is not needed
by the reference-free metrics, but it is cheap and useful for a human spot-check
of the set's quality. Each question keeps a pointer to its source article and
section for traceability.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from ocular import config
from ocular.rag import llm
from ocular.rag.chunk import Chunk

# Where the question set is written; git-ignored under the data tree.
EVAL_DIR = config.DATA_DIR / "eval"

_PROMPT = """You are curating evaluation questions for a general ophthalmology clinical assistant.

First decide whether this passage supports a targeted clinical question that a practising
ophthalmologist would genuinely ask, across any subspecialty (retina, glaucoma, cornea,
cataract, neuro-ophthalmology, paediatric ophthalmology, oculoplastics, uveitis and others).

Set relevant=false when the passage is any of:
- a specific study's results or statistics (proportions, counts, cohort demographics, model-fit numbers)
- animal, cell or molecular bench research with no direct clinical bearing
- device or software engineering, cost, sustainability, administration, funding or conflict-of-interest detail
- a systemic or general-medicine topic that is not itself about the eye or vision, even when an
  ophthalmology article mentions it (for example systemic hypertension, cholesterol or general pharmacology)
- anything a clinician would not consult an assistant about.

Set relevant=true only for durable, clinically useful knowledge that concerns the eye or vision itself:
diagnosis and differential diagnosis, ophthalmic imaging interpretation, management and treatment
choice, disease mechanism, complications and prognosis.

If relevant, write ONE targeted question about an ocular or visual condition that a clinician would
genuinely ask and that this passage answers, plus a concise correct answer, in your own words. The question must test transferable
clinical knowledge, NOT the result of one particular study - do not ask about proportions,
sample sizes, specific figures or study findings. Do NOT copy phrases from the passage or refer
to "the passage" or "the study"; the question must stand on its own. If NOT relevant, set
relevant=false and leave question and answer empty.

PASSAGE:
{text}"""


class _QA(BaseModel):
    """The shape the model returns for one passage.

    ``relevant`` is decided first: it gates whether the passage is used at all, so
    off-topic passages (protocols, statistics, disclosures) are skipped rather than
    turned into weak questions.
    """

    relevant: bool
    question: str = ""
    answer: str = ""


class Question(BaseModel):
    """One evaluation question with its provenance.

    Attributes
    ----------
    id : str
        Stable identifier, e.g. ``"q007"``.
    question : str
        The generated question.
    ground_truth : str
        A concise reference answer, for optional human spot-checks.
    source_doc_id : str
        The article the seed passage came from.
    source_section : str
        The section heading of the seed passage.
    """

    id: str
    question: str
    ground_truth: str
    source_doc_id: str
    source_section: str


def _generate_one(text: str, model: str, retries: int = 4) -> _QA:
    """Generate one question/answer, retrying with backoff on transient errors."""
    for attempt in range(retries):
        try:
            return llm.generate_structured(
                _PROMPT.format(text=text), _QA, temperature=0.7, model=model
            )
        except Exception as exc:  # rate limits (429) and transient API errors
            if attempt == retries - 1:
                raise
            wait = 2**attempt
            print(f"  retry in {wait}s ({exc.__class__.__name__})")
            time.sleep(wait)


# Section headings whose passages tend to hold generic protocol (antibody
# dilutions, software, statistics) rather than clinical content, and so produce
# off-topic questions. Seed passages from these sections are skipped.
EXCLUDE_SECTIONS = ("method", "material")


def generate_questions(
    chunks: list[Chunk],
    n: int = 50,
    seed: int = config.SEED,
    model: str = llm.MODEL,
    min_words: int = 40,
    delay: float = 0.5,
    exclude_sections: tuple[str, ...] = EXCLUDE_SECTIONS,
) -> list[Question]:
    """Sample passages and generate one question each.

    Parameters
    ----------
    chunks : list of Chunk
        The corpus chunks to sample seed passages from.
    n : int, optional
        Number of questions to generate. Defaults to 50.
    seed : int, optional
        Random seed for reproducible sampling. Defaults to ``config.SEED``.
    model : str, optional
        Generation model. Defaults to :data:`ocular.rag.llm.MODEL`.
    min_words : int, optional
        Skip passages shorter than this, so questions are substantive. Defaults to 40.
    delay : float, optional
        Seconds to pause between calls, to stay within free-tier rate limits.

    Returns
    -------
    list of Question
        The generated questions, with provenance.
    """
    rng = random.Random(seed)
    pool = [
        c
        for c in chunks
        if len(c.text.split()) >= min_words
        and not any(kw in c.section.lower() for kw in exclude_sections)
    ]
    rng.shuffle(pool)

    # Walk the pool, at most one accepted passage per article. The model gates each
    # passage on relevance: a passage judged off-topic is skipped and the next is
    # tried, so the set fills with clinically substantive questions only.
    questions: list[Question] = []
    seen: set[str] = set()
    for c in tqdm(pool, desc="generating questions"):
        if len(questions) == n:
            break
        if c.doc_id in seen:
            continue
        qa = _generate_one(c.text, model)
        time.sleep(delay)
        if not qa.relevant or not qa.question.strip():
            continue  # off-topic passage; try the next
        seen.add(c.doc_id)
        questions.append(
            Question(
                id=f"q{len(questions):03d}",
                question=qa.question,
                ground_truth=qa.answer,
                source_doc_id=c.doc_id,
                source_section=c.section,
            )
        )
    return questions


def save_questions(questions: list[Question], path: Path | None = None) -> None:
    """Write questions to a JSON Lines file (default ``data/eval/questions.jsonl``)."""
    path = path or EVAL_DIR / "questions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(q.model_dump_json() + "\n")


def load_questions(path: Path | None = None) -> list[Question]:
    """Read questions written by :func:`save_questions`."""
    path = path or EVAL_DIR / "questions.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [Question(**json.loads(line)) for line in fh]
