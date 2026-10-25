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
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Literal

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
    evidence : str
        The seed passage itself, so retrieval can be scored on whether it found the
        evidence, whatever the chunking. Empty in the original 50-question set.
    subspecialty, category : str
        The model's labels for the question, used to balance the set.
    split : str
        ``train``, ``validation`` or ``test`` for prompt optimisation.
    answer_sentences : list of str
        The sentences of the seed passage the answer is based on, copied word for
        word by the writing model. Empty in the earlier sets.
    gold_chunks : dict
        For each chunking strategy, the ids of the chunks of the source article that
        hold the answer sentences, which are their Qdrant point ids. Scoring against
        these is fair across strategies, since every strategy has chunks containing
        the sentences, while the seed passage itself is one strategy's chunk.
    gold_match : dict
        For each strategy, ``"exact"`` when every answer sentence lies whole inside a
        chunk, or ``"split"`` when one was cut by a chunk boundary and was mapped to
        the chunks holding its pieces.
    """

    id: str
    question: str
    ground_truth: str
    source_doc_id: str
    source_section: str
    evidence: str = ""
    subspecialty: str = ""
    category: str = ""
    split: str = ""
    answer_sentences: list[str] = []
    gold_chunks: dict[str, list[int]] = {}
    gold_match: dict[str, str] = {}


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


# ---------------------------------------------------------------------------
# The full-corpus question set, written by a model from another family
# ---------------------------------------------------------------------------
#
# Gemini answers the questions, so a different model family writes them, which
# keeps the answering model from being tested on questions phrased the way it would
# phrase them. The prompt is the one above, word for word, with the two labels the
# set is balanced on added at the end. The set is then checked, balanced across
# subspecialties and clinical categories, and split for prompt optimisation.

# A small, inexpensive model is enough to write one question per passage, and keeps
# regenerating the set cheap. Pinned to a dated version so a rerun is reproducible.
QUESTION_MODEL = "gpt-5.4-mini-2026-03-17"
# High, for varied questions, as the original set's 0.7 was. GPT-5 models are
# reasoning models whose API accepts only 1.0, so the high setting is also the
# only one they allow; it is passed explicitly so it is recorded and cannot drift.
QUESTION_TEMPERATURE = 1.0

SUBSPECIALTIES = (
    "retina",
    "glaucoma",
    "cornea and external disease",
    "cataract and refractive",
    "neuro-ophthalmology",
    "paediatric ophthalmology and strabismus",
    "oculoplastics and orbit",
    "uveitis and ocular inflammation",
    "ocular oncology",
    "other",
)
CATEGORIES = ("diagnosis", "imaging", "management", "mechanism", "prognosis")

_LABELS = """

Finally, if relevant, label the question with the subspecialty it belongs to and its
clinical category: diagnosis (including differential diagnosis), imaging (ophthalmic
imaging interpretation), management (management and treatment choice), mechanism
(how a disease produces its clinical signs, not molecular pathways) or prognosis
(complications and prognosis).

Then give the sentences your answer comes from. First, in sentence_rules, restate these
rules in your own words. Then, in answer_sentences, copy the one to three sentences of the
passage that state your answer. Copy each one whole and exactly as it appears, without
changing, shortening or joining them. Include only sentences that state the answer, not
sentences that give background, describe a patient or report a study's numbers.

Before you set relevant, check the answer sentences you copied. If the answer rests on any sentence that
reports one patient or case, or what one study's authors did, observed or found, or one
centre's own practice, the passage does not support a general question: set relevant=false
and leave question and answer empty. Only sentences that state accepted clinical knowledge,
or a recommendation offered as general guidance, can support a question."""

# The full-corpus prompt is the original with five additions, made after trial
# runs, and the original stays as it was so the first question set is reproducible.
# The additions require a general ophthalmology question rather than one built on
# the article's own case, an answer grounded only in the passage (the correctness
# judge compares answers with it), the exclusion of one health system's care
# pathways, a single question rather than two joined together, and the exclusion
# of passages whose answer reports one patient, one study's own work or one
# centre's routine, which the model checks again on the sentences it copied.
_ONE_SOURCE = (
    "- one patient's or one case's course: what happened to them, what they were given or why\n"
    "  they were or were not treated (for example \"he did not receive surgery because ...\")\n"
    "- what one study's authors did, observed or found, or one centre's own routine, even when it\n"
    "  is worded as a general statement (for example \"we performed ...\", \"we observed that ...\",\n"
    "  \"common complications included ...\" in a study's own results)\n"
)
_HEALTH_SYSTEM = (
    "- referral pathways, care settings, follow-up arrangements or service organisation\n"
    "  specific to one health system\n"
    "- molecular, genetic or cellular pathways (signalling cascades, gene or protein function)\n"
    "  that do not change how a clinician diagnoses or manages the eye\n"
)
_GENERAL = """ The question must be a general ophthalmology question. It
asks about a condition, sign, investigation or treatment in general, the way a clinician would
ask a colleague or consult a textbook, and it must be answerable without having read this
article. Do not build the question around this article's particular patient, case or cohort.
Ask one question with a single focus. Do not join two questions with "and why" or "and how".
Base the answer only on what the passage states. Do not add facts the passage does not
contain, even if they are true."""

_FULL_PROMPT = (
    _PROMPT
    .replace("- anything a clinician would not consult", _ONE_SOURCE + _HEALTH_SYSTEM + "- anything a clinician would not consult")
    .replace("the question must stand on its own.", "the question must stand on its own." + _GENERAL)
)
assert _FULL_PROMPT.count(_ONE_SOURCE) == 1 and _FULL_PROMPT.count(_HEALTH_SYSTEM) == 1
assert _FULL_PROMPT.count(_GENERAL) == 1

_LABELLED_PROMPT = _FULL_PROMPT.replace("\n\nPASSAGE:", _LABELS + "\n\nPASSAGE:")


class _LabelledQA(BaseModel):
    """:class:`_QA` with the two labels the full-corpus set is balanced on.

    Every field is required, as OpenAI's structured outputs demand. For a passage
    that is not relevant the model returns empty text and any label, which is
    ignored.
    """

    relevant: bool
    question: str
    answer: str
    subspecialty: Literal[SUBSPECIALTIES]
    category: Literal[CATEGORIES]
    sentence_rules: str
    answer_sentences: list[str]


# Tokens spent by the writing model in this process, for a cost report.
USAGE: Counter = Counter()
# Grading and writing need no long deliberation, and a reasoning model bills its
# hidden reasoning as output tokens, so the effort is kept low to keep the set cheap.
QUESTION_REASONING = "low"
OPENAI_TIMEOUT = 60.0


def _ask_openai(
    text: str, model: str, temperature: float = QUESTION_TEMPERATURE, retries: int = 4
) -> _LabelledQA:
    """Ask an OpenAI model for one labelled question, as a validated object."""
    from openai import OpenAI

    # A reply takes seconds; the client's default 10-minute timeout let one stalled
    # request hold up a whole run, so a stalled request is retried after a minute.
    client = OpenAI(timeout=OPENAI_TIMEOUT)
    for attempt in range(retries):
        try:
            completion = client.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": _LABELLED_PROMPT.format(text=text)}],
                response_format=_LabelledQA,
                temperature=temperature,
                reasoning_effort=QUESTION_REASONING,
            )
            USAGE.update(input=completion.usage.prompt_tokens, output=completion.usage.completion_tokens)
            return completion.choices[0].message.parsed
        except Exception as exc:  # rate limits and transient API errors
            if attempt == retries - 1:
                raise
            print(f"  retry in {2 ** attempt}s ({exc.__class__.__name__})")
            time.sleep(2 ** attempt)


# A question that names its source, or that copies a run of this many words from
# its passage, is rejected: the first cannot stand on its own, and the second is
# trivially found by keyword search, which would favour BM25 in the comparison.
_SELF_REFERENCE = re.compile(
    r"\b(the passage|this passage|the study|this study|the authors|this article|"
    r"this paper|the trial|this trial|this cohort|the text)\b",
    re.IGNORECASE,
)
COPY_WORDS = 6
# A question that goes on to ask a second thing ("..., and why does it matter?",
# "... and what findings follow?") is two questions, which the prompt forbids but a
# model at high temperature still writes now and then.
_TWO_PART = re.compile(
    r"\b(and|or)\s+(why|how|what|which|when|where|whether|does|do|is|are|can|should)\b",
    re.IGNORECASE,
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


_STOPWORDS = frozenset(
    "what which when where how does should would could with from that this than into "
    "about after before between most best first more less their there these those "
    "patient patients clinician eyes".split()
)


def _content(text: str) -> set[str]:
    """The meaningful words of a question, for spotting near-duplicates."""
    return {w for w in _words(text) if len(w) >= 4 and w not in _STOPWORDS}


def _similar(a: str, b: str, threshold: float = 0.6) -> bool:
    """Whether two questions share most of their meaningful words.

    Word overlap rather than character similarity, because clinical questions often
    share a template ("What is the first-line treatment for ...") while asking
    about different conditions, which character similarity would call duplicates.
    """
    wa, wb = _content(a), _content(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= threshold


# An answer whose meaningful words mostly appear in its question tells nothing new.
ECHO_SHARE = 0.8
# The answer is asked for in the model's own words; a long run copied from the
# passage usually means it restated a vague sentence rather than answering.
ANSWER_COPY_WORDS = 10
# An answer sentence sharing less than this share of its meaningful words with the
# answer is background, and is dropped before the gold chunks are found.
SENTENCE_SHARE = 0.25
# A question about one patient rather than a condition in general.
_CASE = re.compile(
    r"\b(the patient|this patient|the right eye|the left eye|his eye|her eye|this case|the case)\b",
    re.IGNORECASE,
)


def problems(question: str, answer: str, passage: str, accepted: list[str]) -> list[str]:
    """Why a generated question fails the set's checks, empty when it passes."""
    found = []
    if not question.strip() or not answer.strip():
        found.append("empty")
    if _SELF_REFERENCE.search(question):
        found.append("refers to its source")
    q, p = _words(question), " ".join(_words(passage))
    if any(" ".join(q[i:i + COPY_WORDS]) in p for i in range(len(q) - COPY_WORDS + 1)):
        found.append(f"copies {COPY_WORDS} or more words from the passage")
    a = _words(answer)
    if any(" ".join(a[i:i + ANSWER_COPY_WORDS]) in p for i in range(len(a) - ANSWER_COPY_WORDS + 1)):
        found.append(f"answer copies {ANSWER_COPY_WORDS} or more words from the passage")
    if _TWO_PART.search(question) or question.count("?") > 1:
        found.append("asks two questions")
    if any(_similar(question, a) for a in accepted):
        found.append("near-duplicate of an earlier question")
    if _CASE.search(question):
        found.append("built on one patient or case")
    said = _content(answer)
    if said and len(said & _content(question)) / len(said) >= ECHO_SHARE:
        found.append("answer repeats the question")
    return found


# Category limits for the 75-question set, set after the first full run gave 21
# mechanism questions, several about molecular pathways, and only 5 each of imaging
# and prognosis. Scaled down proportionally for a smaller trial set.
CATEGORY_CAPS = {"mechanism": 15}
CATEGORY_FLOORS = {"imaging": 10, "prognosis": 10}


# Embedding thresholds for all-MiniLM-L6-v2, whose cosines run lower than the
# OpenAI model Chroma used (duplicates above about 0.7, relevance below about 0.4
# there). Set from the trial run's similarities.
DUPLICATE_SIMILARITY = 0.85
MIN_RELEVANCE = 0.3
SEED_WORDS = 250
# Sections skipped when seeding from article windows. Besides methods, they hold a
# study's own results, a single case, the abstract's summary of both, or front and
# back matter, which the writing model mostly rejects as not relevant, so skipping
# them before the call saves most of the cost of a set.
SEED_EXCLUDE_SECTIONS = EXCLUDE_SECTIONS + (
    "abstract", "result", "case", "acknowledg", "funding", "conflict", "competing",
    "ethic", "consent", "data availab", "author", "declaration", "supplement", "abbreviation",
)
# An answer sentence that reports one study's findings or one patient, which the
# question must not rest on.
_STUDY_SENTENCE = re.compile(
    r"\b(our (patients?|results|study|findings|case|cohort|data)|this (study|review|report|case)|"
    r"the present study|we (found|report|observed|showed|describe|present|propose))\b",
    re.IGNORECASE,
)


def seed_windows(docs, words: int = SEED_WORDS) -> list[Chunk]:
    """Cut every article into windows of whole sentences, independent of any chunking.

    Each section is split into sentences, which are packed in reading order into
    windows of at most ``words`` words, never crossing a section. Seeding questions
    from these windows rather than from one strategy's chunks keeps every chunking
    strategy on equal terms. A window is returned as a :class:`Chunk` whose
    ``index`` is its position in the article.
    """
    from ocular.rag.chunk import _split_sentences

    out = []
    for doc in docs:
        n = 0
        for section, text in doc.sections:
            window: list[str] = []
            for sentence in _split_sentences(text):
                if window and len(" ".join(window + [sentence]).split()) > words:
                    out.append(Chunk(doc.pmcid, doc.title, doc.doi, doc.license, section, " ".join(window), n))
                    n += 1
                    window = []
                window.append(sentence)
            if window:
                out.append(Chunk(doc.pmcid, doc.title, doc.doi, doc.license, section, " ".join(window), n))
                n += 1
    return out


def generate_question_set(
    chunks: list[Chunk],
    n: int = 75,
    seed: int = config.SEED,
    model: str = QUESTION_MODEL,
    min_words: int = 40,
    exclude_sections: tuple[str, ...] = EXCLUDE_SECTIONS,
    max_share_subspecialty: float = 0.2,
    max_share_category: float = 1 / 3,
    category_caps: dict[str, int] | None = None,
    category_floors: dict[str, int] | None = None,
    ask=_ask_openai,
    embed=None,
    duplicate_similarity: float = DUPLICATE_SIMILARITY,
    min_relevance: float = MIN_RELEVANCE,
) -> tuple[list[Question], Counter]:
    """Sample passages and generate a balanced, checked question set.

    As :func:`generate_questions`, passages are drawn in a seeded random order, at
    most one per article, and the model first decides whether a passage supports a
    durable clinical question at all. Each accepted question must also pass
    :func:`problems`, and no subspecialty may exceed ``max_share_subspecialty`` and
    no category ``max_share_category`` of the set, so it spans ophthalmology rather
    than following the corpus, which leans heavily towards retina.

    Parameters
    ----------
    chunks : list of Chunk
        The passages to sample from, e.g. the full corpus's section chunks.
    n : int, optional
        Number of questions. Defaults to 75.
    seed : int, optional
        Random seed for the sampling order.
    model : str, optional
        The writing model, a different family from the answering model.
    min_words : int, optional
        Skip passages shorter than this.
    exclude_sections : tuple of str, optional
        Skip passages from sections whose heading contains one of these.
    max_share_subspecialty, max_share_category : float, optional
        The most of the set one subspecialty or one category may take.
    category_caps : dict, optional
        Tighter caps for particular categories, by default 15 mechanism questions,
        since mechanism passages are plentiful and drift towards bench science.
    category_floors : dict, optional
        The fewest questions a category must have, by default 10 each of imaging and
        prognosis, which the corpus offers less often. A question is turned away
        when accepting it would leave too few places to reach the floors.
    ask : callable, optional
        ``(passage, model) -> _LabelledQA``. Replaced in tests.
    embed : callable, optional
        ``list[str] -> array`` of normalised embeddings. When given, a question is
        turned away if it is too close in meaning to an accepted one (cosine at or
        above ``duplicate_similarity``) or too far from its own answer sentences
        (below ``min_relevance``), after the method of Chroma's chunking evaluation.

    Returns
    -------
    tuple
        The questions, without a split yet, and a tally of why passages were
        passed over (``not relevant``, a failed check, or a full subspecialty or
        category).
    """
    rng = random.Random(seed)
    pool = [
        c for c in chunks
        if len(c.text.split()) >= min_words
        and not any(kw in c.section.lower() for kw in exclude_sections)
    ]
    rng.shuffle(pool)
    cap_sub = math.ceil(n * max_share_subspecialty)
    caps = {c: math.ceil(n * max_share_category) for c in CATEGORIES}
    # The default limits are for 75 questions, so they scale with a smaller set.
    scale = n / 75
    caps.update(
        {c: math.ceil(v * scale) for c, v in CATEGORY_CAPS.items()}
        if category_caps is None else category_caps
    )
    floors = (
        {c: math.floor(v * scale) for c, v in CATEGORY_FLOORS.items()}
        if category_floors is None else category_floors
    )
    if sum(floors.values()) > n:
        raise ValueError(f"category floors {floors} need more than {n} questions")

    def blocks_floors(category: str) -> bool:
        """Whether accepting one more of ``category`` would leave the floors unreachable."""
        after = per_cat + Counter({category: 1})
        still_needed = sum(max(0, need - after[c]) for c, need in floors.items())
        return n - (len(questions) + 1) < still_needed

    questions: list[Question] = []
    per_sub: Counter = Counter()
    per_cat: Counter = Counter()
    passed_over: Counter = Counter()
    seen: set[str] = set()
    accepted_vecs: list = []
    bar = tqdm(total=n, desc="writing questions")
    for c in pool:
        if len(questions) == n:
            break
        if c.doc_id in seen:
            continue
        qa = ask(c.text, model)
        if not qa.relevant:
            passed_over["not relevant"] += 1
            continue
        if failed := problems(qa.question, qa.answer, c.text, [q.question for q in questions]):
            passed_over.update(failed)
            continue
        sentences = [x.strip() for x in qa.answer_sentences if x.strip()]
        if not sentences or not all(_normalise(x) in _normalise(c.text) for x in sentences):
            passed_over["answer sentences not copied from the passage"] += 1
            continue
        if any(_STUDY_SENTENCE.search(x) for x in sentences):
            passed_over["answer sentences describe one study or case"] += 1
            continue
        # Background sentences the model included share little with the answer.
        said = _content(qa.answer)
        sentences = [x for x in sentences
                     if _content(x) and len(_content(x) & said) / len(_content(x)) >= SENTENCE_SHARE]
        if not sentences:
            passed_over["no answer sentence states the answer"] += 1
            continue
        if embed is not None:
            q_vec, s_vec = embed([qa.question, " ".join(sentences)])
            if float(q_vec @ s_vec) < min_relevance:
                passed_over["question too far from its answer sentences"] += 1
                continue
            if any(float(q_vec @ v) >= duplicate_similarity for v in accepted_vecs):
                passed_over["same meaning as an earlier question"] += 1
                continue
        if per_sub[qa.subspecialty] >= cap_sub:
            passed_over[f"subspecialty full: {qa.subspecialty}"] += 1
            continue
        if per_cat[qa.category] >= caps[qa.category]:
            passed_over[f"category full: {qa.category}"] += 1
            continue
        if blocks_floors(qa.category):
            passed_over[f"place kept for imaging or prognosis: {qa.category}"] += 1
            continue
        seen.add(c.doc_id)
        if embed is not None:
            accepted_vecs.append(q_vec)
        per_sub[qa.subspecialty] += 1
        per_cat[qa.category] += 1
        questions.append(
            Question(
                id=f"f{len(questions):03d}",
                question=qa.question.strip(),
                ground_truth=qa.answer.strip(),
                source_doc_id=c.doc_id,
                source_section=c.section,
                evidence=c.text,
                subspecialty=qa.subspecialty,
                category=qa.category,
                answer_sentences=sentences,
            )
        )
        bar.update(1)
    bar.close()
    return questions, passed_over


def _normalise(text: str) -> str:
    """Lower-case text with runs of whitespace collapsed, for verbatim checks."""
    return " ".join(text.lower().split())


# A chunk holds an answer sentence when it contains at least this share of the
# sentence's five-word sequences, so a sentence cut across a chunk boundary maps to
# the chunk holding most of it (to both when it is cut in half), and to every piece
# when no chunk holds half.
GOLD_OVERLAP = 0.5


def map_gold_chunks(questions: list[Question], chunk_sets: dict[str, list[Chunk]]) -> list[Question]:
    """Find, for every chunking strategy, the chunks that hold each question's answer sentences.

    A chunk is gold when it comes from the question's article and contains an answer
    sentence word for word (ignoring case and spacing). A sentence that no chunk
    holds whole, because a boundary cut it, maps to the chunk holding most of it,
    and the strategy's ``gold_match`` is recorded as ``"split"``.

    Parameters
    ----------
    questions : list of Question
        Questions with ``answer_sentences`` set.
    chunk_sets : dict
        Each strategy's chunks, in their stored order, so a chunk's position is its
        Qdrant point id.

    Returns
    -------
    list of Question
        The same questions with ``gold_chunks`` set, sorted ids per strategy.
    """
    from ocular.rag.eval_retrieval import _shingles

    wanted = {q.source_doc_id for q in questions}
    by_doc: dict[str, dict[str, list[int]]] = {}
    for strategy, chunks in chunk_sets.items():
        index: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            if c.doc_id in wanted:
                index.setdefault(c.doc_id, []).append(i)
        by_doc[strategy] = index

    out = []
    for q in questions:
        gold, match = {}, {}
        for strategy, chunks in chunk_sets.items():
            ids, how = set(), "exact"
            candidates = by_doc[strategy].get(q.source_doc_id, [])
            for sentence in q.answer_sentences:
                # A chunk holding the whole sentence, word for word.
                exact = {i for i in candidates if _normalise(sentence) in _normalise(chunks[i].text)}
                if exact:
                    ids |= exact
                    continue
                # Cut by a chunk boundary: the chunk holding most of it, or every
                # piece when none holds half.
                how = "split"
                want = _shingles(sentence)
                share = {i: len(want & _shingles(chunks[i].text)) / len(want) for i in candidates}
                held = {i for i, v in share.items() if v >= GOLD_OVERLAP}
                ids |= held or {i for i, v in share.items() if v > 0}
            gold[strategy], match[strategy] = sorted(ids), how
        out.append(q.model_copy(update={"gold_chunks": gold, "gold_match": match}))
    return out


SPLIT = {"train": 10, "validation": 40, "test": 25}


def split_questions(
    questions: list[Question], sizes: dict[str, int] | None = None, seed: int = config.SEED
) -> list[Question]:
    """Assign each question to train, validation or test, stratified by category and subspecialty.

    The questions are ordered by clinical category, then subspecialty, shuffled
    within each, and dealt into the splits in proportion along that order, so every
    category is spread over the three splits roughly as the whole set is. Pairs of
    questions in different splits are then swapped while a swap brings both the
    category and the subspecialty counts of every split closer to their shares of
    the whole set. The default sizes give prompt optimisation 20 percent of the
    questions it sees to learn from and 80 percent to select on, and hold out a
    third of the set that it never sees.

    Returns
    -------
    list of Question
        The same questions with ``split`` set, in their original order.
    """
    sizes = sizes or SPLIT
    n = len(questions)
    if sum(sizes.values()) != n:
        raise ValueError(f"split sizes {sizes} do not add up to {n} questions")
    rng = random.Random(seed)
    order = sorted(range(n), key=lambda i: (questions[i].category, questions[i].subspecialty, rng.random()))
    # Deal labels evenly: position k goes to the split whose running share lags
    # furthest behind its target share.
    dealt: Counter = Counter()
    labels: list[str] = [""] * n
    for k, i in enumerate(order, 1):
        name = max(sizes, key=lambda s: sizes[s] / n * k - dealt[s])
        dealt[name] += 1
        labels[i] = name

    totals = {key: Counter(getattr(q, key) for q in questions) for key in ("category", "subspecialty")}

    def imbalance() -> float:
        out = 0.0
        for key, total in totals.items():
            for name, size in sizes.items():
                got = Counter(getattr(q, key) for q, s in zip(questions, labels) if s == name)
                out += sum(abs(got[v] - c * size / n) for v, c in total.items())
        return out

    best, improved = imbalance(), True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue
                labels[i], labels[j] = labels[j], labels[i]
                now = imbalance()
                if now < best - 1e-9:
                    best, improved = now, True
                else:
                    labels[i], labels[j] = labels[j], labels[i]
    return [q.model_copy(update={"split": s}) for q, s in zip(questions, labels)]
