"""The full-corpus question set: checks, balancing and the split, with a stub model."""
from __future__ import annotations

from collections import Counter

import pytest

from ocular.rag import questions
from ocular.rag.chunk import Chunk
from ocular.rag.questions import _LabelledQA

FILLER = " ".join(["The retina is examined with optical coherence tomography in clinic."] * 6)


def _chunk(i: int, section: str = "Discussion") -> Chunk:
    return Chunk(f"PMC{i}", f"Article {i}", None, "CC BY", section, f"Passage {i}. {FILLER}", 0)


TOPICS = ["drusen", "glaucoma cupping", "keratoconus", "optic neuritis", "uveitic glaucoma",
          "retinoblastoma", "amblyopia", "thyroid orbitopathy", "endophthalmitis", "macular hole",
          "corneal ulcer", "papilloedema", "retinal detachment", "pterygium", "scleritis",
          "choroidal melanoma", "ptosis", "cataract density", "vitreous haemorrhage", "iritis"]
OPENERS = ["Which finding best separates", "How should a clinician manage", "What mechanism underlies",
           "Which imaging sign confirms", "What prognosis follows"]


def _qa(i, sub="retina", cat="diagnosis", relevant=True, question=None):
    return _LabelledQA(
        relevant=relevant,
        question=question if question is not None else f"{OPENERS[(i // 20) % 5]} {TOPICS[i % 20]}?",
        answer=f"Answer {i}: optical coherence tomography examines the retina.",
        subspecialty=sub,
        category=cat,
        sentence_rules="Copy one to three whole sentences exactly.",
        answer_sentences=["The retina is examined with optical coherence tomography in clinic."],
    )


@pytest.mark.parametrize("question, reason", [
    ("What did the study find about drusen?", "refers to its source"),
    ("Is the retina is examined with optical coherence tomography in clinic?", "copies"),
    ("", "empty"),
])
def test_problems_catches_unusable_questions(question, reason):
    found = questions.problems(question, "an answer", FILLER, [])
    assert any(reason in f for f in found)


def test_a_paraphrased_standalone_question_passes():
    assert questions.problems("How is macular oedema confirmed on OCT?", "With OCT.", FILLER, []) == []


def test_questions_sharing_a_template_are_not_duplicates():
    first = "What is the first-line treatment for neovascular age-related macular degeneration?"
    assert questions.problems("What is the first-line treatment for acute angle-closure glaucoma?",
                              "x", FILLER, [first]) == []


def test_near_duplicates_are_rejected():
    first = "How is macular oedema confirmed on OCT imaging?"
    assert "near-duplicate" in questions.problems("How is macular oedema confirmed on OCT imagery?",
                                                  "x", FILLER, [first])[0]


def test_the_set_is_balanced_one_per_article_and_skips_irrelevant_passages():
    chunks = [_chunk(i) for i in range(60)] + [_chunk(99, "Methods")]
    cats = list(questions.CATEGORIES)

    def ask(text, model):
        i = int(text.split(".")[0].split()[-1])
        if i % 7 == 0:
            return _qa(i, relevant=False, question="")
        # Half the passages are retina, as in the corpus, so the cap has work to do.
        sub = "retina" if i % 2 else questions.SUBSPECIALTIES[i % 9]
        return _qa(i, sub=sub, cat=cats[i % 5])

    qs, passed_over = questions.generate_question_set(chunks, n=12, ask=ask)
    assert len(qs) == 12
    assert len({q.source_doc_id for q in qs}) == 12                 # one per article
    assert all(q.source_section != "Methods" for q in qs)           # method sections skipped
    assert max(Counter(q.subspecialty for q in qs).values()) <= 3   # 20 percent of 12, rounded up
    assert max(Counter(q.category for q in qs).values()) <= 4       # a third of 12
    assert passed_over["not relevant"] > 0
    assert passed_over["subspecialty full: retina"] > 0             # the cap held retina back
    assert all(q.evidence.startswith("Passage") for q in qs)


def test_the_split_has_the_planned_sizes_and_spreads_each_subspecialty():
    subs = ["retina"] * 30 + ["glaucoma"] * 15 + ["cornea and external disease"] * 15 + ["uveitis and ocular inflammation"] * 15
    qs = [questions.Question(id=f"f{i:03d}", question=f"q{i}", ground_truth="a", source_doc_id=f"PMC{i}",
                             source_section="", subspecialty=s) for i, s in enumerate(subs)]
    split = questions.split_questions(qs)
    assert Counter(q.split for q in split) == {"validation": 40, "test": 25, "train": 10}
    assert [q.id for q in split] == [q.id for q in qs]              # original order kept
    retina = Counter(q.split for q in split if q.subspecialty == "retina")
    assert retina["test"] == 10 and retina["validation"] == 16 and retina["train"] == 4
    with pytest.raises(ValueError, match="do not add up"):
        questions.split_questions(qs[:10])


def test_the_split_spreads_each_category_too():
    # Categories cycle within each subspecialty, so balancing on subspecialty alone
    # could still leave a category nearly missing from the test split.
    subs = ["retina"] * 30 + ["glaucoma"] * 15 + ["cornea and external disease"] * 15 + ["uveitis and ocular inflammation"] * 15
    cats = [questions.CATEGORIES[i % 5] if i < 45 else "management" for i in range(75)]
    qs = [questions.Question(id=f"f{i:03d}", question=f"q{i}", ground_truth="a", source_doc_id=f"PMC{i}",
                             source_section="", subspecialty=s, category=c) for i, (s, c) in enumerate(zip(subs, cats))]
    split = questions.split_questions(qs)
    for cat, total in Counter(cats).items():
        got = Counter(q.split for q in split if q.category == cat)
        for name, size in questions.SPLIT.items():
            assert abs(got[name] - total * size / 75) <= 1.5, (cat, name, got)


def test_the_original_question_file_still_loads(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text('{"id":"q000","question":"Q?","ground_truth":"A","source_doc_id":"PMC1","source_section":"Intro"}\n')
    (q,) = questions.load_questions(path)
    assert q.split == "" and q.evidence == ""


@pytest.mark.parametrize("question", [
    "What determines ciliary muscle stiffness, and why does it matter for intraocular pressure?",
    "How does ocular nocardiosis present, and what findings follow as it progresses?",
    "What is keratoconus? How is it staged?",
])
def test_two_part_questions_are_rejected(question):
    assert "asks two questions" in questions.problems(question, "an answer", FILLER, [])


def test_a_single_question_mentioning_and_passes():
    q = "Which features distinguish central serous chorioretinopathy and polypoidal vasculopathy on OCT?"
    assert questions.problems(q, "an answer", FILLER, []) == []


def test_category_floors_are_met_and_caps_hold():
    chunks = [_chunk(i) for i in range(200)]
    cats = ["mechanism"] * 6 + ["diagnosis", "management", "imaging", "prognosis"]

    def ask(text, model):
        i = int(text.split(".")[0].split()[-1])
        # Mechanism is by far the commonest, imaging and prognosis the rarest.
        return _qa(i, sub=questions.SUBSPECIALTIES[i % 9], cat=cats[i % 10],
                   question=f"{OPENERS[(i // 20) % 5]} {TOPICS[i % 20]} {questions.SUBSPECIALTIES[i % 9]}?")

    qs, passed_over = questions.generate_question_set(
        chunks, n=20, ask=ask, max_share_subspecialty=1.0,
        category_caps={"mechanism": 4}, category_floors={"imaging": 5, "prognosis": 5},
    )
    per_cat = Counter(q.category for q in qs)
    assert len(qs) == 20
    assert per_cat["imaging"] >= 5 and per_cat["prognosis"] >= 5
    assert per_cat["mechanism"] <= 4
    assert any(k.startswith("place kept for imaging or prognosis") for k in passed_over)


def test_impossible_floors_are_refused():
    with pytest.raises(ValueError, match="floors"):
        questions.generate_question_set([], n=5, category_floors={"imaging": 6}, ask=None)


def test_a_question_whose_answer_sentences_are_not_in_the_passage_is_rejected():
    chunks = [_chunk(i) for i in range(3)]

    def ask(text, model):
        i = int(text.split(".")[0].split()[-1])
        qa = _qa(i, sub=questions.SUBSPECIALTIES[i], cat=questions.CATEGORIES[i])
        if i == 0:
            qa.answer_sentences = ["A sentence the passage never contains."]
        return qa

    qs, passed_over = questions.generate_question_set(
        chunks, n=2, ask=ask, category_floors={}, category_caps={}
    )
    assert sorted(q.source_doc_id for q in qs) == ["PMC1", "PMC2"]
    assert passed_over["answer sentences not copied from the passage"] == 1
    assert qs[0].answer_sentences == ["The retina is examined with optical coherence tomography in clinic."]


def test_gold_chunks_are_found_in_every_strategy_and_only_in_the_source_article():
    sentence = "Anti-VEGF injections are the first-line treatment for neovascular macular degeneration."
    q = questions.Question(id="f000", question="q", ground_truth="a", source_doc_id="PMC1",
                           source_section="", answer_sentences=[sentence])
    doc = lambda d, text: Chunk(d, "t", None, "CC BY", "Discussion", text, 0)
    half = sentence.split()
    sets = {
        "whole": [doc("PMC1", "Intro text here."), doc("PMC1", f"Before. {sentence} After."), doc("PMC2", sentence)],
        # cut so that one chunk holds half the sentence: that chunk alone
        "half": [doc("PMC1", " ".join(half[:6])), doc("PMC1", " ".join(half[5:]))],
        # cut into three pieces, none holding half: every piece long enough to share text
        "thirds": [doc("PMC1", " ".join(half[:4])), doc("PMC1", " ".join(half[2:8])),
                   doc("PMC1", " ".join(half[6:]))],
    }
    [mapped] = questions.map_gold_chunks([q], sets)
    assert mapped.gold_chunks == {"whole": [1], "half": [0], "thirds": [0, 1]}
    assert mapped.gold_match == {"whole": "exact", "half": "split", "thirds": "split"}


@pytest.mark.parametrize("question, answer, reason", [
    ("What did the right eye show on fluorescein angiography?", "Late leakage.", "one patient"),
    ("What appears as a gray stalk-like band in the vitreous?", "A gray stalk-like band.", "repeats the question"),
])
def test_case_questions_and_echoed_answers_are_rejected(question, answer, reason):
    assert any(reason in f for f in questions.problems(question, answer, FILLER, []))


def test_seed_windows_keep_whole_sentences_within_one_section():
    from types import SimpleNamespace

    sentences = [f"Sentence number {i} about the macula and its vessels." for i in range(30)]
    doc = SimpleNamespace(pmcid="PMC1", title="t", doi=None, license="CC BY",
                          sections=[("Results", " ".join(sentences)), ("Discussion", "A short closing remark.")])
    windows = questions.seed_windows([doc], words=60)
    assert [w.index for w in windows] == list(range(len(windows)))
    assert windows[-1].section == "Discussion" and windows[-1].text == "A short closing remark."
    results = [w for w in windows if w.section == "Results"]
    assert all(len(w.text.split()) <= 60 for w in results)
    assert " ".join(w.text for w in results) == " ".join(sentences)  # nothing lost or cut


def test_embedding_filters_turn_away_duplicates_and_off_topic_questions():
    import numpy as np

    chunks = [_chunk(i) for i in range(3)]

    def ask(text, model):
        i = int(text.split(".")[0].split()[-1])
        return _qa(i, sub=questions.SUBSPECIALTIES[i], cat=questions.CATEGORIES[i])

    def embed(texts):
        # every question means the same thing, and sits close to its sentences
        return np.array([[1.0, 0.0], [0.9, 0.436]])

    qs, passed_over = questions.generate_question_set(
        chunks, n=3, ask=ask, embed=embed, category_floors={}, category_caps={}
    )
    assert len(qs) == 1 and passed_over["same meaning as an earlier question"] == 2


def test_answer_sentences_reporting_one_study_are_rejected():
    study = "Our results showed a significant reduction of intraocular pressure after the procedure."
    chunks = [Chunk(f"PMC{i}", "t", None, "CC BY", "Discussion", f"Passage {i}. {study} {FILLER}", 0) for i in range(2)]

    def ask(text, model):
        i = int(text.split(".")[0].split()[-1])
        qa = _qa(i, sub=questions.SUBSPECIALTIES[i], cat=questions.CATEGORIES[i])
        if i == 0:
            qa.answer_sentences = [study]
        return qa

    # asking for two from two passages visits both, whatever the shuffled order
    qs, passed_over = questions.generate_question_set(chunks, n=2, ask=ask, category_floors={}, category_caps={})
    assert [q.source_doc_id for q in qs] == ["PMC1"]
    assert passed_over["answer sentences describe one study or case"] == 1


def test_an_answer_copied_from_the_passage_is_rejected():
    answer = "The retina is examined with optical coherence tomography in clinic every year."
    assert any("answer copies" in f for f in questions.problems("How is the macula assessed?", answer, FILLER, []))


def test_background_sentences_are_dropped_and_the_answer_sentence_kept():
    background = "Visual stimuli in persons with normal vision inhibit spontaneous cortical activity."
    passage = f"Passage 0. {background} {FILLER}"
    chunks = [Chunk("PMC0", "t", None, "CC BY", "Discussion", passage, 0)]

    def ask(text, model):
        qa = _qa(0)
        qa.answer = "Optical coherence tomography examines the retina in clinic."
        qa.answer_sentences = [background, "The retina is examined with optical coherence tomography in clinic."]
        return qa

    qs, _ = questions.generate_question_set(chunks, n=1, ask=ask, category_floors={}, category_caps={})
    assert qs[0].answer_sentences == ["The retina is examined with optical coherence tomography in clinic."]
