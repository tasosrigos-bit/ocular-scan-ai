"""The service layer: helpers, and the assistant's tool loop driven by a scripted model."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from langchain_core.messages import AIMessageChunk
from PIL import Image

from ocular import service


def test_text_flattens_block_content():
    assert service.text("plain") == "plain"
    blocks = [{"type": "text", "text": "Drusen "}, {"type": "image"}, {"type": "text", "text": "[1]."}]
    assert service.text(blocks) == "Drusen [1]."


def test_to_png_round_trips_grey_and_colour_images():
    grey = np.linspace(0, 1, 6 * 4).reshape(6, 4)
    colour = np.zeros((6, 4, 3))
    for image, mode in ((grey, "L"), (colour, "RGB")):
        decoded = Image.open(io.BytesIO(service.to_png(image)))
        assert decoded.size == (4, 6) and decoded.mode == mode


def test_sources_of_keeps_what_the_page_shows(tiny_index):
    from ocular.rag.retrieve import Hit

    sources = service.sources_of([Hit(tiny_index.chunks[0], 0.9, 1)])
    assert sources == [{
        "title": "Article A", "section": "Results", "doi": None, "license": "CC BY",
        "score": 0.9, "text": "drusen deposits drusen deposits",
    }]


class ScriptedChat:
    """A stand-in chat model that replays one scripted list of chunks per round."""

    def __init__(self, rounds):
        self.rounds = rounds
        self.calls = 0
        self.inputs = []  # the conversation the model was given, per round
        self.choices = []  # the tool_choice each round was bound with

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        return _Bound(self, tool_choice)

    def stream(self, convo, tool_choice=None):
        self.calls += 1
        self.inputs.append(list(convo))
        self.choices.append(tool_choice)
        yield from self.rounds(self.calls)


class _Bound:
    def __init__(self, chat, tool_choice):
        self.chat, self.tool_choice = chat, tool_choice

    def stream(self, convo):
        return self.chat.stream(convo, self.tool_choice)


def _tool_call(name, args, call_id):
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": name, "args": json.dumps(args), "id": call_id, "index": 0}],
    )


@pytest.fixture
def offline(monkeypatch, tiny_index):
    """Point the service at the tiny index, without the reranker or the real model."""
    monkeypatch.setattr(service, "load_index", lambda: tiny_index)
    monkeypatch.setattr("ocular.rag.rerank.DEFAULT_RERANKER", None)

    def use(chat):
        monkeypatch.setattr(service, "chat_model", lambda: chat)
        return chat

    return use


def test_reply_searches_then_streams_a_grounded_answer(offline):
    def rounds(n):
        if n == 1:
            return [_tool_call("search_corpus", {"query": "drusen deposits"}, "c1")]
        return [AIMessageChunk(content="Drusen are "), AIMessageChunk(content="deposits [1].")]

    chat = offline(ScriptedChat(rounds))
    statuses, pieces = [], []
    answer, sources = service.reply(
        [{"role": "user", "content": "What are drusen?"}],
        on_status=statuses.append,
        on_text=pieces.append,
    )
    assert answer == "Drusen are deposits [1]."
    assert "".join(pieces) == answer
    assert statuses == [
        "Reading the question",
        "Searching the literature for “drusen deposits”",
        "Reviewing the results",
    ]
    assert chat.calls == 2
    assert sources[0]["title"] == "Article A" and len(sources) == 5


def test_reply_searches_once_per_question(offline, monkeypatch):
    # Basic RAG: the model's query is searched, and a second search for the same
    # question is declined.
    def rounds(n):
        if n < 3:
            return [_tool_call("search_corpus", {"query": f"model query {n}"}, f"c{n}")]
        return [AIMessageChunk(content="Drusen are deposits [1].")]

    chat = offline(ScriptedChat(rounds))
    queries = []
    real = service.tools.retrieve.retrieve
    monkeypatch.setattr("ocular.rag.retrieve.retrieve",
                        lambda idx, query, **kw: queries.append(query) or real(idx, query, **kw))
    _, sources = service.reply([{"role": "user", "content": "What are drusen?"}])
    assert queries == ["model query 1"]
    assert len(sources) == 5
    assert "already been searched" in chat.inputs[-1][-1].content


def test_reply_stops_after_the_round_cap(offline):
    # A model that only ever asks for a tool must still end, at MAX_TOOL_ROUNDS.
    chat = offline(ScriptedChat(lambda n: [_tool_call("classify_scan", {}, f"c{n}")]))
    service.reply([{"role": "user", "content": "What does this scan show?"}])
    assert chat.calls == service.MAX_TOOL_ROUNDS


def test_the_assistant_is_a_model_and_tools_graph(offline):
    offline(ScriptedChat(lambda n: [AIMessageChunk(content="ok")]))
    graph = service.build_assistant({}, []).get_graph()
    assert {n for n in graph.nodes} == {"__start__", "model", "tools", "__end__"}
    edges = {(e.source, e.target) for e in graph.edges}
    assert {("__start__", "model"), ("model", "tools"), ("model", "__end__"), ("tools", "model")} <= edges


def test_a_failing_tool_becomes_a_message_instead_of_an_error(offline, monkeypatch):
    def rounds(n):
        if n == 1:
            return [_tool_call("search_corpus", {"query": "drusen"}, "c1")]
        return [AIMessageChunk(content="The literature search failed, so I cannot cite sources.")]

    chat = offline(ScriptedChat(rounds))

    def broken(*args, **kwargs):
        raise ConnectionError("Qdrant unreachable")

    monkeypatch.setattr("ocular.rag.retrieve.retrieve", broken)
    answer, sources = service.reply([{"role": "user", "content": "What are drusen?"}])
    assert answer.startswith("The literature search failed")
    assert sources == []
    tool_message = chat.inputs[-1][-1]  # what the model saw before its second round
    assert "search_corpus tool failed (ConnectionError: Qdrant unreachable)" in tool_message.content


def test_a_call_to_an_unknown_tool_is_reported_back_to_the_model(offline):
    def rounds(n):
        if n == 1:
            return [_tool_call("delete_corpus", {}, "c1")]
        return [AIMessageChunk(content="I can only search and classify.")]

    offline(ScriptedChat(rounds))
    answer, _ = service.reply([{"role": "user", "content": "Delete everything."}])
    assert answer == "I can only search and classify."


def test_describe_patient_writes_the_filled_fields_only():
    assert service.describe_patient(None) == ""
    assert service.describe_patient({"symptoms": [], "notes": ""}) == ""
    line = service.describe_patient({
        "age": 74, "eye": "left", "diabetes": "no", "hba1c": 5.4, "onset": "sudden",
        "duration": "2 weeks", "symptoms": ["distortion", "blurred vision"], "notes": None,
    })
    assert line == ("Patient details entered by the doctor: age 74; left eye; no diabetes; "
                    "HbA1c 5.4%; sudden onset; for 2 weeks; symptoms: distortion, blurred vision.")


def test_reply_puts_the_patient_details_before_the_question(offline):
    chat = offline(ScriptedChat(lambda n: [AIMessageChunk(content="ok")]))
    service.reply([{"role": "user", "content": "What next?"}], patient={"age": 60, "eye": "both"})
    assert chat.inputs[0][-1].content == (
        "Patient details entered by the doctor: age 60; both eyes.\n\nWhat next?")


def test_an_uploaded_scan_is_classified_first(offline, tmp_path):
    def rounds(n):
        if n == 1:
            return [_tool_call("classify_scan", {}, "c1")]
        return [AIMessageChunk(content="ok")]

    chat = offline(ScriptedChat(rounds))
    service.reply([{"role": "user", "content": "Routine check."}], scan_path=tmp_path / "missing.png")
    assert chat.choices[0] == "classify_scan" and chat.choices[1] is None
    chat = offline(ScriptedChat(lambda n: [AIMessageChunk(content="ok")]))
    service.reply([{"role": "user", "content": "What are drusen?"}])
    assert chat.choices == [None]


def test_remove_doses_takes_out_the_sentences_with_an_amount():
    answer = ("Ranibizumab is given monthly at first [1]. The maximum tolerated dose was 500 μg [2].\n\n"
              "### Recommended actions\n1. Give 0.5 mg in 0.05 mL.\n2. Refer to a retina specialist [3].")
    cleaned, removed = service.remove_doses(answer)
    assert removed
    assert cleaned == ("Ranibizumab is given monthly at first [1].\n\n" + service.DOSE_NOTE + "\n\n"
                       "### Recommended actions\n2. Refer to a retina specialist [3].")
    plain = "IOP was 28 mmHg, HbA1c 9% and the classifier's output is CNV (63%).\n\n1. Refer today."
    assert service.remove_doses(plain) == (plain, False)
    assert service.remove_doses("Use 1.25 mg off-label.")[0] == service.DOSE_NOTE


def test_the_streamed_text_never_shows_a_dose(offline):
    pieces = ["Aflibercept is given ", "every 8 weeks [1]. The dose is ", "2 mg in 0.05 ",
              "mL [2]. Take it from the label."]
    offline(ScriptedChat(lambda n: [AIMessageChunk(content=p) for p in pieces]))
    shown = []
    answer, _ = service.reply([{"role": "user", "content": "Aflibercept dose?"}], on_text=shown.append)
    assert "".join(shown) == "Aflibercept is given every 8 weeks [1]. Take it from the label."
    assert "2 mg" not in answer and answer.endswith(service.DOSE_NOTE)
