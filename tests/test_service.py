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

    def bind_tools(self, tools, **kwargs):
        return self

    def stream(self, convo):
        self.calls += 1
        yield from self.rounds(self.calls)


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


def test_reply_stops_after_the_round_cap(offline):
    # A model that only ever asks for a tool must still end, at MAX_TOOL_ROUNDS.
    chat = offline(ScriptedChat(lambda n: [_tool_call("classify_scan", {}, f"c{n}")]))
    service.reply([{"role": "user", "content": "What does this scan show?"}])
    assert chat.calls == service.MAX_TOOL_ROUNDS
