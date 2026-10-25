"""Tracing: off by default, and when on, one span per question, model round and tool.

A recording stand-in replaces the ``mlflow`` module, so the tests check which spans
the service opens and what they record, without a tracking server.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessageChunk

from ocular import observability, service
from tests.test_service import ScriptedChat, _tool_call


class RecordingSpan:
    def __init__(self, name, span_type):
        self.name, self.span_type = name, span_type
        self.inputs = self.outputs = None
        self.attributes = {}

    def set_inputs(self, inputs):
        self.inputs = inputs

    def set_outputs(self, outputs):
        self.outputs = outputs

    def set_attributes(self, attributes):
        self.attributes.update(attributes)


@pytest.fixture
def recorded(monkeypatch):
    spans = []

    @contextmanager
    def start_span(name, span_type):
        spans.append(RecordingSpan(name, span_type))
        yield spans[-1]

    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(start_span=start_span))
    monkeypatch.setattr(observability, "_enabled", True)
    return spans


def test_tracing_is_off_without_a_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert observability.setup() is False
    with observability.span("anything") as live:
        live.set_outputs("ignored")  # the no-op span accepts the same calls


def test_an_unreachable_server_turns_tracing_off_instead_of_failing(monkeypatch):
    def unreachable(name):
        raise ConnectionError("connection refused")

    fake = SimpleNamespace(set_tracking_uri=lambda uri: None, set_experiment=unreachable)
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setattr(observability, "_enabled", False)
    assert observability.setup() is False


def test_a_question_is_traced_round_by_round(recorded, monkeypatch, tiny_index):
    monkeypatch.setattr(service, "load_index", lambda: tiny_index)
    monkeypatch.setattr("ocular.rag.rerank.DEFAULT_RERANKER", None)

    def rounds(n):
        if n == 1:
            return [_tool_call("search_corpus", {"query": "drusen deposits"}, "c1")]
        answer = AIMessageChunk(content="Drusen are deposits [1].")
        answer.usage_metadata = {"input_tokens": 120, "output_tokens": 8, "total_tokens": 128}
        return [answer]

    monkeypatch.setattr(service, "chat_model", lambda: ScriptedChat(rounds))
    service.reply([{"role": "user", "content": "What are drusen?"}])

    assert [(s.name, s.span_type) for s in recorded] == [
        ("answer_question", "AGENT"),
        (f"{service.llm.MODEL} round 1", "CHAT_MODEL"),
        ("search_corpus", "RETRIEVER"),
        (f"{service.llm.MODEL} round 2", "CHAT_MODEL"),
    ]
    question, first_round, search, second_round = recorded
    assert question.inputs["question"] == "What are drusen?"
    assert question.outputs == {"answer": "Drusen are deposits [1].", "passages": 5}
    assert first_round.outputs["tool_calls"] == [
        {"name": "search_corpus", "args": {"query": "drusen deposits"}}
    ]
    assert search.inputs == {"query": "drusen deposits"}
    assert search.outputs[0]["id"] == "Article A - Results"
    assert search.outputs[0]["page_content"] == "drusen deposits drusen deposits"
    assert second_round.attributes == {
        "mlflow.llm.model": service.llm.MODEL,
        "mlflow.llm.provider": "gemini",
        "mlflow.chat.tokenUsage": {"input_tokens": 120, "output_tokens": 8, "total_tokens": 128},
    }
