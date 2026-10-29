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
        self.tags = {}

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

    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(
        start_span=start_span, update_current_trace=lambda **kw: spans[0].tags.update(kw.get("tags", {}))))
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
    assert question.inputs["earlier_turns"] == []
    assert question.outputs["answer"] == "Drusen are deposits [1]."
    assert question.outputs["passages"].startswith("[1] (Article A - Results)")
    assert question.tags == {"searched": "true", "doses_removed": "false", "prompt_version": "unknown"}
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


def _registry(monkeypatch, versions):
    calls = []

    class Client:
        def search_prompt_versions(self, name):
            calls.append(name)
            if versions is None:
                raise ConnectionError("registry unreachable")
            return [SimpleNamespace(version=v, template=t) for v, t in versions]

    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(MlflowClient=Client))
    monkeypatch.setattr(observability, "_enabled", True)
    monkeypatch.setattr(observability, "_prompt_versions", {})
    return calls


def test_the_prompt_version_is_the_registered_one_with_the_same_text(monkeypatch):
    calls = _registry(monkeypatch, [(1, "old prompt"), (2, "current prompt")])
    assert observability.prompt_version("p", "current prompt") == "2"
    assert observability.prompt_version("p", "current prompt") == "2"
    assert calls == ["p"]  # found once, then remembered


def test_an_edited_prompt_is_unregistered_until_registered(monkeypatch):
    calls = _registry(monkeypatch, [(1, "old prompt")])
    assert observability.prompt_version("p", "edited prompt") == "unregistered"
    assert observability.prompt_version("p", "edited prompt") == "unregistered"
    assert len(calls) == 2  # asked again, so a later registration is picked up


def test_a_registry_outage_leaves_the_prompt_version_unknown(monkeypatch):
    _registry(monkeypatch, None)
    assert observability.prompt_version("p", "current prompt") == "unknown"
