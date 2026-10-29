"""The FastAPI service, with the classifier and the assistant replaced by stubs.

The client is created without its lifespan, so the real checkpoint and index are
never loaded.
"""
from __future__ import annotations

import json
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import api
from ocular import service

PROBS = {"CNV": 0.02, "DME": 0.95, "DRUSEN": 0.02, "NORMAL": 0.01}
SOURCE = {"title": "Article A", "section": "Results", "doi": None, "license": "CC BY",
          "score": 0.93, "text": "drusen deposits"}


@pytest.fixture
def analysed(monkeypatch):
    """The paths the stub analysis was given, to check the scan is not kept."""
    paths = []

    def analyse(path, **kwargs):
        assert path.exists()
        paths.append(path)
        return {"label": "DME", "confidence": 0.95, "probs": PROBS,
                "scan": np.zeros((256, 384)), "overlay": np.zeros((256, 384, 3))}

    monkeypatch.setattr(service, "analyse", analyse)
    return paths


@pytest.fixture
def client(analysed):
    return TestClient(api.app)


def _upload(client, name="scan.png"):
    return client.post("/api/predict", files={"file": (name, b"not really a png", "image/png")})


def _chat(client, path="/api/chat", **body):
    body.setdefault("messages", [{"role": "user", "content": "How is DME managed?"}])
    return client.post(path, json=body)


def test_health_and_the_page_are_served(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200 and "Ocular" in page.text


def test_predict_returns_the_prediction_and_both_images(client, analysed):
    res = _upload(client)
    assert res.status_code == 200
    data = res.json()
    assert set(data) == {"label", "confidence", "probs", "model_input", "gradcam"}
    assert (data["label"], data["confidence"], data["probs"]) == ("DME", 0.95, PROBS)
    assert data["model_input"].startswith("data:image/png;base64,")
    assert data["gradcam"].startswith("data:image/png;base64,")
    # The scan is analysed from a temporary file that is gone once it has answered.
    assert not analysed[0].exists()


def test_predict_rejects_other_file_types(client):
    assert _upload(client, "notes.txt").status_code == 415


def test_predict_reports_and_cleans_up_an_unreadable_scan(client, monkeypatch):
    paths = []

    def broken(path, **kwargs):
        paths.append(path)
        raise OSError("cannot identify image file")

    monkeypatch.setattr(service, "analyse", broken)
    res = _upload(client)
    assert res.status_code == 422
    assert not paths[0].exists()


def test_chat_sends_the_conversation_only(client, monkeypatch):
    # The assistant answers general questions, so a scan or patient details sent
    # by an older page are ignored rather than passed on.
    seen = {}

    def reply(messages, **kwargs):
        seen.update(messages=messages, kwargs=kwargs)
        return "DME is treated with anti-VEGF injections [1].", [SOURCE]

    monkeypatch.setattr(service, "reply", reply)
    res = _chat(client, scan_id=uuid.uuid4().hex, patient={"age": 74})
    assert res.json() == {"reply": "DME is treated with anti-VEGF injections [1].", "sources": [SOURCE]}
    assert seen["messages"] == [{"role": "user", "content": "How is DME managed?"}]
    assert seen["kwargs"] == {"session_id": None}


def test_chat_passes_the_session_to_the_assistant(client, monkeypatch):
    seen = {}

    def reply(messages, **kwargs):
        seen.update(kwargs)
        return "Drusen are deposits [1].", [SOURCE]

    monkeypatch.setattr(service, "reply", reply)
    session = uuid.uuid4()
    _chat(client, session_id=str(session))
    assert seen["session_id"] == session.hex


def test_predict_passes_the_session_to_the_analysis(client, monkeypatch):
    seen = {}

    def analyse(path, **kwargs):
        seen.update(kwargs)
        raise OSError("stop after the call")

    monkeypatch.setattr(service, "analyse", analyse)
    session = uuid.uuid4()
    client.post("/api/predict", files={"file": ("scan.png", b"x", "image/png")},
                data={"session_id": str(session)})
    assert seen["session_id"] == session.hex


def test_the_page_is_revalidated_on_each_load(client):
    assert client.get("/app.js").headers["cache-control"] == "no-cache"
    assert "cache-control" not in client.get("/api/health").headers


def test_chat_rejects_a_malformed_session_id(client):
    assert _chat(client, session_id="not-a-session").status_code == 422


def test_chat_needs_a_question_last(client):
    res = _chat(client, messages=[{"role": "assistant", "content": "Hello."}])
    assert res.status_code == 422


def test_chat_turns_a_model_failure_into_a_readable_error(client, monkeypatch):
    def reply(*args, **kwargs):
        raise PermissionError("API key not valid")

    monkeypatch.setattr(service, "reply", reply)
    res = _chat(client)
    assert res.status_code == 502
    assert res.json()["detail"] == "The language model could not answer (PermissionError)."


def _events(res):
    return [json.loads(line) for line in res.text.splitlines() if line]


def test_stream_sends_steps_then_text_then_the_answer(client, monkeypatch):
    def reply(messages, on_status, on_text, session_id=None):
        on_status("Reading the question")
        on_status("Searching the literature for “DME”")
        on_text("DME is treated ")
        on_text("with anti-VEGF [1].")
        return "DME is treated with anti-VEGF [1].", [SOURCE]

    monkeypatch.setattr(service, "reply", reply)
    res = _chat(client, path="/api/chat/stream")
    assert res.headers["content-type"].startswith("application/x-ndjson")
    assert _events(res) == [
        {"type": "status", "text": "Reading the question"},
        {"type": "status", "text": "Searching the literature for “DME”"},
        {"type": "text", "text": "DME is treated "},
        {"type": "text", "text": "with anti-VEGF [1]."},
        {"type": "answer", "reply": "DME is treated with anti-VEGF [1].", "sources": [SOURCE]},
    ]


def test_stream_ends_with_an_error_event_when_the_model_fails(client, monkeypatch):
    def reply(messages, on_status, on_text, session_id=None):
        on_status("Reading the question")
        raise TimeoutError

    monkeypatch.setattr(service, "reply", reply)
    events = _events(_chat(client, path="/api/chat/stream"))
    assert events[-1] == {"type": "error", "detail": "The language model could not answer (TimeoutError)."}
