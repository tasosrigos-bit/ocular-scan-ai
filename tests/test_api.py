"""The FastAPI service, with the classifier and the assistant replaced by stubs.

The client is created without its lifespan, so the real checkpoint and index are
never loaded. Uploads go to a temporary directory per test.
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
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOADS", tmp_path)
    monkeypatch.setattr(service, "analyse", lambda path: {
        "label": "DME", "confidence": 0.95, "probs": PROBS,
        "scan": np.zeros((256, 384)), "overlay": np.zeros((256, 384, 3)),
    })
    return TestClient(api.app)


def _upload(client, name="scan.png"):
    return client.post("/api/predict", files={"file": (name, b"not really a png", "image/png")})


def _chat(client, path="/api/chat", **body):
    body.setdefault("messages", [{"role": "user", "content": "What does this scan show?"}])
    return client.post(path, json=body)


def test_health_and_the_page_are_served(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200 and "Ocular" in page.text


def test_predict_returns_the_prediction_and_both_images(client, tmp_path):
    res = _upload(client)
    assert res.status_code == 200
    data = res.json()
    assert (data["label"], data["confidence"], data["probs"]) == ("DME", 0.95, PROBS)
    assert data["model_input"].startswith("data:image/png;base64,")
    assert data["gradcam"].startswith("data:image/png;base64,")
    assert (tmp_path / f"{data['scan_id']}.png").exists()


def test_predict_rejects_other_file_types(client):
    assert _upload(client, "notes.txt").status_code == 415


def test_predict_reports_and_cleans_up_an_unreadable_scan(client, monkeypatch, tmp_path):
    def broken(path):
        raise OSError("cannot identify image file")

    monkeypatch.setattr(service, "analyse", broken)
    res = _upload(client)
    assert res.status_code == 422
    assert list(tmp_path.iterdir()) == []


def test_chat_passes_the_uploaded_scan_to_the_assistant(client, monkeypatch, tmp_path):
    seen = {}

    def reply(messages, path, **kwargs):
        seen.update(messages=messages, path=path)
        return "It shows DME [1].", [SOURCE]

    monkeypatch.setattr(service, "reply", reply)
    scan_id = _upload(client).json()["scan_id"]
    res = _chat(client, scan_id=scan_id)
    assert res.json() == {"reply": "It shows DME [1].", "sources": [SOURCE]}
    assert seen["path"] == str(tmp_path / f"{scan_id}.png")
    assert seen["messages"] == [{"role": "user", "content": "What does this scan show?"}]


@pytest.mark.parametrize("scan_id, status", [
    ("../../etc/passwd", 422),     # not a UUID, so it never reaches the filesystem
    (uuid.uuid4().hex, 404),       # well formed, but nothing was uploaded under it
])
def test_chat_rejects_bad_scan_ids(client, scan_id, status):
    assert _chat(client, scan_id=scan_id).status_code == status


def test_chat_passes_the_session_to_the_assistant(client, monkeypatch):
    seen = {}

    def reply(messages, path, **kwargs):
        seen.update(kwargs)
        return "Drusen are deposits [1].", [SOURCE]

    monkeypatch.setattr(service, "reply", reply)
    session = uuid.uuid4()
    _chat(client, session_id=str(session))
    assert seen["session_id"] == session.hex


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
    def reply(messages, path, on_status, on_text, session_id=None):
        on_status("Reading the question")
        on_status("Searching the literature for “DME”")
        on_text("It shows ")
        on_text("DME [1].")
        return "It shows DME [1].", [SOURCE]

    monkeypatch.setattr(service, "reply", reply)
    res = _chat(client, path="/api/chat/stream")
    assert res.headers["content-type"].startswith("application/x-ndjson")
    assert _events(res) == [
        {"type": "status", "text": "Reading the question"},
        {"type": "status", "text": "Searching the literature for “DME”"},
        {"type": "text", "text": "It shows "},
        {"type": "text", "text": "DME [1]."},
        {"type": "answer", "reply": "It shows DME [1].", "sources": [SOURCE]},
    ]


def test_stream_ends_with_an_error_event_when_the_model_fails(client, monkeypatch):
    def reply(messages, path, on_status, on_text, session_id=None):
        on_status("Reading the question")
        raise TimeoutError

    monkeypatch.setattr(service, "reply", reply)
    events = _events(_chat(client, path="/api/chat/stream"))
    assert events[-1] == {"type": "error", "detail": "The language model could not answer (TimeoutError)."}
