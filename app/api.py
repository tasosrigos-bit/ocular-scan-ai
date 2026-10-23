"""FastAPI service: the classifier and the grounded assistant over HTTP.

The service exposes the scan analysis and the assistant in :mod:`ocular.service`,
and serves the single-page web front-end in ``app/web`` from the same process, so
there is one server to start and no cross-origin setup.

``POST /api/predict`` takes an uploaded B-scan, classifies it and returns the
prediction with the model input and its Grad-CAM overlay as PNG data URLs, plus a
``scan_id``. ``POST /api/chat`` takes the conversation, kept on the client, and the
``scan_id`` of the scan it is about, and returns the grounded answer with the
passages it cites. ``POST /api/chat/stream`` takes the same request and streams the
assistant's progress and the answer's text as they are produced, one JSON object
per line, ending with the full answer and its sources.
The service holds no conversation state, only the uploaded scans, so the assistant's
``classify_scan`` tool can reach the scan a question refers to.

Run with::

    uv run uvicorn app.api:app --reload
"""
from __future__ import annotations

import base64
import json
import queue
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ocular import observability, service
from ocular.rag import tools

WEB = Path(__file__).parent / "web"
UPLOADS = Path(tempfile.gettempdir()) / "ocular_uploads"
SUFFIXES = {".png", ".jpg", ".jpeg"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the classifier and the index at startup, so no request pays for it.

    Tracing to MLflow starts here too when ``MLFLOW_TRACKING_URI`` is set.
    """
    observability.setup()
    UPLOADS.mkdir(exist_ok=True)
    tools.load_classifier()
    service.load_index()
    yield


app = FastAPI(title="Ocular Scan AI", lifespan=lifespan)


class Prediction(BaseModel):
    scan_id: str
    label: str
    confidence: float
    probs: dict[str, float]
    model_input: str
    gradcam: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    scan_id: str | None = None


class Source(BaseModel):
    title: str
    section: str | None = None
    doi: str | None = None
    license: str | None = None
    score: float
    text: str


class ChatResponse(BaseModel):
    reply: str
    sources: list[Source]


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _scan_path(scan_id: str | None) -> Path | None:
    """Resolve a ``scan_id`` to its uploaded file, or ``None`` if there is none.

    The id is parsed as a UUID before it touches the filesystem, so a crafted id
    cannot reach a path outside the upload directory.
    """
    if not scan_id:
        return None
    try:
        key = uuid.UUID(scan_id).hex
    except ValueError:
        raise HTTPException(422, "Malformed scan_id.")
    for path in UPLOADS.glob(f"{key}.*"):
        return path
    raise HTTPException(404, "Unknown scan_id; upload the scan again.")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# The endpoints are plain functions rather than coroutines, because the model and
# the language model calls block; FastAPI runs such endpoints in a thread pool.
@app.post("/api/predict", response_model=Prediction)
def predict(file: UploadFile) -> Prediction:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUFFIXES:
        raise HTTPException(415, "Upload a PNG or JPEG B-scan.")
    scan_id = uuid.uuid4().hex
    path = UPLOADS / f"{scan_id}{suffix}"
    path.write_bytes(file.file.read())
    try:
        result = service.analyse(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise HTTPException(422, "The file could not be read as an OCT B-scan.")
    return Prediction(
        scan_id=scan_id,
        label=result["label"],
        confidence=result["confidence"],
        probs=result["probs"],
        model_input=_data_url(service.to_png(result["scan"])),
        gradcam=_data_url(service.to_png(result["overlay"])),
    )


def _chat_args(req: ChatRequest) -> tuple[list[dict], str | None]:
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(422, "The conversation must end with a user message.")
    path = _scan_path(req.scan_id)
    return [m.model_dump() for m in req.messages], str(path) if path else None


def _failure(exc: Exception) -> str:
    # A failed language model call (a bad key, a rate limit, no network) reaches the
    # page as a readable message rather than a bare server error.
    return f"The language model could not answer ({type(exc).__name__})."


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    messages, path = _chat_args(req)
    try:
        answer, sources = service.reply(messages, path)
    except Exception as exc:
        raise HTTPException(502, _failure(exc))
    return ChatResponse(reply=answer, sources=sources)


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream the assistant's progress as newline-delimited JSON.

    Each line is one event: ``{"type": "status", "text": ...}`` as a step starts,
    ``{"type": "text", "text": ...}`` for each piece of text the model writes, and
    finally either ``{"type": "answer", "reply": ..., "sources": [...]}`` or
    ``{"type": "error", "detail": ...}``. The answer is produced on a worker thread
    that reports its steps through a queue, which the response drains as it goes.
    """
    messages, path = _chat_args(req)
    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            answer, sources = service.reply(
                messages,
                path,
                on_status=lambda text: events.put({"type": "status", "text": text}),
                on_text=lambda text: events.put({"type": "text", "text": text}),
            )
            events.put({"type": "answer", "reply": answer, "sources": sources})
        except Exception as exc:
            events.put({"type": "error", "detail": _failure(exc)})
        finally:
            events.put(None)

    def lines():
        threading.Thread(target=work, daemon=True).start()
        while (event := events.get()) is not None:
            yield json.dumps(event) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


# Mounted last, so the API routes above take precedence over the static files.
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
