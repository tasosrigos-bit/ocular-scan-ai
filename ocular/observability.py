"""Tracing of the scan analysis and the assistant in MLflow.

Every question the assistant answers becomes one MLflow trace: a root span for the
question, a span for each round of the language model with its token counts, and a
span for each tool call, the literature search recording the passages it returned
with their scores. Every scan analysis is a trace of its own. The MLflow UI then
shows, per question, what the model was asked, which tools it called with which
arguments, what came back, and where the time went.

Tracing is on only when ``MLFLOW_TRACKING_URI`` is set, as it is in
``docker-compose.yml``. Without it, and when the server cannot be reached at
startup, :func:`span` hands back a no-op span, so the application, the notebooks
and the tests run unchanged and never wait on a tracing server.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager

log = logging.getLogger(__name__)

EXPERIMENT = "ocular-assistant"

_enabled = False
_prompt_versions: dict[str, str] = {}


def setup() -> bool:
    """Turn tracing on if a tracking server is configured and reachable.

    Returns
    -------
    bool
        Whether tracing is on.
    """
    global _enabled
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(EXPERIMENT)
    except Exception as exc:  # a tracing outage must not take the application down
        log.warning("MLflow tracing is off, %s is unreachable: %s", uri, exc)
        return False
    _enabled = True
    return True


class _NoSpan:
    """Stands in for an MLflow span when tracing is off."""

    def set_inputs(self, inputs) -> None:
        pass

    def set_outputs(self, outputs) -> None:
        pass

    def set_attributes(self, attributes) -> None:
        pass


def set_session(session_id: str | None) -> None:
    """Mark the current trace as part of a conversation, so MLflow groups its traces."""
    if _enabled and session_id:
        import mlflow

        mlflow.update_current_trace(session_id=session_id)


def prompt_version(name: str, text: str) -> str:
    """The Prompt Registry version of ``name`` whose text is ``text``.

    The prompt itself lives in the code, so the application never depends on the
    registry to run. Its versions are registered by ``scripts/register_monitoring.py``,
    and this looks up which one the running code holds, so each trace records it.
    A text not registered yet is ``"unregistered"``, and looked up again on the
    next call; ``"unknown"`` means the registry could not be asked.
    """
    if not _enabled:
        return "unknown"
    if text in _prompt_versions:
        return _prompt_versions[text]
    try:
        import mlflow

        versions = mlflow.MlflowClient().search_prompt_versions(name)
    except Exception as exc:  # a registry outage must not take the application down
        log.warning("Prompt version of %s unknown: %s", name, exc)
        return "unknown"
    match = next((str(v.version) for v in versions if v.template == text), None)
    if match is None:
        return "unregistered"
    _prompt_versions[text] = match
    return match


def tag_trace(tags: dict[str, str]) -> None:
    """Tag the current trace, for example so a judge runs only on some traces."""
    if _enabled:
        import mlflow

        mlflow.update_current_trace(tags=tags)


@contextmanager
def span(name: str, span_type: str = "UNKNOWN", inputs=None):
    """Open a traced span, nested under the current one if there is one.

    Parameters
    ----------
    name : str
        The span name shown in the MLflow UI.
    span_type : str, optional
        An MLflow span type such as ``"CHAT_MODEL"``, ``"RETRIEVER"`` or ``"TOOL"``.
    inputs : optional
        Recorded as the span's inputs.

    Yields
    ------
    The live span, or a no-op stand-in when tracing is off. Either way it offers
    ``set_inputs``, ``set_outputs`` and ``set_attributes``.
    """
    if not _enabled:
        yield _NoSpan()
        return
    import mlflow

    with mlflow.start_span(name=name, span_type=span_type) as live:
        if inputs is not None:
            live.set_inputs(inputs)
        yield live
