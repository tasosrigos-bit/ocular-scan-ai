"""A small Gemini client for generation, structured output, and judging.

Every part of the RAG layer that needs a language model shares this one thin
wrapper around the ``google-genai`` SDK: synthetic question generation, the answer
generator, and the reference-free LLM judge. Keeping it in one place means the
model, the API key handling, and the structured-output plumbing are defined once.

The model is pinned to a specific version for reproducibility rather than an alias
that drifts over time. The API key is read from the environment, which
:mod:`ocular.config` populates from the git-ignored ``.env`` file, so no key is
ever hard-coded.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Callable, TypeVar

from pydantic import BaseModel

T = TypeVar("T")

from ocular import config  # noqa: F401 - importing loads .env via config

# Pinned model: the newest small, cheap Gemini "lite" tier. Pinned (not the
# ``-latest`` alias) so an experiment run is reproducible.
MODEL = "gemini-3.5-flash-lite"

_CLIENT = None


def client():
    """Return a cached ``google-genai`` client, keyed on ``GEMINI_API_KEY``."""
    global _CLIENT
    if _CLIENT is None:
        from google import genai

        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set; add it to the .env file.")
        _CLIENT = genai.Client(api_key=key)
    return _CLIENT


def _with_retry(fn: Callable[[], T], retries: int = 6) -> T:
    """Call ``fn``, retrying on transient API errors, honouring rate-limit delays.

    The free tier caps requests per minute and returns HTTP 429 with a suggested
    ``retry in Ns`` delay, so every model call goes through this. On a 429 the
    server's suggested delay is respected (or a minute-clearing wait if none is
    given); other errors back off exponentially. The last failure re-raises.
    """
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            message = str(exc)
            if match := re.search(r"retry in ([\d.]+)s", message):
                delay = float(match.group(1)) + 1.0
            elif "429" in message or "RESOURCE_EXHAUSTED" in message:
                delay = 35.0  # clear the per-minute window
            else:
                delay = 2**attempt
            time.sleep(delay)


def generate(prompt: str, *, model: str = MODEL, temperature: float = 0.0, system: str | None = None) -> str:
    """Generate free-text from a prompt.

    Parameters
    ----------
    prompt : str
        The user prompt.
    model : str, optional
        Model id. Defaults to :data:`MODEL`.
    temperature : float, optional
        Sampling temperature. ``0`` for deterministic tasks (judging), higher for
        variety (question generation). Defaults to 0.
    system : str, optional
        A system instruction that frames the task.

    Returns
    -------
    str
        The model's text response.
    """
    from google.genai import types

    cfg = types.GenerateContentConfig(temperature=temperature, system_instruction=system)
    resp = _with_retry(lambda: client().models.generate_content(model=model, contents=prompt, config=cfg))
    return resp.text


def generate_structured(
    prompt: str,
    schema: type[BaseModel],
    *,
    model: str = MODEL,
    temperature: float = 0.0,
    system: str | None = None,
) -> BaseModel:
    """Generate a response constrained to a Pydantic schema.

    Gemini is asked to return JSON matching ``schema``; the response is parsed and
    validated into an instance of that model. This is how we get reliably shaped
    output (a question and answer, a judge's verdict) instead of free text we have
    to parse by hand.

    Parameters
    ----------
    prompt : str
        The user prompt.
    schema : type[pydantic.BaseModel]
        The expected response shape.
    model, temperature, system
        As in :func:`generate`.

    Returns
    -------
    pydantic.BaseModel
        An instance of ``schema``.
    """
    from google.genai import types

    cfg = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema,
    )
    resp = _with_retry(lambda: client().models.generate_content(model=model, contents=prompt, config=cfg))
    # The SDK parses the JSON for us when a Pydantic schema is given; fall back to
    # parsing the raw text if `parsed` is not populated.
    return resp.parsed or schema(**json.loads(resp.text))
