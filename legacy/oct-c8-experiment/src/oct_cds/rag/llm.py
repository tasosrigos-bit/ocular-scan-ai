"""LLM backends for the narrator. All share one interface, so swapping local <->
API is a config change (`rag.llm.backend`), not a code change.

  - stub       deterministic, no model — for tests and offline dry-runs
  - hf_local   Qwen2.5-3B-Instruct (default) via transformers, temperature 0
  - anthropic  Claude via the anthropic SDK, API key from ANTHROPIC_API_KEY
"""

from __future__ import annotations

import os
import re
from typing import Any, Protocol


class LLMBackend(Protocol):
    name: str

    def generate(self, system: str, user: str, *, max_tokens: int, temperature: float) -> str: ...


# ---------------------------------------------------------------------------
class StubBackend:
    """No model. Emits a minimal grounded narrative parsed from the prompt:
    the fixed impression + the first cited passage id. Enough to exercise the
    retrieval / verify / report path deterministically."""

    name = "stub"

    def generate(self, system: str, user: str, *, max_tokens: int, temperature: float) -> str:
        impression = "the predicted finding"
        m = re.search(r"impression:\s*([A-Za-z][A-Za-z ]*)", user)
        if m:
            impression = m.group(1).strip()
        ids = re.findall(r"\[([a-z0-9][a-z0-9#\-]*)\]", user)
        cite = f" [{ids[0]}]" if ids else ""
        note = ""
        if "abstained" in user.lower() and "abstained: true" in user.lower():
            impression = "an uncertain finding; specialist review is appropriate"
        return (
            f"The macular OCT was classified as {impression}.{cite} "
            f"This explanation is generated from the retrieved reference passages "
            f"and does not alter the impression or triage above."
        )


# ---------------------------------------------------------------------------
class HFLocalBackend:
    name = "hf_local"

    def __init__(self, model: str = "Qwen/Qwen2.5-3B-Instruct", device: str | None = None,
                 dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model
        self._tok = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=("auto" if dtype == "auto" else getattr(torch, dtype)),
            device_map=(device or ("cuda" if torch.cuda.is_available() else "cpu")),
        )
        self._model.eval()

    def generate(self, system: str, user: str, *, max_tokens: int, temperature: float) -> str:
        import torch

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", api_key_env: str = "ANTHROPIC_API_KEY"):
        import anthropic

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} not set")
        self.model_id = model
        self._client = anthropic.Anthropic(api_key=key)

    def generate(self, system: str, user: str, *, max_tokens: int, temperature: float) -> str:
        resp = self._client.messages.create(
            model=self.model_id,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


# ---------------------------------------------------------------------------
def make_backend(cfg: dict[str, Any]) -> LLMBackend:
    kind = (cfg or {}).get("backend", "hf_local")
    if kind == "stub":
        return StubBackend()
    if kind == "hf_local":
        return HFLocalBackend(
            model=cfg.get("model", "Qwen/Qwen2.5-3B-Instruct"),
            device=cfg.get("device"),
            dtype=cfg.get("dtype", "auto"),
        )
    if kind == "anthropic":
        model = cfg.get("model", "")
        if not model or "/" in model:      # a HF-style id left over from hf_local
            model = cfg.get("anthropic_model", "claude-sonnet-5")
        return AnthropicBackend(model=model, api_key_env=cfg.get("api_key_env", "ANTHROPIC_API_KEY"))
    raise ValueError(f"unknown rag.llm.backend: {kind!r}")
