"""On-disk cache for generated narratives — keyed so the same case yields the
same narrative across runs (temperature 0 + this cache = fully reproducible)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from oct_cds.common.paths import REPO_ROOT
from oct_cds.cds.schema import Recommendation

PROMPT_VERSION = 3   # bump to invalidate the on-disk cache (prompt / capture changes)
_DEFAULT_DIR = REPO_ROOT / "knowledge_base" / ".cache"


def cache_key(rec: Recommendation, *, kb_version: int, model: str, retrieved_ids: list[str]) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "kb_version": kb_version,
        "model": model,
        "predicted_class": rec.predicted_class,
        "abstained": rec.abstained,
        "ood_rejected": rec.ood_rejected,
        "urgency": rec.urgency.value,
        "confidence": round(rec.confidence, 3),
        "differential": [(d["class"], round(float(d["probability"]), 3))
                         for d in (rec.differential or [])],
        "retrieved_ids": sorted(retrieved_ids),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:24]


class NarrativeCache:
    def __init__(self, dir_: str | Path | None = None, enabled: bool = True):
        self.dir = Path(dir_) if dir_ else _DEFAULT_DIR
        self.enabled = enabled

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        f = self.dir / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        return None

    def put(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{key}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
