from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "metadata" / "label_map.json"
)


@dataclass(frozen=True)
class LabelMap:
    """Frozen mapping between canonical class keys, integer ids and on-disk dirs."""

    num_classes: int
    key_to_id: dict[str, int]
    id_to_key: dict[int, str]
    dir_aliases: dict[str, str]           # folder name (any casing) -> canonical key
    group_by_key: dict[str, str]          # canonical key -> CDS group tag

    def id(self, key: str) -> int:
        return self.key_to_id[key]

    def key(self, id_: int) -> str:
        return self.id_to_key[int(id_)]

    def group(self, key: str) -> str:
        return self.group_by_key[key]

    def normalize_dir(self, folder: str) -> str:
        """Map an on-disk class folder name to a canonical key (case-insensitive)."""
        f = folder.strip()
        if f in self.dir_aliases:
            return self.dir_aliases[f]
        lowered = {k.lower(): v for k, v in self.dir_aliases.items()}
        if f.lower() in lowered:
            return lowered[f.lower()]
        # also accept the canonical key itself
        for key in self.key_to_id:
            if f.lower() == key.lower() or f.lower() == key.replace(" ", "").lower():
                return key
        raise KeyError(f"Unknown class folder / token: {folder!r}")

    @property
    def keys(self) -> list[str]:
        return [self.id_to_key[i] for i in range(self.num_classes)]


@lru_cache(maxsize=4)
def load_label_map(path: str | Path | None = None) -> LabelMap:
    p = Path(path) if path is not None else _DEFAULT_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))

    key_to_id = {k: int(v) for k, v in raw["key_to_id"].items()}
    id_to_key = {int(k): v for k, v in raw["id_to_key"].items()}
    dir_aliases = dict(raw.get("dir_aliases", {}))
    group_by_key = {c["key"]: c["group"] for c in raw["classes"]}

    n = int(raw["num_classes"])
    assert len(key_to_id) == n == len(id_to_key), "label_map.json is internally inconsistent"
    assert sorted(id_to_key) == list(range(n)), "class ids must be contiguous 0..n-1"

    return LabelMap(
        num_classes=n,
        key_to_id=key_to_id,
        id_to_key=id_to_key,
        dir_aliases=dir_aliases,
        group_by_key=group_by_key,
    )
