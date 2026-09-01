"""Load `knowledge_base/*.md` -> list[Passage], and validate the covers contract."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from oct_cds.common.paths import REPO_ROOT
from oct_cds.data.label_map import load_label_map
from oct_cds.rag.schema import Passage

KB_DIR = REPO_ROOT / "knowledge_base"

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


class KnowledgeBaseError(RuntimeError):
    pass


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return s or "section"


@dataclass
class KnowledgeBase:
    passages: list[Passage]
    covers_map: dict[str, str]           # class_key -> entry_id
    content_hash: str
    kb_dir: Path
    _by_entry: dict[str, list[Passage]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for p in self.passages:
            self._by_entry.setdefault(p.entry_id, []).append(p)

    def entry_for_class(self, class_key: str) -> str | None:
        return self.covers_map.get(class_key)

    def passages_for_entry(self, entry_id: str) -> list[Passage]:
        return list(self._by_entry.get(entry_id, []))

    def by_id(self, passage_id: str) -> Passage | None:
        return next((p for p in self.passages if p.id == passage_id), None)


def _parse_entry(path: Path) -> list[Passage]:
    raw = path.read_text(encoding="utf-8")
    m = _FRONTMATTER.match(raw)
    if not m:
        raise KnowledgeBaseError(f"{path.name}: missing YAML front matter")
    fm = yaml.safe_load(m.group(1)) or {}
    body = raw[m.end():]

    entry_id = path.stem
    title = str(fm.get("title") or entry_id)
    covers = list(fm.get("covers") or [])
    kb_version = int(fm.get("kb_version", 1))
    sources = _normalize_sources(fm.get("sources"))

    if not isinstance(covers, list):
        raise KnowledgeBaseError(f"{path.name}: `covers` must be a list")

    # split body on `## ` headings
    heads = list(_HEADING.finditer(body))
    if not heads:
        raise KnowledgeBaseError(f"{path.name}: no `## ` sections")

    passages: list[Passage] = []
    seen_slugs: set[str] = set()
    for i, h in enumerate(heads):
        heading = h.group(1).strip()
        start = h.end()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        text = body[start:end].strip()
        if not text:
            continue
        slug = _slug(heading)
        if slug in seen_slugs:
            raise KnowledgeBaseError(f"{path.name}: duplicate section slug {slug!r}")
        seen_slugs.add(slug)
        passages.append(
            Passage(
                id=f"{entry_id}#{slug}",
                entry_id=entry_id,
                entry_title=title,
                heading=heading,
                text=text,
                covers=[str(c) for c in covers],
                sources=sources,
                kb_version=kb_version,
            )
        )
    if not passages:
        raise KnowledgeBaseError(f"{path.name}: every section was empty")
    return passages


def _normalize_sources(value) -> list[str]:
    if not value:
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("title") or ""
            url = item.get("url")
            out.append(f"{name} ({url})" if url else str(name))
    return out


def load_knowledge_base(kb_dir: str | Path | None = None) -> KnowledgeBase:
    kb_dir = Path(kb_dir) if kb_dir else KB_DIR
    if not kb_dir.is_dir():
        raise KnowledgeBaseError(f"knowledge base dir not found: {kb_dir}")

    files = sorted(p for p in kb_dir.glob("*.md") if not p.name.startswith("_")
                   and p.name not in {"README.md", "SOURCES.md"})
    if not files:
        raise KnowledgeBaseError(f"no entry files in {kb_dir}")

    passages: list[Passage] = []
    covers_map: dict[str, str] = {}
    hasher = hashlib.sha256()
    for f in files:
        hasher.update(f.read_bytes())
        entry_passages = _parse_entry(f)
        passages.extend(entry_passages)
        for cls in entry_passages[0].covers:
            if cls in covers_map:
                raise KnowledgeBaseError(
                    f"class {cls!r} is covered by both {covers_map[cls]!r} and {f.stem!r}"
                )
            covers_map[cls] = f.stem

    _validate_covers(covers_map)
    return KnowledgeBase(
        passages=passages,
        covers_map=covers_map,
        content_hash=hasher.hexdigest()[:16],
        kb_dir=kb_dir,
    )


def _validate_covers(covers_map: dict[str, str]) -> None:
    lm = load_label_map()
    pathology = [k for k in lm.keys if k != "Normal"]
    missing = [k for k in pathology if k not in covers_map]
    if missing:
        raise KnowledgeBaseError(f"no knowledge-base entry covers: {missing}")
    if "Normal" in covers_map:
        raise KnowledgeBaseError("`Normal` must not appear in any entry's `covers`")
