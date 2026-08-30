from __future__ import annotations

from pathlib import Path

# src/oct_cds/common/paths.py -> repo root is 3 parents up from this file's dir
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"


def rel_to_root(path: str | Path) -> str:
    """Path as a POSIX string relative to the repo root when possible."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()
