"""Shared checkpoint helpers for eval.py / explain.py."""

from __future__ import annotations

import re
from pathlib import Path

from oct_cds.common.logging import get_logger

log = get_logger(__name__)


def find_best_ckpt(ckpt_dir: str | Path) -> Path:
    """Pick the highest-scoring *.ckpt under ``ckpt_dir`` (recursively).

    Score = the last float in the filename (our template is
    ``epochNN-valf10.9236.ckpt``); older runs put it in a ``val/`` subdir, so
    the parent name is a fallback. ``last.ckpt`` is ignored unless it's all
    that's there.
    """
    ckpt_dir = Path(ckpt_dir)
    cands = [p for p in ckpt_dir.rglob("*.ckpt") if p.name != "last.ckpt"]
    if not cands:
        cands = list(ckpt_dir.rglob("*.ckpt"))
    if not cands:
        raise SystemExit(f"no .ckpt files under {ckpt_dir}")

    def score(p: Path) -> float:
        m = re.findall(r"[0-9]*\.[0-9]+", p.name) or re.findall(r"[0-9]*\.[0-9]+", p.parent.name)
        return float(m[-1]) if m else -1.0

    best = max(cands, key=lambda p: (score(p), p.stat().st_mtime))
    log.info("auto-selected checkpoint: %s", best)
    return best


def resolve_ckpt(ckpt_path, output_dir: str | Path) -> Path:
    p = Path(ckpt_path) if ckpt_path else find_best_ckpt(Path(output_dir) / "checkpoints")
    if not p.exists():
        raise SystemExit(f"checkpoint not found: {p}")
    return p


def load_classifier(ckpt_path: str | Path, model_cfg: dict, train_cfg: dict):
    """Load an ``OCTClassifier`` from a Lightning checkpoint, onto the best device."""
    import torch

    from oct_cds.models.classifier import OCTClassifier

    try:
        model = OCTClassifier.load_from_checkpoint(
            str(ckpt_path), map_location="cpu",
            model_cfg=model_cfg, training_cfg=train_cfg,
        )
    except Exception as exc:  # noqa: BLE001 - fall back to manual state_dict load
        log.warning("load_from_checkpoint failed (%s); loading state_dict manually", exc)
        model = OCTClassifier(model_cfg, train_cfg)
        sd = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(sd.get("state_dict", sd))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval()
