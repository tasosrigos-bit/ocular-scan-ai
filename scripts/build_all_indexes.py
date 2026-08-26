"""Build the full index grid: every chunk strategy by every embedding model.

This is a convenience batch over ``scripts/build_index.py``. It walks the grid of
chunk strategies and embedding models, skips any index that already exists, and
runs the rest one at a time. A failure on one configuration (for example a gated
model that cannot be downloaded) is reported and the batch continues.

Each embedder carries the device it must run on. The MiniLM baseline runs on the
default device (MPS on Apple silicon); the BERT-based encoders are pinned to CPU,
because their full-corpus encode stalls on the MPS backend on this hardware.

Usage
-----
    python scripts/build_all_indexes.py
"""
from __future__ import annotations

import subprocess
import sys
import time

from ocular.rag import chunk, index

# (model id, device). ``None`` lets sentence-transformers choose the device.
EMBEDDERS = [
    ("sentence-transformers/all-MiniLM-L6-v2", None),   # baseline; MPS-fast
    ("abhinand/MedEmbed-base-v0.1", "cpu"),             # domain (biomedical)
    ("google/embeddinggemma-300m", "cpu"),             # SOTA general (Gemma 3)
]


def _slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def main() -> None:
    """Build every (strategy, embedder) index that does not already exist."""
    grid = [(s, m, d) for s in chunk.STRATEGIES for m, d in EMBEDDERS]
    print(f"index grid: {len(grid)} configurations")

    results: list[tuple[str, str]] = []
    for strategy, model, device in grid:
        out_dir = index.INDEX_DIR / f"{strategy}__{_slug(model)}"
        name = out_dir.name
        if (out_dir / "manifest.json").exists():
            print(f"\n[skip] {name} (already built)")
            results.append((name, "skipped"))
            continue

        print(f"\n[build] {name} (device={device or 'auto'}) ...")
        cmd = [
            sys.executable, "scripts/build_index.py",
            "--chunk-strategy", strategy,
            "--embed-model", model,
        ]
        if device:
            cmd += ["--device", device]

        t0 = time.time()
        proc = subprocess.run(cmd)
        dt = (time.time() - t0) / 60
        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"[{status}] {name} in {dt:.1f} min")
        results.append((name, status))

    print("\n=== summary ===")
    for name, status in results:
        print(f"  {status:22} {name}")


if __name__ == "__main__":
    main()
