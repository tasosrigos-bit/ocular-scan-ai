"""Build the full index grid: every chunk strategy by every embedding model.

This is a convenience batch over ``scripts/build_index.py``. It walks the grid of
chunk strategies and embedding models, skips any index that already exists, and
runs the rest one at a time. A failure on one configuration (for example a gated
model that cannot be downloaded) is reported and the batch continues.

Each embedder carries the device it must run on. The MiniLM baseline runs on the
default device (MPS on Apple silicon); the BERT-based encoders are pinned to CPU,
because their full-corpus encode stalls on the MPS backend on this hardware.

With ``--qdrant`` every index is embedded straight into a Qdrant collection, and
``--corpus-dir`` builds the grid over another corpus. A collection that is already
complete is a quick no-op and a partial one resumes, so the batch can be stopped and
run again at any point. ``--device`` runs every embedder on one device, overriding
the per-model defaults below; on Apple silicon ``mps`` is now the faster choice for
all three, since each batch is a separate short encode rather than one call over the
whole corpus.

Usage
-----
    python scripts/build_all_indexes.py

    # the full corpus into Qdrant, on the Apple GPU
    python scripts/build_all_indexes.py --corpus-dir data/corpus_full --qdrant --device mps
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Build the full index grid.")
    parser.add_argument("--corpus-dir", type=Path, default=None, help="Corpus to index.")
    parser.add_argument(
        "--qdrant", nargs="?", metavar="URL", default=None, const="",
        help="Embed into Qdrant (at URL, else $QDRANT_URL or localhost:6333).",
    )
    parser.add_argument("--device", default=None, help="Run every embedder on this device.")
    args = parser.parse_args()

    # Fastest embedder first, so most of the grid is ready early, and within each
    # embedder the fixed chunks, the configuration the application uses.
    grid = [(s, m, args.device or d) for m, d in EMBEDDERS for s in chunk.STRATEGIES]
    print(f"index grid: {len(grid)} configurations")

    results: list[tuple[str, str]] = []
    for strategy, model, device in grid:
        out_dir = index.INDEX_DIR / f"{strategy}__{_slug(model)}"
        name = out_dir.name
        if args.qdrant is None and args.corpus_dir is None and (out_dir / "manifest.json").exists():
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
        if args.corpus_dir:
            cmd += ["--corpus-dir", str(args.corpus_dir)]
        if args.qdrant is not None:
            cmd += ["--qdrant"] + ([args.qdrant] if args.qdrant else [])

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
