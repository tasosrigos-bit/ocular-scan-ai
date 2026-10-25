"""The retrieval experiments configuration grid."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_retrieval_experiments", Path(__file__).resolve().parents[1] / "experiments" / "run_retrieval_experiments.py"
)
pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pa)

COLLECTIONS = [
    f"{s}__{m}__corpus_full"
    for s in ("fixed", "section", "semantic")
    for m in ("all-MiniLM-L6-v2", "MedEmbed-base-v0.1", "embeddinggemma-300m")
]
RERANKERS = list(pa.RERANKERS)


def test_the_full_grid_runs_bm25_once_per_chunking_strategy():
    configs = pa.grid(COLLECTIONS, RERANKERS)
    names = [pa.run_name(c) for c in configs]
    # 9 collections x (vector, hybrid) x (none, gte, MedCPT) + 3 strategies x BM25 x 3
    assert len(configs) == 9 * 2 * 3 + 3 * 3 == 63
    assert len(set(names)) == len(names)
    assert {n for n in names if "bm25" in n} == {
        f"{s} | none | bm25 | {r}" for s in ("fixed", "section", "semantic") for r in ("none", "gte", "MedCPT")
    }


def test_a_configuration_records_its_settings():
    cfg = next(c for c in pa.grid(COLLECTIONS[:1], RERANKERS) if c["mode"] == "hybrid" and c["_reranker_id"])
    assert pa.run_name(cfg) == "fixed | MiniLM | hybrid | gte"
    assert cfg["collection"] == "fixed__all-MiniLM-L6-v2__corpus_full"
    assert cfg["reranker"] == "gte-reranker-modernbert-base"


def test_medcpt_runs_are_named_and_recorded():
    cfgs = [c for c in pa.grid(COLLECTIONS[:1], RERANKERS) if c["_reranker_id"] == "ncbi/MedCPT-Cross-Encoder"]
    assert {pa.run_name(c) for c in cfgs} == {
        "fixed | MiniLM | vector | MedCPT", "fixed | none | bm25 | MedCPT", "fixed | MiniLM | hybrid | MedCPT"
    }
    assert {c["reranker"] for c in cfgs} == {"MedCPT-Cross-Encoder"}
