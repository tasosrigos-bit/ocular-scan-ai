"""The two tools the assistant calls: the literature search and the scan classifier."""
from __future__ import annotations

from ocular.rag import tools


def test_search_corpus_numbers_the_passages_and_collects_them(tiny_index):
    collected = []
    search = tools.make_search_corpus(tiny_index, k=2, rerank_model=None, collector=collected)
    out = search.invoke({"query": "drusen deposits"})
    assert out.startswith("[1] (Article A - Results)\ndrusen deposits drusen deposits")
    assert "\n\n[2] (Article B - Results)\n" in out
    assert [h.chunk.title for h in collected] == ["Article A", "Article B"]


def test_format_says_so_when_nothing_was_found():
    assert tools.format_passages([]) == "No relevant passages found."


def test_classify_scan_without_an_upload_does_not_load_the_model(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("the classifier should not run without a scan")

    monkeypatch.setattr(tools, "predict_scan", fail)
    classify = tools.make_classify_scan(lambda: None)
    assert "No OCT scan has been uploaded" in classify.invoke({})


def test_classify_scan_reports_the_prediction_and_every_class(monkeypatch):
    probs = {"CNV": 0.05, "DME": 0.9, "DRUSEN": 0.04, "NORMAL": 0.01}
    monkeypatch.setattr(tools, "predict_scan", lambda path, **kw: ("DME", 0.9, probs))
    out = tools.make_classify_scan(lambda: "scan.png").invoke({})
    assert out.startswith("The uploaded B-scan is classified as DME with 90% confidence.")
    assert "CNV 5%, DME 90%, DRUSEN 4%, NORMAL 1%" in out
