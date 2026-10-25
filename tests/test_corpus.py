"""The resumable corpus build and the paged search, with NCBI replaced by stubs."""
from __future__ import annotations

import json

import pytest
from lxml import etree

from ocular.rag import corpus


def _article(pmcid: str, licence: str, body: bool = True) -> str:
    abstract = "<abstract><p>Drusen are deposits under the RPE.</p></abstract>" if body else ""
    return f"""<article><front>
      <journal-meta><journal-title-group><journal-title>Retina</journal-title></journal-title-group></journal-meta>
      <article-meta>
        <article-id pub-id-type="pmcid">{pmcid}</article-id>
        <title-group><article-title>Paper {pmcid}</article-title></title-group>
        <permissions><license xmlns:xlink="http://www.w3.org/1999/xlink"
          xlink:href="https://creativecommons.org/licenses/{licence}/4.0/"/></permissions>
        {abstract}
      </article-meta></front></article>"""


LICENCES = {"PMC1": "by", "PMC2": "by-nc-nd", "PMC3": "by", "PMC4": "by"}


@pytest.fixture
def ncbi(monkeypatch):
    """A fake efetch serving four articles: two kept, one NoDerivatives, one empty."""
    requested = []

    def fetch_articles(pmcids):
        requested.extend(pmcids)
        xml = "<pmc-articleset>" + "".join(
            _article(p, LICENCES[p], body=(p != "PMC3")) for p in pmcids
        ) + "</pmc-articleset>"
        root = etree.fromstring(xml)
        return {a.findtext(".//article-id"): a for a in root.iter("article")}

    monkeypatch.setattr(corpus, "fetch_articles", fetch_articles)
    return requested


def test_the_build_keeps_permitted_articles_and_records_every_outcome(ncbi, tmp_path):
    outcomes = corpus.build_corpus_resumable(list(LICENCES), tmp_path, batch=2, workers=2)
    assert outcomes == {"kept": 2, "licence:CC BY-NC-ND 4.0": 1, "empty": 1}
    assert [d.pmcid for d in corpus.load_corpus(tmp_path)] == ["PMC1", "PMC4"]
    assert (tmp_path / "metadata.csv").read_text().count("\n") == 3  # header + 2 rows


def test_a_rerun_fetches_only_what_is_left(ncbi, tmp_path):
    corpus.build_corpus_resumable(["PMC1", "PMC2"], tmp_path, batch=2)
    ncbi.clear()
    corpus.build_corpus_resumable(list(LICENCES), tmp_path, batch=2)
    assert ncbi == ["PMC3", "PMC4"]


def test_a_cut_off_record_is_repaired_and_fetched_again(ncbi, tmp_path):
    corpus.build_corpus_resumable(list(LICENCES), tmp_path, batch=4)
    path = tmp_path / "corpus.jsonl"
    first, last = path.read_text().splitlines()
    path.write_text(first + "\n" + last[:30])                     # killed mid-write
    (tmp_path / "progress.tsv").write_text("PMC1\tkept\n")         # progress lost too
    ncbi.clear()
    corpus.build_corpus_resumable(list(LICENCES), tmp_path, batch=4)
    assert sorted(ncbi) == ["PMC2", "PMC3", "PMC4"]                # PMC1 was intact
    docs = corpus.load_corpus(tmp_path)
    assert sorted(d.pmcid for d in docs) == ["PMC1", "PMC4"]
    assert all(json.loads(line) for line in path.read_text().splitlines())


def test_a_failed_batch_is_marked_for_retry(monkeypatch, tmp_path):
    def down(pmcids):
        raise ConnectionError("NCBI unreachable")

    monkeypatch.setattr(corpus, "fetch_articles", down)
    assert corpus.build_corpus_resumable(["PMC1"], tmp_path) == {"error": 1}


def test_a_large_scope_is_searched_year_by_year(monkeypatch):
    years = {"2026": 3, "2025": 2}

    def count(term):
        if "[pdat]" not in term:
            return 12000  # over the cap, so the search must split
        return next((n for y, n in years.items() if f'"{y}"[pdat]' in term), 0)

    def search(term, retmax):
        year = next((y for y in years if f'"{y}"[pdat]' in term), None)
        return [f"PMC{year}{i}" for i in range(years[year])] if year else []

    monkeypatch.setattr(corpus, "_count", count)
    monkeypatch.setattr(corpus, "search_pmc", search)
    monkeypatch.setattr(corpus.time, "gmtime", lambda: type("t", (), {"tm_year": 2025})())
    ids = corpus.search_pmc_all('"Retina"[Journal]', first_year=2024)
    assert ids == ["PMC20260", "PMC20261", "PMC20262", "PMC20250", "PMC20251"]
