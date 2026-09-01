import pytest

from oct_cds.rag.ingest import KnowledgeBaseError, load_knowledge_base


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


def test_all_pathology_classes_covered_exactly_once(kb):
    assert set(kb.covers_map) == {
        "AMD", "CNV", "CSR", "DME", "DR", "Drusen", "Macular Hole",
    }
    assert "Normal" not in kb.covers_map
    # AMD entry covers both AMD and Drusen
    assert kb.covers_map["AMD"] == kb.covers_map["Drusen"] == "amd"
    assert kb.covers_map["CNV"] == "cnv"


def test_passages_have_stable_ids_and_text(kb):
    ids = [p.id for p in kb.passages]
    assert len(ids) == len(set(ids)), "passage ids must be unique"
    assert "amd#overview" in ids
    assert "csr#model-behavior-note" in ids
    for p in kb.passages:
        assert p.text.strip()
        assert p.id == f"{p.entry_id}#{p.id.split('#', 1)[1]}"


def test_model_behavior_notes_flagged(kb):
    mbn = [p for p in kb.passages if p.is_model_behavior_note]
    assert {p.entry_id for p in mbn} == {"amd", "cnv", "csr", "dme", "dr", "macular_hole"}


def test_sources_parsed_as_strings(kb):
    amd = next(p for p in kb.passages if p.entry_id == "amd")
    assert amd.sources and all(isinstance(s, str) for s in amd.sources)
    assert any("National Eye Institute" in s for s in amd.sources)


def test_missing_class_raises(tmp_path):
    # only one entry, covering AMD -> the other pathology classes are missing
    (tmp_path / "amd.md").write_text(
        "---\ntitle: X\ncovers: [AMD, Drusen]\nkb_version: 1\n---\n## Overview\ntext\n",
        encoding="utf-8",
    )
    with pytest.raises(KnowledgeBaseError, match="no knowledge-base entry covers"):
        load_knowledge_base(tmp_path)


def test_normal_in_covers_raises(tmp_path):
    for name, cov in [
        ("amd.md", "[AMD, Drusen]"), ("cnv.md", "[CNV]"), ("csr.md", "[CSR]"),
        ("dme.md", "[DME]"), ("dr.md", "[DR]"), ("mh.md", "[Macular Hole]"),
        ("norm.md", "[Normal]"),
    ]:
        (tmp_path / name).write_text(
            f"---\ntitle: X\ncovers: {cov}\nkb_version: 1\n---\n## Overview\ntext\n",
            encoding="utf-8",
        )
    with pytest.raises(KnowledgeBaseError, match="Normal"):
        load_knowledge_base(tmp_path)
