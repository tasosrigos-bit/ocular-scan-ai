"""Clinic Case Browser — a chat-style viewer for the pre-computed Part 2
narratives. No model, no GPU: reads only the JSONL / JSON that `rag_narrate.py`
already wrote.

    streamlit run app/case_browser.py
    OCT_CDS_RAG_DIR=/path/to/<output_dir>/rag  streamlit run app/case_browser.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="OCT CDS — Clinic Case Browser", page_icon="🔬", layout="wide"
)

DEFAULT_RAG_DIR = "/kaggle/working/oct_cds_outputs/oct_c8_densenet121/rag"

URGENCY_COLOR = {
    "emergent": "#b02a37", "urgent": "#dc3545", "soon": "#fd7e14",
    "routine": "#0d6efd", "none": "#198754",
}
_CITE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9#\-_ ]*)\]")


# ─────────────────────────────── data loading ───────────────────────────────
@st.cache_data(show_spinner=False)
def load_split(rag_dir: str, split: str) -> tuple[list[dict], dict | None]:
    d = Path(rag_dir)
    jf = d / f"narratives_{split}.jsonl"
    cases = [json.loads(x) for x in jf.read_text(encoding="utf-8").splitlines() if x.strip()]
    sf = d / f"summary_{split}.json"
    summary = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else None
    return cases, summary


def available_splits(rag_dir: str) -> list[str]:
    d = Path(rag_dir)
    return sorted(p.stem.replace("narratives_", "") for p in d.glob("narratives_*.jsonl"))


# ─────────────────────────────── rendering ──────────────────────────────────
def outcome_of(case: dict) -> str:
    m = case.get("narrator_meta", {})
    if not m.get("rag_used"):
        return "skipped"
    if m.get("fallback_used"):
        return "fallback"
    return "verified"


ICON = {"verified": "✓", "skipped": "⤳", "fallback": "✗"}


def badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85em">{text}</span>'
    )


def highlight_citations(text: str) -> str:
    return _CITE.sub(
        lambda m: (
            f'<span style="background:#e7f1ff;color:#0a58ca;border:1px solid #b6d4fe;'
            f'border-radius:3px;padding:0 3px;font-family:monospace;font-size:0.85em">'
            f'[{m.group(1)}]</span>'
        ),
        text,
    )


def render_decision(case: dict) -> None:
    imp, tri = case["impression"], case["triage"]
    img_col, meta_col = st.columns([1, 2], gap="large")

    with img_col:
        p = case["case"]["image_path"]
        if p and Path(p).exists():
            st.image(p, caption=Path(p).name, use_container_width=True)
        else:
            st.warning(f"image not found:\n`{p}`")

    with meta_col:
        pred = imp["predicted_class"] or "— (no class asserted)"
        st.markdown(f"### {pred}")
        conf = float(imp["confidence"])
        st.progress(min(max(conf, 0.0), 1.0), text=f"model confidence {conf:.1%}")
        u = tri["urgency"]
        st.markdown(
            "**Triage:** " + badge(u.upper(), URGENCY_COLOR.get(u, "#6c757d"))
            + f" &nbsp; {tri['recommendation']}",
            unsafe_allow_html=True,
        )
        diff = ", ".join(
            f"{d['class']} {float(d['probability']):.0%}" for d in case.get("differential", [])
        )
        st.caption(f"differential — {diff}")
        st.caption(
            "This block is Part 1's rule-engine output. The language model below "
            "never changes it."
        )


def render_narrative(case: dict) -> None:
    m = case.get("narrator_meta", {})
    outcome = outcome_of(case)

    if outcome == "skipped":
        st.info(
            f"**Narrative skipped — {m.get('reason', 'not eligible')}.**\n\n"
            "Routed straight to Part 1's plain output, exactly as the live system "
            "does for normal / abstained / out-of-distribution cases."
        )
        st.code(case["narrative"], language="text")
        return

    if outcome == "fallback":
        st.error(
            f"**Narrative rejected by `verify.py` → fell back to Part 1's template.**\n\n"
            f"flags: `{m.get('flags')}`"
        )
        st.code(case["narrative"], language="text")
        with st.expander("raw model output (rejected)"):
            st.code(m.get("raw_text", "") or "(empty)", language="text")
        return

    st.markdown(
        f'<div style="font-size:1.02rem;line-height:1.6">{highlight_citations(case["narrative_rag"])}</div>',
        unsafe_allow_html=True,
    )
    cites = case.get("citations", [])
    with st.expander(f"Citations ({len(cites)}) — all resolve to knowledge_base/"):
        for c in cites:
            st.markdown(f"**`[{c['id']}]`** &nbsp; {c['label']}", unsafe_allow_html=True)
            for s in c.get("sources", []):
                st.markdown(f"- {s}")
            st.caption(c.get("snippet", ""))
    if m.get("flags"):
        st.caption(f"note: {m['flags']}")


# ─────────────────────────────── sidebar ────────────────────────────────────
st.sidebar.title("🔬 OCT CDS")
st.sidebar.caption("Clinic case browser — pre-computed Part 2 narratives, no inference.")

rag_dir = st.sidebar.text_input(
    "Artifacts directory (`<output_dir>/rag`)",
    value=os.environ.get("OCT_CDS_RAG_DIR", DEFAULT_RAG_DIR),
)

splits = available_splits(rag_dir) if Path(rag_dir).is_dir() else []
if not splits:
    st.sidebar.error("No `narratives_*.jsonl` found here.")
    st.title("Clinic Case Browser")
    st.warning(
        f"No Part 2 artifacts in `{rag_dir}`.\n\n"
        "Run first:  `python rag_narrate.py paths=kaggle rag_run.split=external_test`"
    )
    st.stop()

split = st.sidebar.selectbox(
    "Split", splits, index=(splits.index("external_test") if "external_test" in splits else 0)
)
cases, summary = load_split(rag_dir, split)

if summary:
    st.sidebar.subheader("Verification summary")
    oc = summary.get("outcomes", {})
    eligible = sum(v for k, v in oc.items() if not k.startswith("skipped"))
    verified = oc.get("verified", 0) + oc.get("verified_with_flags", 0)
    st.sidebar.metric(
        "Verified narratives",
        f"{verified}/{eligible}" + (f"  ({verified / eligible:.0%})" if eligible else ""),
    )
    st.sidebar.write({k: v for k, v in sorted(oc.items())})
    st.sidebar.caption(f"backend: `{summary.get('backend', '?')}`")

# filters
classes = sorted({c.get("true_class", "?") for c in cases})
f_out = st.sidebar.multiselect("Filter: outcome", ["verified", "skipped", "fallback"])
f_cls = st.sidebar.multiselect("Filter: true class", classes)


# ─────────────────────────────── main ───────────────────────────────────────
st.title("Clinic Case Browser")
st.caption(
    f"{len(cases)} real OPTOPOL REVO clinic scans · split `{split}` · "
    "the classifier + rule engine decide; the language model only explains."
)

visible = [
    c for c in cases
    if (not f_out or outcome_of(c) in f_out)
    and (not f_cls or c.get("true_class") in f_cls)
]
if not visible:
    st.info("No cases match the filters.")
    st.stop()


def label(c: dict) -> str:
    o = outcome_of(c)
    return f"{ICON[o]}  {Path(c['case']['image_path']).name}   ·   true: {c.get('true_class', '?')}"


_idx = st.selectbox("Case", range(len(visible)), format_func=lambda i: label(visible[i]))
pick = visible[_idx]

fname = Path(pick["case"]["image_path"]).name
with st.chat_message("user"):
    st.markdown(
        f"**Case `{fname}`** &mdash; ground truth **{pick.get('true_class', '?')}** "
        f"&mdash; device: OPTOPOL REVO (external validation set)",
        unsafe_allow_html=True,
    )

with st.chat_message("assistant", avatar="🔬"):
    st.markdown("**Classifier + CDS rule engine — Part 1 (fixed)**")
    render_decision(pick)

with st.chat_message("assistant", avatar="📖"):
    o = outcome_of(pick)
    m = pick.get("narrator_meta", {})
    if o == "verified":
        st.markdown(
            "**Grounded narrative** &nbsp; "
            + badge(f"✓ VERIFIED · narrator_meta.verified = {m.get('verified')}", "#198754"),
            unsafe_allow_html=True,
        )
    elif o == "skipped":
        st.markdown(
            "**Narrative** &nbsp; " + badge("⤳ SKIPPED (routing)", "#6c757d"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "**Narrative** &nbsp; " + badge("✗ NOT VERIFIED — fell back", "#b02a37"),
            unsafe_allow_html=True,
        )
    render_narrative(pick)

with st.expander("full report JSON for this case"):
    st.json(pick)
