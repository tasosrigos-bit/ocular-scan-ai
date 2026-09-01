"""Hydra entrypoint — Part 2. Attach a grounded, cited narrative to CDS
recommendations over a split.

    python rag_narrate.py paths=kaggle                         # external_test, Qwen local
    python rag_narrate.py paths=kaggle rag_run.split=test rag_run.limit=50
    python rag_narrate.py paths=kaggle rag.llm.backend=anthropic   # API fallback
    python rag_narrate.py paths=kaggle rag.llm.backend=stub        # offline dry-run

Writes <output_dir>/rag/narratives_<split>.jsonl (one full report per case) and
summary_<split>.json (verify outcomes: verified / fell back / flagged, skips).
The impression and triage in every report come verbatim from Part 1's rule
engine — the LLM only fills `narrative_rag`.
"""

from __future__ import annotations

import gc
import json
import sys
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent / "src"))

from oct_cds.common.logging import get_logger  # noqa: E402
from oct_cds.common.seed import seed_everything  # noqa: E402
from oct_cds.models.loading import load_classifier, resolve_ckpt  # noqa: E402

log = get_logger("rag_narrate")


def _calibrator(mode, model, dm, out_dir):
    if mode == "none":
        return None
    import torch

    from oct_cds.evaluation.evaluate import collect_logits
    from oct_cds.models.calibration import TemperatureScaler

    saved = out_dir.parent / "calibrators" / "temperature.json"
    if mode == "load" and saved.exists():
        return TemperatureScaler.load(saved)
    got = collect_logits(model, dm.val_dataloader(num_workers=0))
    sc = TemperatureScaler().fit(got["logits"], torch.as_tensor(got["y_true"]))
    sc.save(out_dir / "temperature_rag.json")
    return sc


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import torch

    seed_everything(cfg.seed)
    rr = cfg.rag_run

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    pre_cfg = OmegaConf.to_container(cfg.preprocess, resolve=True)
    train_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    rules = OmegaConf.to_container(cfg.cds, resolve=True)
    rag_cfg = OmegaConf.to_container(cfg.rag, resolve=True)

    split = rr.get("split", "external_test")
    out_dir = Path(cfg.output_dir) / "rag"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = resolve_ckpt(rr.get("ckpt_path"), cfg.output_dir)

    manifest = Path(data_cfg["manifest"].get(split, ""))
    if not manifest.exists():
        raise SystemExit(f"missing manifest for {split!r}: {manifest}")

    from oct_cds.cds.ood_detection import energy_ood_score, msp_ood_score
    from oct_cds.cds.report import build_report
    from oct_cds.cds.rules import CDSRuleEngine
    from oct_cds.cds.schema import CaseInput, ModelResult
    from oct_cds.data.dataset import make_datamodule
    from oct_cds.data.label_map import load_label_map
    from oct_cds.rag.index import build_or_load_index
    from oct_cds.rag.llm import make_backend
    from oct_cds.rag.narrator import Narrator

    lm = load_label_map()
    dm = make_datamodule(data_cfg, pre_cfg, train_cfg)
    dm.setup("test")
    model = load_classifier(ckpt, model_cfg, train_cfg)
    device = next(model.parameters()).device
    calibrator = _calibrator(rr.get("calibration", "fit_val"), model, dm, out_dir)
    temperature = float(getattr(calibrator, "temperature", 1.0)) if calibrator else 1.0
    engine = CDSRuleEngine(rules=rules, label_map=lm)

    # --- build the RAG narrator (index + backend) ---
    backend_cfg = rag_cfg["llm"]
    log.info("rag backend=%s  embed=%s", backend_cfg["backend"], rag_cfg["embed"]["model"])
    index = embedder = kb = None
    if backend_cfg["backend"] != "stub":
        index, embedder, kb = build_or_load_index(
            model_name=rag_cfg["embed"]["model"],
            rebuild=bool(rag_cfg["embed"].get("rebuild_index", False)),
        )
    narrator = Narrator(
        backend=make_backend(backend_cfg),
        kb=kb, index=index, embedder=embedder,
        strict_verify=bool(rag_cfg["verify"]["strict"]),
        max_tokens=int(backend_cfg.get("max_tokens", 400)),
        temperature=float(backend_cfg.get("temperature", 0.0)),
        retry_uncited=int(backend_cfg.get("retry_uncited", 1)),
    )

    ood_mode = rr.get("ood", "msp")
    limit = rr.get("limit")
    only_errors = bool(rr.get("only_errors", False))
    loader = dm._loader(split, shuffle=False, num_workers=min(2, int(train_cfg["num_workers"])))

    reports: list[dict] = []
    outcomes: Counter = Counter()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if limit is not None and len(reports) >= int(limit):
                break
            logits = model(batch["image"].to(device)).float().cpu()
            probs = (calibrator.transform(logits).numpy() if calibrator is not None
                     else torch.softmax(logits, dim=1).numpy())
            ys = batch["label"].tolist()
            for i in range(len(ys)):
                if limit is not None and len(reports) >= int(limit):
                    break
                p = probs[i]
                probs_d = {lm.key(c): float(p[c]) for c in range(lm.num_classes)}
                pred_k = max(probs_d, key=probs_d.get)
                if only_errors and pred_k == lm.key(ys[i]):
                    continue
                ood = (None if ood_mode == "none"
                       else energy_ood_score(logits[i].numpy(), temperature) if ood_mode == "energy"
                       else msp_ood_score(probs_d))
                mr = ModelResult(
                    probs=probs_d,
                    logits={lm.key(c): float(logits[i][c]) for c in range(lm.num_classes)},
                    temperature=temperature, ood_score=ood,
                    model_version=f"{model_cfg['name']}::{ckpt.name}",
                    calibrator_version=f"temp={temperature:.3f}",
                )
                case = CaseInput(
                    image_path=str(batch["image_path"][i]),
                    eye=str(batch.get("eye", ["unknown"] * len(ys))[i]),
                    acquisition_device=str(batch.get("source", [""] * len(ys))[i]) or None,
                )
                rec = engine.evaluate(mr, case)
                report = build_report(rec, mr, case, narrator=narrator)
                report["true_class"] = lm.key(ys[i])
                reports.append(report)
                outcomes[_outcome(report)] += 1

    (out_dir / f"narratives_{split}.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in reports), encoding="utf-8"
    )
    summary = {
        "split": split,
        "n_cases": len(reports),
        "outcomes": dict(outcomes),
        "flag_counts": dict(Counter(
            f for r in reports for f in r.get("narrator_meta", {}).get("flags", [])
        )),
        "backend": backend_cfg["backend"],
        "checkpoint": str(ckpt),
    }
    (out_dir / f"summary_{split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # readable dump of every rejected / flagged raw narrative, for prompt debugging
    rejected = [
        r for r in reports
        if r.get("narrator_meta", {}).get("fallback_used")
        or r.get("narrator_meta", {}).get("flags")
    ]
    if rejected:
        blocks = []
        for r in rejected:
            m = r["narrator_meta"]
            blocks.append(
                f"### {r['case']['image_path']}\n"
                f"true={r.get('true_class')}  pred={r['impression']['predicted_class']}  "
                f"urgency={r['triage']['urgency']}\n"
                f"flags: {m['flags']}\n"
                f"retrieved: {m['retrieved_ids']}\n"
                f"--- raw LLM output ---\n{m.get('raw_text', '') or '(empty)'}\n"
            )
        (out_dir / f"rejected_{split}.txt").write_text("\n\n".join(blocks), encoding="utf-8")
        log.info("wrote %d rejected/flagged raw narratives -> %s",
                 len(rejected), out_dir / f"rejected_{split}.txt")
        log.info("first rejected raw output:\n%s",
                 (rejected[0]["narrator_meta"].get("raw_text") or "(empty)")[:800])

    log.info("── RAG narrate · %s ──", split)
    log.info("outcomes: %s", dict(outcomes))
    log.info("flags: %s", summary["flag_counts"])
    log.info("wrote %s", out_dir / f"narratives_{split}.jsonl")

    if hasattr(dm, "_sets"):
        dm._sets.clear()
    gc.collect()


def _outcome(report: dict) -> str:
    meta = report.get("narrator_meta", {})
    if not meta.get("rag_used"):
        return f"skipped:{meta.get('reason', '?')[:24]}"
    if meta.get("fallback_used"):
        return "fell_back"
    if meta.get("flags"):
        return "verified_with_flags"
    return "verified"


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    gc.collect()
    try:
        from IPython import get_ipython

        _ipy = get_ipython() is not None
    except Exception:  # noqa: BLE001
        _ipy = False
    if not _ipy:
        import os

        os._exit(0)
