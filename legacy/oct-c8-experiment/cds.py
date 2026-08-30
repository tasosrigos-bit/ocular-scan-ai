"""Hydra entrypoint: run the trained model's calibrated predictions through the
CDS rule layer and summarise the triage recommendations.

    python cds.py paths=kaggle                       # OCT-C8 test set
    python cds.py paths=kaggle cds_run.split=external_test   # the 37 clinic scans

Outputs (under <output_dir>/cds/):
    recommendations_<split>.csv   one row per image: probs summary, ood score,
                                  abstained / ood_rejected, urgency, text
    summary_<split>.json          aggregate: urgency mix, abstention rate, and the
                                  key cross-tab -> for MISCLASSIFIED images, did
                                  CDS abstain (good) or assert a confident wrong
                                  call (bad)?
    reports/<split>/<stem>.txt    full build_report() narrative (cds_run.write_reports_for)
    audit_<split>.jsonl           append-only audit log (cds_run.audit)

The rule thresholds live in configs/cds/rules_v1.yaml and can be overridden on
the CLI, e.g.  cds.min_confidence=0.75  cds.min_margin=0.20
"""

from __future__ import annotations

import csv
import gc
import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent / "src"))

from oct_cds.cds.batch import summarize_recommendations  # noqa: E402
from oct_cds.common.logging import get_logger  # noqa: E402
from oct_cds.common.seed import seed_everything  # noqa: E402
from oct_cds.data.label_map import load_label_map  # noqa: E402
from oct_cds.models.loading import load_classifier, resolve_ckpt  # noqa: E402

log = get_logger("cds")


def _calibrator(mode: str, model, dm, out_dir: Path):
    if mode == "none":
        return None
    import torch

    from oct_cds.evaluation.evaluate import collect_logits
    from oct_cds.models.calibration import TemperatureScaler

    saved = out_dir.parent / "calibrators" / "temperature.json"
    if mode == "load" and saved.exists():
        sc = TemperatureScaler.load(saved)
        log.info("loaded temperature=%.3f", sc.temperature)
        return sc
    if mode == "load":
        log.warning("calibration=load but %s missing -> refitting on val", saved)

    got = collect_logits(model, dm.val_dataloader(num_workers=0))
    sc = TemperatureScaler().fit(got["logits"], torch.as_tensor(got["y_true"]))
    sc.save(out_dir / "temperature_cds.json")
    log.info("fitted temperature=%.3f on val", sc.temperature)
    return sc


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import torch

    seed_everything(cfg.seed)
    rc = cfg.cds_run

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    pre_cfg = OmegaConf.to_container(cfg.preprocess, resolve=True)
    train_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    rules = OmegaConf.to_container(cfg.cds, resolve=True)

    split = rc.get("split", "test")
    ood_mode = rc.get("ood", "msp")
    out_dir = Path(cfg.output_dir) / "cds"
    out_dir.mkdir(parents=True, exist_ok=True)

    if ood_mode == "energy":
        log.warning(
            "ood=energy produces scores well outside [0,1]; the rules' "
            "ood_reject_score (default 0.80) will never trip. Override "
            "cds.ood_reject_score to a suitable negative value or use ood=msp."
        )

    ckpt_path = resolve_ckpt(rc.get("ckpt_path"), cfg.output_dir)

    manifest = Path(data_cfg["manifest"].get(split, ""))
    if not manifest.exists():
        raise SystemExit(f"missing manifest for {split!r}: {manifest}. Run data build.")

    from oct_cds.cds.audit import log_recommendation
    from oct_cds.cds.ood_detection import energy_ood_score, msp_ood_score
    from oct_cds.cds.report import build_report
    from oct_cds.cds.rules import CDSRuleEngine
    from oct_cds.cds.schema import CaseInput, ModelResult
    from oct_cds.data.dataset import make_datamodule

    lm = load_label_map()
    dm = make_datamodule(data_cfg, pre_cfg, train_cfg)
    dm.setup("test")
    model = load_classifier(ckpt_path, model_cfg, train_cfg)
    device = next(model.parameters()).device
    calibrator = _calibrator(rc.get("calibration", "fit_val"), model, dm, out_dir)
    temperature = float(getattr(calibrator, "temperature", 1.0)) if calibrator else 1.0

    engine = CDSRuleEngine(rules=rules, label_map=lm)
    loader = dm._loader(split, shuffle=False, num_workers=min(2, int(train_cfg["num_workers"])))

    rows: list[dict] = []
    reports: list[dict] = []
    want_reports = rc.get("write_reports_for", "errors")
    audit_path = out_dir / f"audit_{split}.jsonl" if rc.get("audit", True) else None

    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device)).float().cpu()
            cal_probs = (
                calibrator.transform(logits).numpy()
                if calibrator is not None
                else torch.softmax(logits, dim=1).numpy()
            )
            ys = batch["label"].tolist()
            for i in range(len(ys)):
                p = cal_probs[i]
                probs_d = {lm.key(c): float(p[c]) for c in range(lm.num_classes)}
                if ood_mode == "energy":
                    ood = energy_ood_score(logits[i].numpy(), temperature)
                elif ood_mode == "none":
                    ood = None
                else:
                    ood = msp_ood_score(probs_d)

                mr = ModelResult(
                    probs=probs_d,
                    logits={lm.key(c): float(logits[i][c]) for c in range(lm.num_classes)},
                    temperature=temperature,
                    ood_score=ood,
                    model_version=f"{model_cfg['name']}::{ckpt_path.name}",
                    calibrator_version=f"temp={temperature:.3f}",
                )
                case = CaseInput(
                    image_path=str(batch["image_path"][i]),
                    eye=str(batch.get("eye", ["unknown"] * len(ys))[i]),
                    acquisition_device=str(batch.get("source", [""] * len(ys))[i]) or None,
                )
                rec = engine.evaluate(mr, case)

                true_k, pred_k = lm.key(ys[i]), max(probs_d, key=probs_d.get)
                correct = pred_k == true_k
                deferred = rec.abstained or rec.ood_rejected
                rows.append({
                    "image": case.image_path,
                    "true": true_k,
                    "pred": pred_k,
                    "pred_prob": round(probs_d[pred_k], 4),
                    "margin": rec.margin,
                    "ood_score": None if ood is None else round(float(ood), 4),
                    "correct": correct,
                    "abstained": rec.abstained,
                    "ood_rejected": rec.ood_rejected,
                    "deferred_to_specialist": deferred,
                    "cds_class": rec.predicted_class or "",
                    "urgency": rec.urgency.value,
                    "recommendation": rec.recommendation_text,
                })
                if audit_path is not None:
                    log_recommendation(case, mr, rec, log_path=audit_path)
                if want_reports == "all" or (want_reports == "errors" and not correct):
                    reports.append(build_report(rec, mr, case))

    _write_csv(out_dir / f"recommendations_{split}.csv", rows)
    _write_reports(out_dir / "reports" / split, reports)
    summary = summarize_recommendations(rows, split)
    (out_dir / f"summary_{split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log_summary(summary)

    if hasattr(dm, "_sets"):
        dm._sets.clear()
    gc.collect()
    log.info("checkpoint: %s", ckpt_path)
    log.info("outputs: %s", out_dir)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _write_reports(dir_: Path, reports: list[dict]) -> None:
    if not reports:
        return
    dir_.mkdir(parents=True, exist_ok=True)
    for rep in reports:
        stem = Path(rep["case"]["image_path"]).stem
        (dir_ / f"{stem}.txt").write_text(rep["narrative"], encoding="utf-8")
        (dir_ / f"{stem}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")


def _log_summary(s: dict) -> None:
    log.info(
        "── CDS on %s (n=%d, model acc=%.4f) ──",
        s["split"], s["n_images"], s["model_accuracy"] or 0.0,
    )
    log.info("urgency mix: %s", s["urgency_distribution"])
    log.info(
        "abstention rate: %.4f   ood-reject rate: %.4f",
        s["abstention_rate"] or 0.0, s["ood_reject_rate"] or 0.0,
    )
    m = s["on_misclassified"]
    log.info(
        "MISCLASSIFIED (n=%d): deferred to specialist = %d  |  confident wrong call = %d  %s",
        m["n"], m["deferred_to_specialist"], m["confident_call"], m["confident_call_urgencies"],
    )
    c = s["on_correct"]
    log.info(
        "CORRECT (n=%d): deferred = %d (over-cautious)  |  confident = %d",
        c["n"], c["deferred_to_specialist"], c["confident_call"],
    )
    for k, b in s["misclassified_by_true_class"].items():
        log.info(
            "  true=%-13s  wrong=%d  -> deferred %d / confident %d %s",
            k, b["n"], b["deferred_to_specialist"], b["confident_call"],
            b["confident_call_urgencies"],
        )


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
