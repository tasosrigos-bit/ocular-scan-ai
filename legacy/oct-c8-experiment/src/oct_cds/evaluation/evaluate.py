"""Run a trained checkpoint over a split (internal ``test`` or the OPTOPOL
``external_test``) and write a metrics report.

Reports both uncalibrated and (if a temperature scaler is supplied) calibrated
numbers, so you can see what temperature scaling actually bought you.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from oct_cds.common.logging import get_logger
from oct_cds.data.label_map import load_label_map
from oct_cds.evaluation.metrics import (
    bootstrap_ci,
    classification_report_dict,
    confusion_matrix_dict,
    expected_calibration_error,
)

log = get_logger(__name__)


def collect_logits(model, dataloader) -> dict[str, np.ndarray]:
    """Forward the whole split once; return raw logits + labels + paths."""
    import torch

    device = next(model.parameters()).device
    model.eval()
    logits_all, y_all, paths = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            logits_all.append(model(batch["image"].to(device)).float().cpu())
            y_all.append(batch["label"])
            paths.extend(batch["image_path"])
    return {
        "logits": torch.cat(logits_all),
        "y_true": torch.cat(y_all).numpy(),
        "paths": np.asarray(paths),
    }


def _report_from_logits(logits, y_true, calibrator, class_names, bootstrap: bool) -> dict:
    import torch

    if calibrator is not None:
        probs = calibrator.transform(logits).numpy()
    else:
        probs = torch.softmax(logits, dim=1).numpy()
    y_pred = probs.argmax(1)

    rep = classification_report_dict(y_true, y_pred, probs, class_names=class_names)
    rep["confusion_matrix"] = confusion_matrix_dict(y_true, y_pred, class_names)
    if bootstrap and len(y_true):
        rep["accuracy_ci95"] = bootstrap_ci(
            lambda a, b: float((a == b).mean()), y_true, y_pred
        )
        rep["macro_f1_ci95"] = bootstrap_ci(_macro_f1, y_true, y_pred)
    return rep


def _macro_f1(a, b) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(a, b, average="macro", zero_division=0))


def evaluate_split(
    model,
    dataloader,
    split_name: str,
    out_dir: str | Path,
    calibrator=None,
    bootstrap: bool = True,
) -> dict[str, Any]:
    """Evaluate one split, write ``<out_dir>/metrics_<split>.json`` +
    ``confusion_<split>.csv``, and return the report dict."""
    import torch

    lm = load_label_map()
    names = lm.keys

    got = collect_logits(model, dataloader)
    logits, y_true = got["logits"], got["y_true"]

    raw_probs = torch.softmax(logits, dim=1).numpy()
    report: dict[str, Any] = {
        "split": split_name,
        "n_images": int(len(y_true)),
        "classes": names,
        "uncalibrated": _report_from_logits(logits, y_true, None, names, bootstrap),
        "ece_uncalibrated": expected_calibration_error(raw_probs, y_true),
    }
    if calibrator is not None:
        report["temperature"] = float(getattr(calibrator, "temperature", 1.0))
        report["calibrated"] = _report_from_logits(
            logits, y_true, calibrator, names, bootstrap
        )
        report["ece_calibrated"] = report["calibrated"]["ece"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{split_name}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_confusion_csv(
        out_dir / f"confusion_{split_name}.csv",
        report.get("calibrated", report["uncalibrated"])["confusion_matrix"],
    )

    _log_summary(report)
    log.info("wrote %s", out_dir / f"metrics_{split_name}.json")
    return report


def _write_confusion_csv(path: Path, cm: dict) -> None:
    labels = cm["labels"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred", *labels])
        for name, row in zip(labels, cm["matrix"]):
            w.writerow([name, *row])


def _log_summary(report: dict) -> None:
    block = report.get("calibrated", report["uncalibrated"])
    tag = "calibrated" if "calibrated" in report else "uncalibrated"
    log.info("── %s (%s, n=%d) ──", report["split"], tag, report["n_images"])
    log.info(
        "accuracy=%.4f  macro_f1=%.4f  bal_acc=%.4f  QWK=%.4f",
        block["accuracy"], block["macro_f1"], block["balanced_accuracy"],
        block["quadratic_weighted_kappa"],
    )
    log.info(
        "AUROC(macro)=%.4f  AUPRC(macro)=%.4f  ECE(raw)=%.4f%s",
        block.get("auroc_macro", float("nan")),
        block.get("auprc_macro", float("nan")),
        report["ece_uncalibrated"],
        f"  ECE(cal)={report['ece_calibrated']:.4f}" if "ece_calibrated" in report else "",
    )
    for cname, m in block["per_class"].items():
        log.info(
            "  %-14s sens=%.3f spec=%.3f prec=%.3f f1=%.3f auroc=%.3f (n=%d)",
            cname, m["sensitivity"], m["specificity"], m["precision"],
            m["f1"], m.get("auroc", float("nan")), m["support"],
        )
