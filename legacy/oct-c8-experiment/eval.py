"""Hydra entrypoint for evaluation.

    # held-out internal test set, auto-pick best checkpoint, refit temp on val
    python eval.py paths=kaggle

    # a specific checkpoint / split
    python eval.py paths=kaggle eval.ckpt_path=/kaggle/working/oct_cds_outputs/oct_c8_densenet121/checkpoints/'epoch=17-val'/'macro_f1=0.9236.ckpt'
    python eval.py paths=kaggle eval.split=external_test

Writes  <output_dir>/eval/metrics_<split>.json  +  confusion_<split>.csv
and prints the full breakdown (accuracy, macro-F1, per-class sens/spec/precision/
F1/AUROC/AUPRC, macro AUROC/AUPRC, QWK, confusion matrix, ECE before & after
temperature scaling).
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent / "src"))

from oct_cds.common.logging import get_logger  # noqa: E402
from oct_cds.common.seed import seed_everything  # noqa: E402
from oct_cds.models.loading import load_classifier, resolve_ckpt  # noqa: E402

log = get_logger("eval")


def _get_calibrator(cfg, model, dm, out_dir: Path):
    mode = cfg.eval.get("calibration", "fit_val")
    if mode == "none":
        return None

    from oct_cds.models.calibration import TemperatureScaler

    saved = out_dir.parent / "calibrators" / "temperature.json"
    if mode == "load":
        if saved.exists():
            sc = TemperatureScaler.load(saved)
            log.info("loaded temperature=%.3f from %s", sc.temperature, saved)
            return sc
        log.warning("calibration=load but %s missing -> refitting on val", saved)

    # fit_val (default): fit temperature on the val split for THIS checkpoint
    import torch

    from oct_cds.evaluation.evaluate import collect_logits

    got = collect_logits(model, dm.val_dataloader(num_workers=0))
    sc = TemperatureScaler().fit(got["logits"], torch.as_tensor(got["y_true"]))
    dest = out_dir / "temperature_eval.json"
    sc.save(dest)
    log.info("fitted temperature=%.3f on val -> %s", sc.temperature, dest)
    return sc


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    pre_cfg = OmegaConf.to_container(cfg.preprocess, resolve=True)
    train_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    split = cfg.eval.get("split", "test")
    out_dir = Path(cfg.output_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_ckpt(cfg.eval.get("ckpt_path"), cfg.output_dir)

    manifest = Path(data_cfg["manifest"].get(split, ""))
    if not manifest.exists():
        raise SystemExit(
            f"missing manifest for split {split!r}: {manifest}. "
            f"Run: python -m oct_cds.cli data build"
        )

    from oct_cds.data.dataset import make_datamodule
    from oct_cds.evaluation.evaluate import evaluate_split

    dm = make_datamodule(data_cfg, pre_cfg, train_cfg)
    dm.setup("test")
    model = load_classifier(ckpt_path, model_cfg, train_cfg)

    try:
        calibrator = _get_calibrator(cfg, model, dm, out_dir)
        loader = dm._loader(split, shuffle=False, num_workers=int(train_cfg["num_workers"]))
        evaluate_split(
            model, loader, split, out_dir,
            calibrator=calibrator,
            bootstrap=bool(cfg.eval.get("bootstrap", True)),
        )
        log.info("checkpoint: %s", ckpt_path)
        log.info("report:     %s", out_dir / f"metrics_{split}.json")
    finally:
        if hasattr(dm, "_sets"):
            dm._sets.clear()
        gc.collect()


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
