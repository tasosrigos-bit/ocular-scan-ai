"""Hydra entrypoint for training.

    python train.py                              # densenet121 on oct_c8 (defaults)
    python train.py model=resnet50
    python train.py model=efficientnet_b3 data=oct_c8 training.max_epochs=40
    python train.py -m model=densenet121,resnet50,efficientnet_b3   # sweep

Requires manifests to exist:  python -m oct_cds.cli data build
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

log = get_logger("train")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    seed_everything(cfg.seed, deterministic=cfg.training.get("deterministic", True))
    log.info("\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    from oct_cds.data.dataset import make_datamodule
    from oct_cds.models.classifier import OCTClassifier

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    pre_cfg = OmegaConf.to_container(cfg.preprocess, resolve=True)
    train_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    manifest = Path(data_cfg["manifest"]["train"])
    if not manifest.exists():
        raise SystemExit(
            f"missing manifest {manifest}. Run: python -m oct_cds.cli data build"
        )

    # optional class weights
    class_weights = None
    if train_cfg.get("loss", {}).get("class_weighted"):
        from oct_cds.models.losses import class_weights_from_manifest

        class_weights = class_weights_from_manifest(str(manifest), model_cfg["num_classes"])

    dm = make_datamodule(data_cfg, pre_cfg, train_cfg)
    dm.setup("fit")
    model = OCTClassifier(model_cfg, train_cfg, class_weights=class_weights)

    from oct_cds.common.lightning_compat import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
        require_lightning,
    )

    pl = require_lightning()

    ckpt_cfg = train_cfg.get("checkpoint", {})
    es_cfg = train_cfg.get("early_stopping", {})
    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        # no '/' in the filename -> avoids a spurious "val/" subdirectory
        filename="epoch{epoch:02d}-valf1{val/macro_f1:.4f}",
        auto_insert_metric_name=False,
        monitor=ckpt_cfg.get("monitor", "val/macro_f1"),
        mode=ckpt_cfg.get("mode", "max"),
        save_top_k=int(ckpt_cfg.get("save_top_k", 2)),
        save_last=bool(ckpt_cfg.get("save_last", True)),   # last.ckpt -> resume anchor
        every_n_epochs=1,
    )
    callbacks = [
        ckpt,
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor=es_cfg.get("monitor", "val/macro_f1"),
            mode=es_cfg.get("mode", "max"),
            patience=int(es_cfg.get("patience", 7)),
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=int(train_cfg["max_epochs"]),
        precision=train_cfg.get("precision", "32-true"),
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", "auto"),
        deterministic=bool(train_cfg.get("deterministic", True)),
        default_root_dir=cfg.output_dir,
        callbacks=callbacks,
    )

    # Resume automatically from the last checkpoint of a previous (interrupted)
    # run in the same output dir, unless training.auto_resume=false or an explicit
    # training.resume_from=<path> is given.
    resume_from = train_cfg.get("resume_from")
    if not resume_from and train_cfg.get("auto_resume", True):
        last = ckpt_dir / "last.ckpt"
        if last.exists():
            resume_from = str(last)
    if resume_from:
        log.info("RESUMING from %s", resume_from)

    best_f1 = 0.0
    try:
        trainer.fit(
            model,
            dm.train_dataloader(),
            dm.val_dataloader(),
            ckpt_path=resume_from,
        )

        best_f1 = float(ckpt.best_model_score) if ckpt.best_model_score is not None else 0.0
        log.info("best val/macro_f1 = %.4f (%s)", best_f1, ckpt.best_model_path)

        # post-fit calibration on val — fit temperature on the BEST checkpoint,
        # not the last-epoch weights still in memory.
        if train_cfg.get("calibrate_after_fit"):
            cal_model = model
            if ckpt.best_model_path and Path(ckpt.best_model_path).exists():
                cal_model = OCTClassifier.load_from_checkpoint(
                    ckpt.best_model_path, map_location="cpu",
                    model_cfg=model_cfg, training_cfg=train_cfg,
                ).to(next(model.parameters()).device)
            _calibrate(cal_model, dm, cfg)
    finally:
        _teardown(trainer, dm, model)

    return best_f1


def _calibrate(model, dm, cfg) -> None:
    import torch

    from oct_cds.models.calibration import TemperatureScaler

    device = next(model.parameters()).device
    model.eval()
    # num_workers=0: a single short pass over val — no worker processes to leak.
    loader = dm.val_dataloader(num_workers=0)
    logits, labels = [], []
    try:
        with torch.no_grad():
            for batch in loader:
                logits.append(model(batch["image"].to(device)).cpu())
                labels.append(batch["label"])
    finally:
        del loader

    scaler = TemperatureScaler().fit(torch.cat(logits), torch.cat(labels))
    dest = Path(cfg.output_dir) / "calibrators" / "temperature.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    scaler.save(dest)
    log.info("fitted temperature = %.3f -> %s", scaler.temperature, dest)


def _teardown(*objs) -> None:
    """Drop big objects and reap DataLoader workers / free VRAM so the process
    (and the Colab cell) exits promptly instead of hanging."""
    for o in objs:
        try:
            if hasattr(o, "_sets"):      # DataModule: drop dataset refs
                o._sets.clear()
        except Exception:  # noqa: BLE001
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _running_under_ipython() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    gc.collect()
    # As a standalone process, hard-exit so lingering non-daemon threads from
    # Hydra/Lightning/DataLoader can't keep the `!python train.py` cell spinning.
    # Skip this when run via `%run` inside a kernel (it would kill the kernel).
    if not _running_under_ipython():
        import os

        os._exit(0)
