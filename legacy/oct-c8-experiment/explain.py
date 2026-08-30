"""Hydra entrypoint for Grad-CAM explainability.

    # test set, all classes, cap 200 overlays
    python explain.py paths=kaggle

    # the 37 clinic scans, only the Drusen ones the model got wrong,
    # heatmap for BOTH the predicted and the true class
    python explain.py paths=kaggle explain.split=external_test \
        explain.classes=[Drusen] explain.only_errors=true explain.target=both

Overlays are written to
    <output_dir>/explain/<split>/<correct|wrong>/<true_class>/<stem>__...png
plus an index.csv listing every overlay with true / predicted / probability.

Needs the explain extra:  pip install -e '.[explain]'
"""

from __future__ import annotations

import csv
import gc
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent / "src"))

from oct_cds.common.logging import get_logger  # noqa: E402
from oct_cds.common.seed import seed_everything  # noqa: E402
from oct_cds.data.label_map import load_label_map  # noqa: E402
from oct_cds.models.loading import load_classifier, resolve_ckpt  # noqa: E402

log = get_logger("explain")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import torch

    seed_everything(cfg.seed)
    ex = cfg.explain

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    pre_cfg = OmegaConf.to_container(cfg.preprocess, resolve=True)
    train_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    split = ex.get("split", "test")
    target_mode = ex.get("target", "pred")            # pred | true | both
    only_errors = bool(ex.get("only_errors", False))
    max_images = int(ex.get("max_images", 200))
    alpha = float(ex.get("alpha", 0.45))
    bs = int(ex.get("batch_size", 16))
    class_filter = ex.get("classes")
    class_filter = set(class_filter) if class_filter else None

    lm = load_label_map()
    mean, std = pre_cfg["normalize_mean"], pre_cfg["normalize_std"]

    out_root = Path(cfg.output_dir) / "explain" / split
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_ckpt(ex.get("ckpt_path"), cfg.output_dir)

    manifest = Path(data_cfg["manifest"].get(split, ""))
    if not manifest.exists():
        raise SystemExit(
            f"missing manifest for split {split!r}: {manifest}. "
            f"Run: python -m oct_cds.cli data build"
        )

    from oct_cds.data.dataset import make_datamodule
    from oct_cds.explainability.gradcam import GradCAMRunner
    from oct_cds.explainability.overlay import denormalize, save_overlay

    dm = make_datamodule(data_cfg, pre_cfg, train_cfg)
    dm.setup("test")
    model = load_classifier(ckpt_path, model_cfg, train_cfg)
    device = next(model.parameters()).device
    loader = dm._loader(split, shuffle=False, num_workers=min(2, int(train_cfg["num_workers"])))

    runner = GradCAMRunner(model)
    rows: list[dict] = []
    n_written = 0

    try:
        for batch in loader:
            if n_written >= max_images:
                break
            imgs = batch["image"].to(device)
            ys = batch["label"].tolist()
            paths = batch["image_path"]

            with torch.no_grad():
                probs = torch.softmax(model(imgs), dim=1)
            preds = probs.argmax(1).tolist()
            confs = probs.max(1).values.tolist()

            for i in range(len(ys)):
                if n_written >= max_images:
                    break
                true_k, pred_k = lm.key(ys[i]), lm.key(preds[i])
                if class_filter and true_k not in class_filter:
                    continue
                if only_errors and preds[i] == ys[i]:
                    continue

                rgb01 = denormalize(imgs[i], mean, std)
                bucket = "correct" if preds[i] == ys[i] else "wrong"
                stem = Path(paths[i]).stem

                wanted = (
                    [("pred", preds[i]), ("true", ys[i])]
                    if target_mode == "both"
                    else [(target_mode, preds[i] if target_mode == "pred" else ys[i])]
                )
                for tag, cls in wanted:
                    cam = runner.cam(imgs[i], target_class=cls)
                    dest = (
                        out_root / bucket / true_k
                        / f"{stem}__pred-{pred_k}_p{confs[i]:.2f}__cam-{tag}-{lm.key(cls)}.png"
                    )
                    save_overlay(rgb01, cam, dest, alpha=alpha)
                    rows.append({
                        "image": paths[i], "stem": stem,
                        "true": true_k, "pred": pred_k, "prob": round(confs[i], 4),
                        "correct": preds[i] == ys[i],
                        "cam_target": f"{tag}:{lm.key(cls)}",
                        "overlay": str(dest.relative_to(out_root)),
                    })
                n_written += 1
    finally:
        runner.close()
        if hasattr(dm, "_sets"):
            dm._sets.clear()
        gc.collect()

    idx = out_root / "index.csv"
    with idx.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["image", "stem", "true", "pred", "prob", "correct", "cam_target", "overlay"],
        )
        w.writeheader()
        w.writerows(rows)

    log.info("wrote %d overlays for %d images -> %s", len(rows), n_written, out_root)
    log.info("index: %s", idx)
    if class_filter:
        log.info("filtered to true class(es): %s", sorted(class_filter))
    log.info("checkpoint: %s", ckpt_path)


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
