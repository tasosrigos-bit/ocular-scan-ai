"""Run the architecture grid on the frozen preprocessing.

The size phase selected the cropped, curvature-corrected, squished pipeline. This
script holds that preprocessing fixed at the two carried frames, 256 by 256 and
384 by 256, and varies the model instead. Each model is trained on the same cache
and scored on OCTDL and the clinic, so the rows are directly comparable and also
comparable to the ResNet50 results in the size phase.

The Vision Transformer requires a fixed 224 by 224 input, so it is wrapped in
:class:`ResizeTo`, which downsizes any input before the forward pass. The other
backbones accept the cache frame directly through their global pooling.

The grid is resumable, a (frame, model) pair already present in the results CSV is
skipped, and the cache for a frame is reused across all its models.

Usage
-----
    python experiments/run_models.py --epochs 5 --num-workers 2 --out model_results.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ocular import config, data, eval, model, train
from ocular.data import PreConfig

#: The two frames carried from the size phase, both cropped and curvature corrected.
SIZES = [
    PreConfig(256, 256, crop=True, curvature=True),
    PreConfig(384, 256, crop=True, curvature=True),
]

#: The models compared. The custom network is the no-pretraining anchor.
MODELS = ["resnet50", "convnext_tiny", "vit_b_16", "custom"]

#: Input size required by the Vision Transformer.
VIT_SIZE = 224

FIELDS = [
    "model", "tag", "width", "height", "params", "seed", "epochs",
    "octdl_acc", "octdl_macro_f1",
    "octdl_recall_CNV", "octdl_recall_DME", "octdl_recall_DRUSEN", "octdl_recall_NORMAL",
    "clinic_acc", "clinic_macro_f1",
    "clinic_recall_CNV", "clinic_recall_DME", "clinic_recall_DRUSEN", "clinic_recall_NORMAL",
    "min_per_epoch", "timestamp",
]


class ResizeTo(nn.Module):
    """Wrap a model so inputs are resized to a fixed square before the forward.

    Parameters
    ----------
    module : torch.nn.Module
        The wrapped model.
    size : int
        Side length the input is resized to.
    """

    def __init__(self, module: nn.Module, size: int) -> None:
        super().__init__()
        self.module = module
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return self.module(x)


def make_model(arch: str) -> nn.Module:
    """Build a model, wrapping the Vision Transformer to accept any input size."""
    net = model.build_model(arch, pretrained=(arch != "custom"))
    if arch == "vit_b_16":
        net = ResizeTo(net, VIT_SIZE)
    return net


def set_seed(seed: int) -> None:
    """Seed the Python, NumPy, and torch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prefixed(metrics: dict, prefix: str) -> dict:
    """Return the metric dictionary with each key prefixed by ``prefix``."""
    row = {f"{prefix}_acc": metrics["accuracy"], f"{prefix}_macro_f1": metrics["macro_f1"]}
    for cls in config.CLASSES:
        row[f"{prefix}_recall_{cls}"] = metrics[f"recall_{cls}"]
    return row


def done_pairs(csv_path: Path) -> set:
    """Return the (tag, model) pairs already present in the results CSV."""
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        return {(r["tag"], r["model"]) for r in csv.DictReader(f)}


def append_row(csv_path: Path, row: dict) -> None:
    """Append one result row to the CSV, writing a header if it is new."""
    new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def run_one(
    cfg: PreConfig,
    arch: str,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    cache_dir: Path,
) -> dict:
    """Train and score one frame and model pair.

    The cache for the frame is assumed to exist already, since ``main`` builds it
    once per frame and reuses it across models.

    Parameters
    ----------
    cfg : PreConfig
        The preprocessing configuration naming the frame and its cache.
    arch : str
        The backbone name passed to :func:`make_model`.
    device : torch.device
        Device to train on.
    epochs, batch_size, num_workers, seed : int
        Training settings.
    lr : float
        Learning rate for the optimiser.
    cache_dir : pathlib.Path
        Directory holding the preprocessed cache for the frame.

    Returns
    -------
    dict
        A row of results with the columns in :data:`FIELDS`.
    """
    set_seed(seed)
    train_loader, val_loader = data.train_val_loaders(
        cfg, cache_dir=cache_dir, batch_size=batch_size, num_workers=num_workers
    )
    weights = data.class_weights(np.load(cache_dir / f"{cfg.tag}_train_y.npy"))

    net = make_model(arch)
    start = time.time()
    net, _ = train.train(
        net, train_loader, val_loader,
        weights=weights, epochs=epochs, lr=lr, device=device,
    )
    min_per_epoch = (time.time() - start) / 60 / epochs

    octdl = eval.evaluate(net, data.octdl_loader(cfg, num_workers=num_workers), device)
    clinic = eval.evaluate(net, data.clinic_loader(cfg), device)

    return {
        "model": arch, "tag": cfg.tag, "width": cfg.out_w, "height": cfg.out_h,
        "params": model.count_parameters(net), "seed": seed, "epochs": epochs,
        **_prefixed(octdl, "octdl"),
        **_prefixed(clinic, "clinic"),
        "min_per_epoch": round(min_per_epoch, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    """Parse arguments and run the requested (frame, model) pairs."""
    parser = argparse.ArgumentParser(description="Run the architecture grid.")
    parser.add_argument("--models", nargs="*", default=MODELS, help="Models to run.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--cache-dir", type=Path, default=config.DATA_DIR / "cache")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "model_results.csv")
    parser.add_argument("--delete-cache", action="store_true")
    args = parser.parse_args()

    device = train.get_device()
    already = done_pairs(args.out)

    for cfg in SIZES:
        pending = [a for a in args.models if (cfg.tag, a) not in already]
        if not pending:
            continue
        data.build_cache(cfg, out_dir=args.cache_dir)
        for arch in pending:
            print(f"=== {cfg.tag}  {arch} ===")
            row = run_one(
                cfg, arch, device,
                args.epochs, args.lr, args.batch_size, args.num_workers, args.seed, args.cache_dir,
            )
            append_row(args.out, row)
            print(f"    OCTDL acc {row['octdl_acc']:.3f}  macro-F1 {row['octdl_macro_f1']:.3f}  "
                  f"DRUSEN {row['octdl_recall_DRUSEN']:.3f}  params {row['params'] / 1e6:.1f}M")
        if args.delete_cache:
            for split in ("train", "val"):
                for suffix in ("x", "y"):
                    (args.cache_dir / f"{cfg.tag}_{split}_{suffix}.npy").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
