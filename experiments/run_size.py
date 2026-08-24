"""Run the size and preprocessing phase.

This script evaluates the eleven preprocessing configurations that make up the
size phase. Each configuration is preprocessed into a cache, a ResNet50 is trained
on it with class weights, and the trained model is scored on the held-out OCTDL
set and on the clinic set. One row of metrics per configuration is appended to a
results CSV, which the preprocessing notebook reads to report the search.

The configurations are the single source of truth for the phase and are defined
below in ``RUNS``. The script is resumable, a configuration already present in the
results CSV is skipped, and caches are reused across invocations unless
``--delete-cache`` is given.

Usage
-----
Run every configuration with the defaults::

    python experiments/run_size.py

Run a subset for a quick pilot::

    python experiments/run_size.py --runs 4 7 --epochs 3
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ocular import config, data, eval, model, train
from ocular.data import PreConfig

#: The eleven configurations of the size phase. The fill knob is implied by the
#: regime, black in the cropped regime and background fill in the full-frame one.
RUNS = [
    {"run": 1, "set": "A", "ratio": "1:1", "cfg": PreConfig(448, 448, crop=False)},
    {"run": 2, "set": "A", "ratio": "1.5:1", "cfg": PreConfig(544, 352, crop=False)},
    {"run": 3, "set": "A", "ratio": "3:1", "cfg": PreConfig(768, 256, crop=False)},
    {"run": 4, "set": "B", "ratio": "1:1", "cfg": PreConfig(256, 256, curvature=False)},
    {"run": 5, "set": "B", "ratio": "1.5:1", "cfg": PreConfig(384, 256, curvature=False)},
    {"run": 6, "set": "B", "ratio": "3:1", "cfg": PreConfig(768, 256, curvature=False)},
    {"run": 7, "set": "B", "ratio": "1:1", "cfg": PreConfig(256, 256, curvature=True)},
    {"run": 8, "set": "B", "ratio": "1.5:1", "cfg": PreConfig(384, 256, curvature=True)},
    {"run": 9, "set": "B", "ratio": "3:1", "cfg": PreConfig(768, 256, curvature=True)},
    {"run": 10, "set": "B", "ratio": "1:1", "cfg": PreConfig(256, 256, curvature=False, fit="letterbox")},
    {"run": 11, "set": "B", "ratio": "1:1", "cfg": PreConfig(256, 256, curvature=False, fit="width_crop")},
]

#: Column order of the results CSV.
FIELDS = [
    "run", "tag", "set", "ratio", "width", "height", "crop", "curvature", "fit",
    "seed", "epochs",
    "octdl_acc", "octdl_macro_f1",
    "octdl_recall_CNV", "octdl_recall_DME", "octdl_recall_DRUSEN", "octdl_recall_NORMAL",
    "clinic_acc", "clinic_macro_f1",
    "clinic_recall_CNV", "clinic_recall_DME", "clinic_recall_DRUSEN", "clinic_recall_NORMAL",
    "min_per_epoch", "timestamp",
]


def set_seed(seed: int) -> None:
    """Seed the Python, NumPy, and torch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prefixed(m: dict, prefix: str) -> dict:
    """Return the metric dictionary with each key prefixed by ``prefix``."""
    row = {f"{prefix}_acc": m["accuracy"], f"{prefix}_macro_f1": m["macro_f1"]}
    for cls in config.CLASSES:
        row[f"{prefix}_recall_{cls}"] = m[f"recall_{cls}"]
    return row


def done_runs(csv_path: Path) -> set[int]:
    """Return the run numbers already present in the results CSV."""
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        return {int(r["run"]) for r in csv.DictReader(f)}


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
    entry: dict,
    cache_dir: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict:
    """Preprocess, train, and score one configuration.

    Parameters
    ----------
    entry : dict
        One item of :data:`RUNS`.
    cache_dir : pathlib.Path
        Directory for the preprocessed caches.
    device : torch.device
        Device to train on.
    epochs, lr, batch_size, num_workers, seed : int or float
        Training settings.

    Returns
    -------
    dict
        A row of results with the columns in :data:`FIELDS`.
    """
    cfg: PreConfig = entry["cfg"]
    set_seed(seed)

    data.build_cache(cfg, out_dir=cache_dir)
    train_loader, val_loader = data.train_val_loaders(
        cfg, cache_dir=cache_dir, batch_size=batch_size, num_workers=num_workers
    )
    weights = data.class_weights(np.load(cache_dir / f"{cfg.tag}_train_y.npy"))

    net = model.build_model("resnet50", pretrained=True)
    start = time.time()
    net, _ = train.train(
        net, train_loader, val_loader,
        weights=weights, epochs=epochs, lr=lr, device=device,
    )
    min_per_epoch = (time.time() - start) / 60 / epochs

    octdl = eval.evaluate(net, data.octdl_loader(cfg, num_workers=num_workers), device)
    clinic = eval.evaluate(net, data.clinic_loader(cfg), device)

    return {
        "run": entry["run"], "tag": cfg.tag, "set": entry["set"], "ratio": entry["ratio"],
        "width": cfg.out_w, "height": cfg.out_h, "crop": cfg.crop,
        "curvature": cfg.curvature, "fit": cfg.fit, "seed": seed, "epochs": epochs,
        **_prefixed(octdl, "octdl"),
        **_prefixed(clinic, "clinic"),
        "min_per_epoch": round(min_per_epoch, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    """Parse arguments and run the requested configurations."""
    parser = argparse.ArgumentParser(description="Run the size and preprocessing phase.")
    parser.add_argument("--runs", type=int, nargs="*", help="Run numbers to execute, default all.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--cache-dir", type=Path, default=config.DATA_DIR / "cache")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "size_results.csv")
    parser.add_argument("--delete-cache", action="store_true", help="Delete each cache after its run.")
    args = parser.parse_args()

    device = train.get_device()
    selected = args.runs if args.runs else [e["run"] for e in RUNS]
    already = done_runs(args.out)

    for entry in RUNS:
        if entry["run"] not in selected:
            continue
        if entry["run"] in already:
            print(f"run {entry['run']} already in {args.out.name}, skipping")
            continue
        print(f"=== run {entry['run']}  set {entry['set']}  ratio {entry['ratio']}  "
              f"{entry['cfg'].tag} ===")
        row = run_one(
            entry, args.cache_dir, device,
            args.epochs, args.lr, args.batch_size, args.num_workers, args.seed,
        )
        append_row(args.out, row)
        print(f"    OCTDL acc {row['octdl_acc']:.3f}  macro-F1 {row['octdl_macro_f1']:.3f}  "
              f"DRUSEN {row['octdl_recall_DRUSEN']:.3f}  clinic acc {row['clinic_acc']:.3f}")
        if args.delete_cache:
            cfg = entry["cfg"]
            for split in ("train", "val"):
                for suffix in ("x", "y"):
                    (args.cache_dir / f"{cfg.tag}_{split}_{suffix}.npy").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
