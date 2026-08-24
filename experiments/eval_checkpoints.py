"""Evaluate saved per-epoch checkpoints on validation, OCTDL, and clinic.

The final training run saves one checkpoint per epoch. This script loads each of
them and reports accuracy, macro-F1, and per-class recall on the Kermany
validation split, on OCTDL, and on the clinic, so that the training curve across
all three sets can be inspected side by side.

Important
---------
Reading these numbers is fine for understanding how training progressed. Choosing
the model of record from them is not. Selecting the epoch with the best OCTDL or
clinic score turns those held-out sets into training signals and inflates the
reported result, and Kermany validation is anti-predictive of transfer. The last
epoch stays the model of record unless a genuinely separate development set is
used for selection.

Usage
-----
    python experiments/eval_checkpoints.py --ckpt /content/drive/MyDrive/convnext_final.pt \
        --model convnext_tiny --width 384 --height 256
"""
from __future__ import annotations

import argparse
import csv
import gc
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ocular import config, data, eval, model, train
from ocular.data import OCTDataset, PreConfig

SETS = ("val", "octdl", "clinic")


def find_checkpoints(base: Path) -> list[tuple[int, Path]]:
    """Return the per-epoch checkpoints for a base path, sorted by epoch.

    Parameters
    ----------
    base : pathlib.Path
        The base checkpoint path, for example ``convnext_final.pt``. Files named
        ``convnext_final_e1.pt``, ``convnext_final_e2.pt``, and so on are matched.

    Returns
    -------
    list of (int, pathlib.Path)
        Pairs of epoch number and path, sorted by epoch.
    """
    pattern = re.compile(rf"^{re.escape(base.stem)}_e(\d+){re.escape(base.suffix)}$")
    found = []
    for path in base.parent.glob(f"{base.stem}_e*{base.suffix}"):
        match = pattern.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def build_val_loader(
    cfg: PreConfig, cache_dir: Path, num_workers: int, batch_size: int = 16
) -> DataLoader:
    """Build the Kermany validation loader, caching the split if needed.

    The validation cache is rebuilt only when it is missing. Rebuilding an
    existing cache is both wasteful and, on a memory constrained machine, a
    source of pressure that can have the process killed, since the writable
    memory map of the full split stays resident while the evaluation sets are
    loaded on top of it.
    """
    x_path = cache_dir / f"{cfg.tag}_val_x.npy"
    y_path = cache_dir / f"{cfg.tag}_val_y.npy"
    if not (x_path.exists() and y_path.exists()):
        data.build_cache(cfg, out_dir=cache_dir, splits=("val",))
    x = np.load(cache_dir / f"{cfg.tag}_val_x.npy", mmap_mode="r")
    y = np.load(cache_dir / f"{cfg.tag}_val_y.npy")
    return DataLoader(
        OCTDataset(x, y, augment=False), batch_size=batch_size, num_workers=num_workers
    )


def free_device_memory(device: torch.device) -> None:
    """Release cached allocations so peak memory does not grow across checkpoints.

    On a unified memory device such as Apple MPS the per-process cache is drawn
    from system RAM, so it is emptied between checkpoints to keep the peak within
    the memory of a 16 GB machine.
    """
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def flat_metrics(metrics: dict, prefix: str) -> dict:
    """Return the metric dictionary flattened with a set-name prefix."""
    row = {f"{prefix}_acc": metrics["accuracy"], f"{prefix}_macro_f1": metrics["macro_f1"]}
    for cls in config.CLASSES:
        row[f"{prefix}_recall_{cls}"] = metrics[f"recall_{cls}"]
    return row


def main() -> None:
    """Parse arguments, evaluate every checkpoint, and write a results table."""
    parser = argparse.ArgumentParser(description="Evaluate per-epoch checkpoints.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Base checkpoint path.")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path, default=config.DATA_DIR / "cache")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Evaluation batch size, kept small to cap peak memory.")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "checkpoint_eval.csv")
    args = parser.parse_args()

    device = train.get_device()
    cfg = PreConfig(args.width, args.height, crop=True, curvature=True)

    checkpoints = find_checkpoints(args.ckpt)
    if not checkpoints:
        raise SystemExit(f"no checkpoints matching {args.ckpt.stem}_e*.pt found")
    print(f"found {len(checkpoints)} checkpoints under {args.ckpt.parent}")

    loaders = {
        "val": build_val_loader(cfg, args.cache_dir, args.num_workers, args.batch_size),
        "octdl": data.octdl_loader(cfg, batch_size=args.batch_size, num_workers=args.num_workers),
        "clinic": data.clinic_loader(cfg, batch_size=args.batch_size),
    }

    fields = ["epoch"]
    for name in SETS:
        fields += [f"{name}_acc", f"{name}_macro_f1"]
        fields += [f"{name}_recall_{c}" for c in config.CLASSES]

    # Resume support. A row is appended per checkpoint, so an interrupted run
    # leaves every completed epoch on disk, and a rerun skips those epochs and
    # continues. The header is written only when the file is new.
    done = set()
    if args.out.exists():
        with args.out.open() as f:
            done = {int(r["epoch"]) for r in csv.DictReader(f)}
    else:
        with args.out.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    for epoch, path in checkpoints:
        if epoch in done:
            print(f"epoch {epoch:2d}  already in {args.out.name}, skipping")
            continue
        net = model.build_model(args.model, pretrained=False)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        row = {"epoch": epoch}
        for name in SETS:
            row.update(flat_metrics(eval.evaluate(net, loaders[name], device), name))
        with args.out.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"epoch {epoch:2d}  "
              f"val mF1 {row['val_macro_f1']:.3f}  "
              f"OCTDL mF1 {row['octdl_macro_f1']:.3f} DRUSEN {row['octdl_recall_DRUSEN']:.3f}  "
              f"clinic mF1 {row['clinic_macro_f1']:.3f} DRUSEN {row['clinic_recall_DRUSEN']:.3f}")
        net.to("cpu")
        del net
        free_device_memory(device)

    print("saved", args.out)


if __name__ == "__main__":
    main()
