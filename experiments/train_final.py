"""Train and save one final model on the full training data.

The model grid selected ConvNeXt-Tiny at 384 by 256 under the cropped,
curvature-corrected, squished pipeline. This script trains that configuration on
the full Kermany training set, rather than the screening subset, and saves the
weights so the model can be reused.

Training respects a wall-clock budget and writes a separate checkpoint after every
epoch, named with the epoch number, and the last epoch is also written under the
base name as the official model. A run that is interrupted, or that reaches the
budget, still leaves every completed epoch on disk. No checkpoint is selected on
the Kermany validation split, which is anti-predictive of transfer, nor on the
evaluation sets, which would inflate their scores. The last epoch is the model of
record. The model is scored on OCTDL and the clinic at the end.

Usage
-----
    python experiments/train_final.py --ckpt experiments/convnext_final.pt \
        --max-minutes 210
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocular import config, data, eval, model, train
from ocular.data import OCTDataset, PreConfig


def epoch_ckpt(base: Path, epoch: int) -> Path:
    """Return the per-epoch checkpoint path derived from a base path.

    For a base of ``convnext_final.pt`` and epoch 3 this returns
    ``convnext_final_e3.pt``.

    Parameters
    ----------
    base : pathlib.Path
        The base checkpoint path.
    epoch : int
        The epoch number.

    Returns
    -------
    pathlib.Path
        The per-epoch path.
    """
    return base.with_name(f"{base.stem}_e{epoch}{base.suffix}")


def main() -> None:
    """Parse arguments, train the final model within the budget, and save it."""
    parser = argparse.ArgumentParser(description="Train and save one final model on full data.")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10, help="Maximum number of epochs.")
    parser.add_argument("--max-minutes", type=float, default=210.0, help="Training time budget.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--cache-dir", type=Path, default=config.DATA_DIR / "cache")
    parser.add_argument("--ckpt", type=Path, required=True, help="Where to save the weights.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = train.get_device()
    cfg = PreConfig(args.width, args.height, crop=True, curvature=True)
    print(f"config {cfg.tag}  model {args.model}  device {device}")

    # Build only the training cache, the validation split is not needed here.
    data.build_cache(cfg, out_dir=args.cache_dir, splits=("train",))
    x = np.load(args.cache_dir / f"{cfg.tag}_train_x.npy", mmap_mode="r")
    y = np.load(args.cache_dir / f"{cfg.tag}_train_y.npy")
    loader = DataLoader(
        OCTDataset(x, y, augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    weights = data.class_weights(y).to(device)

    net = model.build_model(args.model, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.max_minutes * 60
    for epoch in range(1, args.epochs + 1):
        if time.time() > deadline:
            print("time budget reached, stopping before this epoch")
            break
        net.train()
        running = 0.0
        for xb, yb in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)
        out = epoch_ckpt(args.ckpt, epoch)
        torch.save(net.state_dict(), out)
        minutes_left = (deadline - time.time()) / 60
        print(f"epoch {epoch}  loss {running / len(loader.dataset):.4f}  "
              f"saved {out.name}  budget left {minutes_left:.0f} min")

    octdl = eval.evaluate(net, data.octdl_loader(cfg, num_workers=args.num_workers), device)
    clinic = eval.evaluate(net, data.clinic_loader(cfg), device)
    print("OCTDL ", {k: round(v, 3) for k, v in octdl.items()})
    print("clinic", {k: round(v, 3) for k, v in clinic.items()})
    torch.save(net.state_dict(), args.ckpt)
    print(f"last-epoch model of record saved as {args.ckpt.name}; "
          f"per-epoch checkpoints kept as {args.ckpt.stem}_e*.pt")


if __name__ == "__main__":
    main()
