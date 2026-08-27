"""Dump per-scan clinic predictions and class probabilities for one checkpoint.

The aggregate evaluation in ``eval_checkpoints.py`` reports recall per class but
not the individual scan outcomes. The clinic confusion matrix and the diagnosis
of the drusen misses both need the prediction and the full probability vector for
each scan, so this script writes one row per clinic B-scan for a single
checkpoint.

The default checkpoint is the first epoch of the final model, which is the
short-schedule model of record. Selecting an epoch by its clinic score would leak
the clinic set, so the epoch here is fixed in advance and not chosen from the
results.

Usage
-----
    python experiments/eval_clinic_scan.py --ckpt experiments/convnext_final_e1.pt \
        --model convnext_tiny --width 384 --height 256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ocular import config
from ocular.classifier import model, train
from ocular.classifier.data import OCTDataset, PreConfig, _clinic_frame


def main() -> None:
    """Parse arguments, score every clinic scan, and write the per-scan table."""
    parser = argparse.ArgumentParser(description="Per-scan clinic predictions for one checkpoint.")
    parser.add_argument("--ckpt", type=Path, default=Path(__file__).parent / "convnext_final_e1.pt")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "clinic_scan_e1.csv")
    args = parser.parse_args()

    device = train.get_device()
    cfg = PreConfig(args.width, args.height, crop=True, curvature=True)

    frame = _clinic_frame()
    label_id = {c: i for i, c in enumerate(config.CLASSES)}
    n = len(frame)
    x = np.empty((n, cfg.out_h, cfg.out_w), dtype=np.uint8)
    y = np.empty(n, dtype=np.int64)
    for i, (path, cls) in enumerate(zip(frame["path"], frame["cls"])):
        x[i] = (cfg.apply(path) * 255).astype(np.uint8)
        y[i] = label_id[cls]

    net = model.build_model(args.model, pretrained=False)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net = net.to(device)
    net.eval()

    loader = DataLoader(OCTDataset(x, y, augment=False), batch_size=64, shuffle=False)
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            probs.append(torch.softmax(net(xb.to(device)), dim=1).cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    preds = probs.argmax(axis=1)

    out = pd.DataFrame({
        "name": [p.name for p in frame["path"]],
        "truth": frame["cls"].to_numpy(),
        "pred": [config.CLASSES[i] for i in preds],
    })
    for j, cls in enumerate(config.CLASSES):
        out[f"p_{cls}"] = probs[:, j].round(4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    correct = int((out["truth"] == out["pred"]).sum())
    print(f"clinic {correct}/{n} correct; wrote {args.out}")


if __name__ == "__main__":
    main()
