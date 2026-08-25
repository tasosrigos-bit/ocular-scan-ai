"""Dump a Grad-CAM explanation for every clinic B-scan under one checkpoint.

Companion to ``eval_clinic_scan.py``, which reports the prediction and class
probabilities per scan but not where the model is looking. This script reuses
the same checkpoint and preprocessing configuration, runs
:func:`ocular.explain.explain_scan` on every scorable clinic B-scan, and writes
the preprocessed scan and its class activation overlay as PNGs, plus one
manifest row per scan recording the prediction and probabilities.

The Grad-CAM target class is fixed to the model's own prediction for each scan,
so the heat map always explains the call actually made, right or wrong.

Usage
-----
    python experiments/explain_clinic.py --ckpt experiments/convnext_final_e1.pt \
        --model convnext_tiny --width 384 --height 256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ocular import config, model, train
from ocular.data import PreConfig, _clinic_frame
from ocular.explain import explain_scan, overlay


def save_gray(arr: np.ndarray, path: Path) -> None:
    """Write a single-channel ``[0, 1]`` array as an 8-bit grey-scale PNG."""
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def save_rgb(arr: np.ndarray, path: Path) -> None:
    """Write a three-channel ``[0, 1]`` array as an 8-bit RGB PNG."""
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB").save(path)


def main() -> None:
    """Parse arguments, explain every clinic scan, and write images and a manifest."""
    parser = argparse.ArgumentParser(description="Grad-CAM explanations for every clinic B-scan.")
    parser.add_argument("--ckpt", type=Path, default=Path(__file__).parent / "convnext_final_e1.pt")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.45, help="Heat map weight in the overlay.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results" / "explanations")
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).parent / "results" / "clinic_explanations.csv"
    )
    args = parser.parse_args()

    device = train.get_device()
    cfg = PreConfig(args.width, args.height, crop=True, curvature=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    net = model.build_model(args.model, pretrained=False)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net = net.to(device)
    net.eval()

    frame = _clinic_frame()
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

    rows = []
    for path, truth in tqdm(list(zip(frame["path"], frame["cls"])), desc="explaining"):
        scan_arr = cfg.apply(path)
        tensor = torch.from_numpy(scan_arr).float().unsqueeze(0).repeat(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(net(tensor), dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())

        _, cam = explain_scan(net, path, cfg, target=pred_idx, device=device)
        blended = overlay(scan_arr, cam, alpha=args.alpha)

        stem = path.stem
        scan_png = args.out_dir / f"{stem}_scan.png"
        overlay_png = args.out_dir / f"{stem}_overlay.png"
        save_gray(scan_arr, scan_png)
        save_rgb(blended, overlay_png)

        row = {
            "name": path.name,
            "truth": truth,
            "pred": config.CLASSES[pred_idx],
            "scan_png": str(scan_png.relative_to(args.manifest.parent.parent)),
            "overlay_png": str(overlay_png.relative_to(args.manifest.parent.parent)),
        }
        for j, cls in enumerate(config.CLASSES):
            row[f"p_{cls}"] = round(float(probs[j]), 4)
        rows.append(row)

    out = pd.DataFrame(rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.manifest, index=False)
    correct = int((out["truth"] == out["pred"]).sum())
    print(f"clinic {correct}/{len(out)} correct; wrote {len(out) * 2} images to {args.out_dir}")
    print(f"manifest written to {args.manifest}")


if __name__ == "__main__":
    main()
