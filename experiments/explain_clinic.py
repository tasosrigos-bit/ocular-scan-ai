"""Dump a Grad-CAM explanation for every clinic B-scan under one checkpoint.

Companion to ``eval_clinic_scan.py``, which reports the prediction and class
probabilities per scan but not where the model is looking. This script reuses
the same checkpoint and preprocessing configuration, runs Grad-CAM on every
scorable clinic B-scan, and writes the preprocessed scan and its class
activation overlay as PNGs, plus one manifest row per scan recording the
prediction and probabilities.

The Grad-CAM target class is fixed to the model's own prediction for each scan,
so the heat map always explains the call actually made, right or wrong.

Usage
-----
    python experiments/explain_clinic.py --ckpt experiments/convnext_final_e1.pt \
        --model convnext_tiny --width 384 --height 256
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ocular import config, model, train
from ocular.data import IMAGENET_MEAN, IMAGENET_STD, PreConfig, _clinic_frame
from ocular.explain import gradcam, overlay


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

    out_dir = args.out_dir
    manifest = args.manifest

    device = train.get_device()
    cfg = PreConfig(args.width, args.height, crop=True, curvature=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    net = model.build_model(args.model, pretrained=False)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net = net.to(device)
    net.eval()

    frame = _clinic_frame()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    rows = []
    seen: dict[str, int] = {}
    for path, truth in tqdm(list(zip(frame["path"], frame["cls"])), desc="explaining"):
        scan_arr = cfg.apply(path)
        tensor = torch.from_numpy(scan_arr).float().unsqueeze(0).repeat(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(net(tensor), dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())

        # Reuse the tensor already built for the forward pass; running Grad-CAM
        # on it directly avoids preprocessing the file a second time.
        cam = gradcam(net, tensor, target=pred_idx, device=device)
        blended = overlay(scan_arr, cam, alpha=args.alpha)

        # Disambiguate any two scans that share a filename stem, so their PNGs
        # do not silently overwrite each other.
        stem = path.stem
        if stem in seen:
            seen[stem] += 1
            stem = f"{stem}_{seen[stem]}"
        else:
            seen[stem] = 0
        scan_png = out_dir / f"{stem}_scan.png"
        overlay_png = out_dir / f"{stem}_overlay.png"
        save_gray(scan_arr, scan_png)
        save_rgb(blended, overlay_png)

        row = {
            "name": path.name,
            "truth": truth,
            "pred": config.CLASSES[pred_idx],
            # os.path.relpath never raises when the paths are unrelated (unlike
            # Path.relative_to), so custom --out-dir/--manifest cannot crash it.
            "scan_png": os.path.relpath(scan_png, manifest.parent.parent),
            "overlay_png": os.path.relpath(overlay_png, manifest.parent.parent),
        }
        for j, cls in enumerate(config.CLASSES):
            row[f"p_{cls}"] = round(float(probs[j]), 4)
        rows.append(row)

    out = pd.DataFrame(rows)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(manifest, index=False)
    correct = int((out["truth"] == out["pred"]).sum())
    print(f"clinic {correct}/{len(out)} correct; wrote {len(out) * 2} images to {out_dir}")
    print(f"manifest written to {manifest}")


if __name__ == "__main__":
    main()
