"""Re-run the clinic Grad-CAM explanations under the v2 preprocessing pipeline.

Identical to ``explain_clinic.py`` except every scan is preprocessed with
:func:`ocular.preprocess_v2.preprocess`, which additionally blanks a
fixed-margin rectangle at each corner to strip device UI icons (scale bars,
laterality markers, scan-direction arrows) that survive the original
``ocular.preprocess`` pipeline because they are inset from the border and not
near-white. The original pipeline and its outputs are untouched, so the two
sets of predictions and images can be compared side by side.

Usage
-----
    python experiments/explain_clinic_v2.py --ckpt experiments/convnext_final_e1.pt \
        --model convnext_tiny --width 384 --height 256
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ocular import config, model, train
from ocular.data import PreConfig, _clinic_frame
from ocular.explain import explain_scan, overlay
from ocular.preprocess_v2 import preprocess as preprocess_v2


@dataclass(frozen=True)
class PreConfigV2(PreConfig):
    """A :class:`PreConfig` whose ``apply`` runs the v2 (corner-blanked) pipeline."""

    def apply(self, path):
        return preprocess_v2(
            path, self.out_w, self.out_h,
            crop=self.crop, curvature=self.curvature, fit=self.fit,
        )


def save_gray(arr: np.ndarray, path: Path) -> None:
    """Write a single-channel ``[0, 1]`` array as an 8-bit grey-scale PNG."""
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def save_rgb(arr: np.ndarray, path: Path) -> None:
    """Write a three-channel ``[0, 1]`` array as an 8-bit RGB PNG."""
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB").save(path)


def main() -> None:
    """Parse arguments, explain every clinic scan under v2 preprocessing, and write outputs."""
    parser = argparse.ArgumentParser(description="Grad-CAM explanations under v2 preprocessing.")
    parser.add_argument("--ckpt", type=Path, default=Path(__file__).parent / "convnext_final_e1.pt")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.45, help="Heat map weight in the overlay.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results" / "explanations_v2")
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).parent / "results" / "clinic_explanations_v2.csv"
    )
    args = parser.parse_args()

    device = train.get_device()
    cfg = PreConfigV2(args.width, args.height, crop=True, curvature=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    net = model.build_model(args.model, pretrained=False)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net = net.to(device)
    net.eval()

    frame = _clinic_frame()
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

    rows = []
    for path, truth in tqdm(list(zip(frame["path"], frame["cls"])), desc="explaining (v2)"):
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
