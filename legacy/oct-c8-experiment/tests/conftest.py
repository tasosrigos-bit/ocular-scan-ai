from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from oct_cds.data.label_map import load_label_map  # noqa: E402

CLASS_DIR_MAP = {
    "AMD": "AMD", "CNV": "CNV", "CSR": "CSR", "DME": "DME", "DR": "DR",
    "Drusen": "DRUSEN", "Macular Hole": "MH", "Normal": "NORMAL",
}


def _write_png(path: Path, seed: int = 0) -> None:
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = (rng.random((32, 32)) * 255).astype("uint8")
    arr[8:24, :] = 200  # bright retinal-ish band so retina_roi has something to find
    Image.fromarray(arr).save(path)


@pytest.fixture
def label_map():
    return load_label_map()


@pytest.fixture
def fake_oct_c8(tmp_path: Path) -> dict:
    """Minimal ImageFolder tree: 2 imgs/class/split across train/val/test."""
    root = tmp_path / "data" / "raw" / "oct_c8"
    for split in ("train", "val", "test"):
        for _canon, d in CLASS_DIR_MAP.items():
            cdir = root / split / d
            cdir.mkdir(parents=True)
            for i in range(2):
                _write_png(cdir / f"{d}-{split}-{i}.png", seed=hash((split, d, i)) % 999)
    return {
        "name": "oct_c8",
        "root": str(root),
        "split_dirs": {"train": "train", "val": "val", "test": "test"},
        "class_dir_map": CLASS_DIR_MAP,
        "label_map": None,
        "manifest": {
            "train": str(tmp_path / "processed" / "train.csv"),
            "val": str(tmp_path / "processed" / "val.csv"),
            "test": str(tmp_path / "processed" / "test.csv"),
        },
    }


@pytest.fixture
def fake_optopol(tmp_path: Path) -> dict:
    root = tmp_path / "data" / "external" / "clinic_optopol"
    root.mkdir(parents=True)
    names = [
        "CNV__L.png",
        "DME_3__R.png",
        "NORMAL_12__p0.png",
        "MH__L.png",
        "Drusen_1__R.png",
    ]
    for i, n in enumerate(names):
        _write_png(root / n, seed=i)
    return {
        "name": "clinic_optopol",
        "root": str(root),
        "use": ["external_test"],
        "train_forbidden": True,
        "label_map": None,
        "manifest": {"external_test": str(tmp_path / "processed" / "external.csv")},
    }
