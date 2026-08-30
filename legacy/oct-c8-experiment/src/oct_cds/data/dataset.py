"""Torch Dataset + Lightning DataModule driven by the CSV manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from oct_cds.common.logging import get_logger
from oct_cds.common.paths import REPO_ROOT
from oct_cds.data.label_map import load_label_map
from oct_cds.preprocessing.transforms import build_transforms

log = get_logger(__name__)


class OCTManifestDataset(Dataset):
    """One row per image. Only ``quality_flag == 'ok'`` rows should reach training;
    filtering is the DataModule's job so eval can still see flagged rows."""

    def __init__(self, manifest_csv: str | Path, transform=None):
        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.transform = transform
        self.label_map = load_label_map()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.df.iloc[i]
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "label": int(row["label_id"]),
            "label_key": row["label_key"],
            "image_path": row["image_path"],
            "patient_id": row["patient_id"],
            "dataset": row["dataset"],
            "eye": str(row.get("eye", "unknown")),
            "source": str(row.get("source", "")),
        }


class OCTDataModule:
    """Framework-agnostic wrapper; exposes the dataloaders Lightning expects.

    Subclasses ``LightningDataModule`` at runtime if available (either the
    ``lightning`` or ``pytorch_lightning`` distribution), but importable without
    either for tests.
    """

    def __init__(self, data_cfg: Any, preprocess_cfg: Any, training_cfg: Any):
        self.data_cfg = data_cfg
        self.pre_cfg = preprocess_cfg
        self.train_cfg = training_cfg
        self._sets: dict[str, OCTManifestDataset] = {}

    # -- manifests -------------------------------------------------------
    def _manifest_path(self, split: str) -> str:
        return self.data_cfg["manifest"][split]

    def setup(self, stage: str | None = None) -> None:
        m = self.data_cfg["manifest"]
        if "train" in m and Path(m["train"]).exists():
            self._sets["train"] = OCTManifestDataset(
                m["train"], build_transforms(self.pre_cfg, train=True)
            )
            # drop non-ok rows from training only
            ts = self._sets["train"]
            ts.df = ts.df[ts.df["quality_flag"] == "ok"].reset_index(drop=True)
        for split in ("val", "test", "external_test"):
            if split in m and Path(m[split]).exists():
                self._sets[split] = OCTManifestDataset(
                    m[split], build_transforms(self.pre_cfg, train=False)
                )
        log.info("datamodule splits: %s", {k: len(v) for k, v in self._sets.items()})

    # -- loaders --------------------------------------------------------
    def _loader(self, split: str, shuffle: bool, num_workers: int | None = None) -> DataLoader:
        nw = int(self.train_cfg["num_workers"] if num_workers is None else num_workers)
        kwargs: dict[str, Any] = dict(
            batch_size=int(self.train_cfg["batch_size"]),
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
        )
        if nw > 0:
            # persistent_workers=False => workers are torn down at the end of each
            # iteration, so the process can exit cleanly after training.
            kwargs["persistent_workers"] = bool(self.train_cfg.get("persistent_workers", False))
            kwargs["prefetch_factor"] = int(self.train_cfg.get("prefetch_factor", 2))
            timeout = int(self.train_cfg.get("loader_timeout", 0))
            if timeout > 0:
                kwargs["timeout"] = timeout
        return DataLoader(self._sets[split], **kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self, num_workers: int | None = None) -> DataLoader:
        return self._loader("val", shuffle=False, num_workers=num_workers)

    def test_dataloader(self, num_workers: int | None = None) -> DataLoader:
        return self._loader("test", shuffle=False, num_workers=num_workers)

    def external_dataloader(self, num_workers: int | None = None) -> DataLoader:
        return self._loader("external_test", shuffle=False, num_workers=num_workers)


def make_datamodule(data_cfg, preprocess_cfg, training_cfg) -> OCTDataModule:
    """Return an ``OCTDataModule`` that also is-a LightningDataModule when possible."""
    from oct_cds.common.lightning_compat import HAS_LIGHTNING, pl

    if not HAS_LIGHTNING:  # pragma: no cover
        return OCTDataModule(data_cfg, preprocess_cfg, training_cfg)

    class _LitDataModule(OCTDataModule, pl.LightningDataModule):
        def __init__(self, *a, **kw):
            pl.LightningDataModule.__init__(self)
            OCTDataModule.__init__(self, *a, **kw)

    return _LitDataModule(data_cfg, preprocess_cfg, training_cfg)
