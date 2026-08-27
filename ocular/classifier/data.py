"""Data loading for the ocular project.

This module is the bridge between :mod:`ocular.preprocess`, which turns a single
file into a fixed-size array, and the training code, which consumes batches of
tensors. It provides four things.

* A :class:`PreConfig` that names one preprocessing configuration, so that a
  cache and its loaders can be identified unambiguously.
* :func:`build_cache`, which materialises the preprocessed Kermany training and
  validation splits to disk under a given configuration, optionally keeping only a
  fixed number of training images per class, with :func:`cache_stem` naming the
  result so that a subsampled cache never collides with a full one.
* :class:`OCTDataset` and :func:`train_val_loaders`, which serve the cache as
  three-channel tensors with optional training augmentation.
* :func:`octdl_loader` and :func:`clinic_loader`, which serve the two held-out
  evaluation sets preprocessed under the same configuration, with their native
  labels mapped onto the four Kermany classes.

The label mapping for OCTDL is derived from ``OCTDL_labels.csv``. The disease and
condition columns are collapsed onto the Kermany vocabulary as follows.

======================  ===================
OCTDL disease/condition  Kermany class
======================  ===================
``NO``                   ``NORMAL``
``DME``                  ``DME``
``AMD`` / ``drusen``     ``DRUSEN``
``AMD`` / ``MNV``        ``CNV``
======================  ===================

All other rows, including suspected neovascular AMD and the diseases without a
Kermany counterpart, are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from ocular import config
from ocular.classifier.preprocess import preprocess

#: Per-channel ImageNet mean, applied to match pretrained backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
#: Per-channel ImageNet standard deviation.
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Root of the downloaded OCTDL dataset.
OCTDL_DIR = config.DATA_DIR / "raw" / "octdl_dl"
#: Directory of extracted clinic B-scans.
CLINIC_DIR = config.DATA_DIR / "raw" / "bscans"


@dataclass(frozen=True)
class PreConfig:
    """One preprocessing configuration.

    The fields mirror the keyword arguments of :func:`ocular.preprocess.preprocess`
    and fully determine the cache produced for a run.

    Attributes
    ----------
    out_w, out_h : int
        Width and height of the output frame, in pixels.
    crop : bool
        Whether to use the cropped regime rather than the full-frame regime.
    curvature : bool
        Whether to flatten the band. Only relevant when ``crop`` is ``True``.
    fit : str
        Width-fitting strategy, one of ``squish``, ``letterbox``, ``width_crop``.
    """

    out_w: int
    out_h: int
    crop: bool = True
    curvature: bool = True
    fit: str = "squish"

    @property
    def tag(self) -> str:
        """Return a filesystem-safe identifier for this configuration."""
        regime = "crop" if self.crop else "full"
        curv = "curv" if self.crop and self.curvature else "flat"
        return f"{regime}_{curv}_{self.fit}_{self.out_w}x{self.out_h}"

    def apply(self, path: str | Path) -> np.ndarray:
        """Preprocess one file under this configuration.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the image file.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(out_h, out_w)`` with values in ``[0, 1]``.
        """
        return preprocess(
            path, self.out_w, self.out_h,
            crop=self.crop, curvature=self.curvature, fit=self.fit,
        )


class OCTDataset(Dataset):
    """Serve cached grey-scale images as three-channel tensors.

    Each image is read as ``uint8``, scaled to ``[0, 1]``, repeated across three
    channels to match a pretrained backbone, optionally augmented, and normalised
    with the ImageNet statistics.

    Parameters
    ----------
    x : numpy.ndarray
        Image stack of shape ``(N, H, W)`` and dtype ``uint8``. May be a memory
        mapped array.
    y : array-like of int
        Integer labels of length ``N``.
    augment : bool, optional
        If ``True``, apply random rotation and horizontal flip. Rotation restores
        the robustness removed by curvature correction and is only used for
        training.
    rotate : float, optional
        Maximum absolute rotation angle in degrees when augmenting.
    normalize : bool, optional
        If ``True``, subtract the ImageNet mean and divide by the ImageNet
        standard deviation.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        rotate: float = 15.0,
        normalize: bool = True,
    ) -> None:
        self.x = x
        self.y = np.asarray(y)
        self.augment = augment
        self.rotate = rotate
        self.normalize = normalize
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img = np.asarray(self.x[i], dtype=np.float32) / 255.0
        t = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)
        if self.augment:
            if self.rotate:
                angle = float(torch.empty(1).uniform_(-self.rotate, self.rotate))
                t = TF.rotate(t, angle)
            if torch.rand(1).item() < 0.5:
                t = TF.hflip(t)
        if self.normalize:
            t = (t - self._mean) / self._std
        return t, int(self.y[i])


def class_weights(y: np.ndarray) -> torch.Tensor:
    """Return inverse-frequency class weights for a set of labels.

    The weight of a class is the total count divided by the number of classes
    times the class count, so that rarer classes are weighted more heavily and
    the weights average to one.

    Parameters
    ----------
    y : array-like of int
        Integer labels.

    Returns
    -------
    torch.Tensor
        Weight per class, of length ``len(config.CLASSES)``.
    """
    counts = np.bincount(np.asarray(y), minlength=len(config.CLASSES)).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def cache_stem(cfg: PreConfig, per_class: int | None = None) -> str:
    """Return the filename stem that identifies a cache on disk.

    A cache is named by the configuration that produced it. When only part of the
    training split was used, the stem carries that count as well, so that a
    subsampled cache and a full one under the same configuration never collide.

    Parameters
    ----------
    cfg : PreConfig
        The preprocessing configuration.
    per_class : int, optional
        Images per class retained from the training split, or ``None`` for all.

    Returns
    -------
    str
        The stem, to which ``_{split}_x.npy`` and ``_{split}_y.npy`` are appended.
    """
    return cfg.tag if per_class is None else f"{cfg.tag}_pc{per_class}"


def build_cache(
    cfg: PreConfig,
    out_dir: str | Path | None = None,
    split_csv: str | Path | None = None,
    splits: tuple[str, ...] = ("train", "val"),
    per_class: int | None = None,
    progress: bool = True,
) -> Path:
    """Materialise the preprocessed Kermany splits under one configuration.

    Each split named in ``splits`` is read from the patient-level split file,
    preprocessed image by image, and written to ``out_dir`` as a ``uint8`` image
    stack and an ``int64`` label array. The image stack is streamed to disk as a
    memory-mapped ``.npy`` so that the full split need not fit in memory.

    Parameters
    ----------
    cfg : PreConfig
        The preprocessing configuration.
    out_dir : str or pathlib.Path, optional
        Destination directory. Defaults to ``data/cache``.
    split_csv : str or pathlib.Path, optional
        Path to the split file. Defaults to ``data/split.csv``.
    splits : tuple of str, optional
        Which splits to build.
    per_class : int, optional
        If given, retain only this many images per class from the training split,
        drawn without replacement under the project seed so that the subsample is
        reproducible. The validation split is always built in full, because it is
        used to monitor training rather than to score a run. This is how the
        preprocessing search is screened, and it is reflected in the cache name.
    progress : bool, optional
        Whether to show a progress bar.

    Returns
    -------
    pathlib.Path
        The output directory. Files are named ``{stem}_{split}_x.npy`` and
        ``{stem}_{split}_y.npy``, where the stem comes from :func:`cache_stem`.
    """
    out_dir = Path(out_dir) if out_dir is not None else config.DATA_DIR / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(split_csv if split_csv is not None else config.DATA_DIR / "split.csv")
    label_id = {c: i for i, c in enumerate(config.CLASSES)}
    stem = cache_stem(cfg, per_class)

    for split in splits:
        part = df[df["split"] == split].reset_index(drop=True)
        if per_class is not None and split == "train":
            part = (part.groupby("class", group_keys=False)
                        .sample(n=per_class, random_state=config.SEED)
                        .reset_index(drop=True))
        n = len(part)
        x = open_memmap(
            out_dir / f"{stem}_{split}_x.npy",
            mode="w+", dtype=np.uint8, shape=(n, cfg.out_h, cfg.out_w),
        )
        y = np.empty(n, dtype=np.int64)
        rows = zip(part["filename"], part["class"])
        if progress:
            rows = tqdm(rows, total=n, desc=f"{stem} {split}")
        for i, (fname, cls) in enumerate(rows):
            img = cfg.apply(config.TRAINING_DIR / cls / fname)
            x[i] = (img * 255).astype(np.uint8)
            y[i] = label_id[cls]
        x.flush()
        np.save(out_dir / f"{stem}_{split}_y.npy", y)
    return out_dir


def train_val_loaders(
    cfg: PreConfig,
    cache_dir: str | Path | None = None,
    batch_size: int = 32,
    augment: bool = True,
    num_workers: int = 4,
    rotate: float = 15.0,
    per_class: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build training and validation loaders from a cache.

    Parameters
    ----------
    cfg : PreConfig
        The configuration whose cache to read.
    cache_dir : str or pathlib.Path, optional
        Directory holding the cache. Defaults to ``data/cache``.
    batch_size : int, optional
        Batch size for both loaders.
    augment : bool, optional
        Whether to augment the training loader.
    num_workers : int, optional
        Number of worker processes per loader.
    rotate : float, optional
        Maximum rotation angle in degrees for training augmentation.
    per_class : int, optional
        The subsample the cache was built with, which must match the value passed
        to :func:`build_cache` because it forms part of the cache name.

    Returns
    -------
    tuple of torch.utils.data.DataLoader
        The training and validation loaders.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else config.DATA_DIR / "cache"
    stem = cache_stem(cfg, per_class)

    def read(split: str) -> tuple[np.ndarray, np.ndarray]:
        x = np.load(cache_dir / f"{stem}_{split}_x.npy", mmap_mode="r")
        y = np.load(cache_dir / f"{stem}_{split}_y.npy")
        return x, y

    xt, yt = read("train")
    xv, yv = read("val")
    train = DataLoader(
        OCTDataset(xt, yt, augment=augment, rotate=rotate),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val = DataLoader(
        OCTDataset(xv, yv, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return train, val


def _octdl_frame() -> pd.DataFrame:
    """Return the mapped and existing OCTDL images.

    Returns
    -------
    pandas.DataFrame
        Frame with columns ``path`` and ``cls``, restricted to rows that map to a
        Kermany class and whose image is present on disk.
    """
    df = pd.read_csv(OCTDL_DIR / "OCTDL_labels.csv")

    def to_class(disease: str, condition: str) -> str | None:
        if disease == "NO":
            return "NORMAL"
        if disease == "DME":
            return "DME"
        if disease == "AMD" and condition == "drusen":
            return "DRUSEN"
        if disease == "AMD" and condition == "MNV":
            return "CNV"
        return None

    df["cls"] = [to_class(d, c) for d, c in zip(df["disease"], df["condition"])]
    df = df[df["cls"].notna()].copy()
    df["path"] = [
        OCTDL_DIR / "OCTDL" / d / f for d, f in zip(df["disease"], df["file_name"])
    ]
    df = df[df["path"].map(lambda p: p.exists())].reset_index(drop=True)
    return df[["path", "cls"]]


def _clinic_frame() -> pd.DataFrame:
    """Return the scorable clinic B-scans.

    The class is taken from the leading token of the filename, and scans whose
    class has no Kermany counterpart are dropped.

    Returns
    -------
    pandas.DataFrame
        Frame with columns ``path`` and ``cls``.
    """
    rows = []
    for p in sorted(CLINIC_DIR.glob("*.png")):
        cls = p.name.split("_")[0]
        if cls in config.CLASSES:
            rows.append((p, cls))
    return pd.DataFrame(rows, columns=["path", "cls"])


def _eval_loader(
    frame: pd.DataFrame,
    cfg: PreConfig,
    batch_size: int,
    num_workers: int,
    progress: bool,
    desc: str,
) -> DataLoader:
    """Preprocess an evaluation frame into memory and wrap it in a loader."""
    label_id = {c: i for i, c in enumerate(config.CLASSES)}
    n = len(frame)
    x = np.empty((n, cfg.out_h, cfg.out_w), dtype=np.uint8)
    y = np.empty(n, dtype=np.int64)
    rows = zip(frame["path"], frame["cls"])
    if progress:
        rows = tqdm(rows, total=n, desc=desc)
    for i, (path, cls) in enumerate(rows):
        x[i] = (cfg.apply(path) * 255).astype(np.uint8)
        y[i] = label_id[cls]
    return DataLoader(
        OCTDataset(x, y, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )


def octdl_loader(
    cfg: PreConfig, batch_size: int = 64, num_workers: int = 4, progress: bool = True
) -> DataLoader:
    """Build the OCTDL evaluation loader under a configuration.

    Parameters
    ----------
    cfg : PreConfig
        The preprocessing configuration, matched to the trained model.
    batch_size : int, optional
        Batch size.
    num_workers : int, optional
        Number of worker processes.
    progress : bool, optional
        Whether to show a progress bar while preprocessing.

    Returns
    -------
    torch.utils.data.DataLoader
        Loader over the mapped OCTDL images, without augmentation.
    """
    return _eval_loader(_octdl_frame(), cfg, batch_size, num_workers, progress, "OCTDL")


def clinic_loader(
    cfg: PreConfig, batch_size: int = 64, num_workers: int = 0, progress: bool = True
) -> DataLoader:
    """Build the clinic evaluation loader under a configuration.

    Parameters
    ----------
    cfg : PreConfig
        The preprocessing configuration, matched to the trained model.
    batch_size : int, optional
        Batch size.
    num_workers : int, optional
        Number of worker processes.
    progress : bool, optional
        Whether to show a progress bar while preprocessing.

    Returns
    -------
    torch.utils.data.DataLoader
        Loader over the scorable clinic B-scans, without augmentation.
    """
    return _eval_loader(_clinic_frame(), cfg, batch_size, num_workers, progress, "clinic")
