"""Preprocessing for OCT B-scans.

The pipeline removes the parts of a scan that belong to the acquiring device and
retains the retina, so that a model trained on one device transfers to devices it
has not seen. Each step is a configurable knob rather than a fixed choice, and the
combination that transfers best is selected empirically on the held-out OCTDL set.
That search is run by ``experiments/run_preprocessing.py`` and the reasoning behind each
knob is illustrated in ``notebooks/02_classifier_preprocessing.ipynb``.

Two regimes are supported through the ``crop`` flag of :func:`preprocess`.

* **Full-frame regime** (``crop=False``). The whole image is retained. The bright
  rotation padding is replaced by the background level to avoid a hard edge, and
  the image is resized to the target frame.
* **Cropped regime** (``crop=True``). The image is reduced to the retinal band,
  the band is optionally flattened by curvature correction, and the result is
  resized to the target frame.

The public entry point is :func:`preprocess`. The individual stages are exposed so
that a caller can build any intermediate result or unit-test a single step.

Notes
-----
Unless stated otherwise, images are represented as two dimensional
``numpy.float32`` arrays of grey values in the range ``[0, 255]``. The final output
of :func:`preprocess` is normalised to ``[0, 1]``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

#: Default input frame, in pixels. Chosen per experiment, so this is only a
#: neutral fallback and is expected to be overridden by the caller.
INPUT_W = 256
INPUT_H = 256

#: Grey level at or above which a pixel is treated as padding.
WHITE_LEVEL = 248
#: Row-brightness threshold for band detection, as a fraction above background.
BAND_THR = 0.18
#: Fraction of rows ignored at the top and bottom during band detection.
BAND_EDGE = 0.04
#: Margin retained above and below the band, as a fraction of the band height.
CROP_MARGIN = 0.20
#: Percentile of the image taken as the dark background level.
BACKGROUND_PCT = 20
#: Supported width-fitting strategies for :func:`fit_frame`.
FITS = ("squish", "letterbox", "width_crop")


def load_gray(path: str | Path) -> np.ndarray:
    """Read an image from disk as a grey-scale array.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(H, W)`` and dtype ``float32`` with values in
        ``[0, 255]``.
    """
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def padding_mask(g: np.ndarray, white: float = WHITE_LEVEL) -> np.ndarray:
    """Return a boolean mask of the rotation padding.

    Near-white pixels form a candidate mask, and only the connected components
    that touch an edge of the frame are kept. This isolates the padding, which
    fills the corners and reaches the border, from the saturated retinal pigment
    epithelium, which is also near-white but lies in the interior. A plain
    brightness threshold cannot separate the two because both reach 255.

    Parameters
    ----------
    g : numpy.ndarray
        Grey-scale image of shape ``(H, W)``.
    white : float, optional
        Grey level at or above which a pixel is a padding candidate.

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape ``(H, W)`` that is ``True`` on padding pixels.
    """
    mask = g >= white
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    border = set(labels[0]) | set(labels[-1]) | set(labels[:, 0]) | set(labels[:, -1])
    border.discard(0)
    return np.isin(labels, list(border))


def blacken_padding(g: np.ndarray, white: float = WHITE_LEVEL) -> np.ndarray:
    """Set the rotation padding to black.

    The padding is identified with :func:`padding_mask` and set to zero, so that
    it cannot capture the brightness-based band detector. Because the mask is
    restricted to border-connected regions, the interior retina is preserved even
    where it saturates to the same near-white level as the padding.

    Parameters
    ----------
    g : numpy.ndarray
        Grey-scale image of shape ``(H, W)``.
    white : float, optional
        Grey level at or above which a pixel is a padding candidate.

    Returns
    -------
    numpy.ndarray
        Copy of ``g`` with padding pixels set to ``0``.
    """
    g = g.copy()
    g[padding_mask(g, white)] = 0.0
    return g


def background_fill(
    g: np.ndarray, white: float = WHITE_LEVEL, pct: int = BACKGROUND_PCT
) -> np.ndarray:
    """Replace the rotation padding with the image background level.

    The padding is identified with :func:`padding_mask` and set to a low
    percentile of the image, which represents the dark background. Unlike
    :func:`blacken_padding` this introduces no hard edge, and it is used in the
    full-frame regime where the padding is retained in the output. The interior
    retina is preserved because the mask is border-connected.

    Parameters
    ----------
    g : numpy.ndarray
        Grey-scale image of shape ``(H, W)``.
    white : float, optional
        Grey level at or above which a pixel is a padding candidate.
    pct : int, optional
        Percentile of ``g`` used as the background level.

    Returns
    -------
    numpy.ndarray
        Copy of ``g`` with padding pixels set to the background level.
    """
    g = g.copy()
    g[padding_mask(g, white)] = np.percentile(g, pct)
    return g


def retina_rows(
    g: np.ndarray, thr_frac: float = BAND_THR, edge: float = BAND_EDGE
) -> tuple[int, int]:
    """Return the top and bottom row index of the retinal band.

    The mean intensity of each row is smoothed, and the band is identified as
    the longest run of consecutive rows whose intensity exceeds a threshold
    above the background. Extreme top and bottom rows are excluded to avoid
    edge artefacts.

    Parameters
    ----------
    g : numpy.ndarray
        Grey-scale image of shape ``(H, W)``, with padding already neutralised.
    thr_frac : float, optional
        Threshold height between background and peak, as a fraction.
    edge : float, optional
        Fraction of rows excluded at the top and bottom of the search.

    Returns
    -------
    tuple of int
        The ``(top, bottom)`` row indices of the band, inclusive.
    """
    row = g.mean(axis=1)
    k = max(3, len(row) // 60)
    row = np.convolve(row, np.ones(k) / k, mode="same")
    lo, hi = int(edge * len(row)), int((1 - edge) * len(row))
    base = np.percentile(row, 20)
    thr = base + thr_frac * (row[lo:hi].max() - base)
    above = row > thr
    above[:lo] = False
    above[hi:] = False

    best_i = best_len = 0
    i = lo
    while i < hi:
        if above[i]:
            j = i
            while j < hi and above[j]:
                j += 1
            if j - i > best_len:
                best_len, best_i = j - i, i
            i = j
        else:
            i += 1
    return best_i, best_i + best_len - 1


def crop_to_retina(g: np.ndarray, margin: float = CROP_MARGIN) -> np.ndarray:
    """Crop an image to the retinal band with a proportional margin.

    The padding is blackened first so that the band detector is not captured by
    bright borders, then the band is located and the image is cropped to it.

    Parameters
    ----------
    g : numpy.ndarray
        Grey-scale image of shape ``(H, W)``.
    margin : float, optional
        Margin retained above and below the band, as a fraction of its height.

    Returns
    -------
    numpy.ndarray
        The cropped image, of shape ``(h, W)`` with ``h <= H``.
    """
    g = blacken_padding(g)
    top, bottom = retina_rows(g)
    m = int(margin * (bottom - top + 1))
    return g[max(0, top - m): bottom + 1 + m]


def retina_center(g: np.ndarray) -> np.ndarray:
    """Estimate the vertical centre of the retina for each column.

    The centre is the intensity weighted mean row, computed after subtracting
    the background level. The result holds one centre position per column.

    Parameters
    ----------
    g : numpy.ndarray
        Grey-scale image of shape ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(W,)`` holding one centre row per column.
    """
    bg = np.percentile(g, BACKGROUND_PCT)
    w = np.clip(g - bg, 0, None)
    rows = np.arange(g.shape[0])[:, None]
    return (w * rows).sum(0) / (w.sum(0) + 1e-6)


def curvature_correct(g: np.ndarray, deg: int = 3) -> np.ndarray:
    """Flatten the curvature of the retinal band.

    A smooth polynomial is fitted to the per column centre of the band, and
    each column is shifted vertically so that the band becomes horizontal.
    Exposed regions are filled with the background level.

    Parameters
    ----------
    g : numpy.ndarray
        Cropped grey-scale band of shape ``(H, W)``.
    deg : int, optional
        Degree of the polynomial fitted to the band centre.

    Returns
    -------
    numpy.ndarray
        Image of shape ``(H, W)`` with the band flattened. Regions exposed by the
        shift are filled with the background level.
    """
    x = np.arange(g.shape[1])
    fit = np.polyval(np.polyfit(x, retina_center(g), deg), x)
    shift = np.round(fit.mean() - fit).astype(int)
    height = g.shape[0]
    out = np.full_like(g, np.percentile(g, BACKGROUND_PCT))
    for xi in range(g.shape[1]):
        s = shift[xi]
        col = g[:, xi]
        if s > 0:
            out[s:, xi] = col[:height - s]
        elif s < 0:
            out[:height + s, xi] = col[-s:]
        else:
            out[:, xi] = col
    return out


def stretch(g: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    """Rescale intensity to a fixed percentile range.

    The image is linearly mapped so that the given lower and upper percentiles
    span the range 0 to 1, and values outside that range are clipped.

    Parameters
    ----------
    g : numpy.ndarray
        Grey-scale image.
    lo : float, optional
        Lower percentile mapped to 0.
    hi : float, optional
        Upper percentile mapped to 1.

    Returns
    -------
    numpy.ndarray
        Image with values in ``[0, 1]``.
    """
    p_lo, p_hi = np.percentile(g, [lo, hi])
    return np.clip((g - p_lo) / (p_hi - p_lo + 1e-6), 0.0, 1.0)


def squish(g: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize an image to a fixed frame without preserving aspect ratio."""
    return np.asarray(Image.fromarray(g.astype("uint8")).resize((w, h)), np.float32)


def letterbox(g: np.ndarray, w: int, h: int, fill: float = 0) -> np.ndarray:
    """Resize an image to a fixed frame while preserving aspect ratio.

    The image is scaled to fit within the frame and the remaining area is
    padded with a constant fill value.

    Parameters
    ----------
    g : numpy.ndarray
        Grey-scale image in the range ``[0, 255]``.
    w, h : int
        Target width and height in pixels.
    fill : float, optional
        Padding value.

    Returns
    -------
    numpy.ndarray
        Image of shape ``(h, w)`` containing the aspect-preserved content.
    """
    src_h, src_w = g.shape
    s = min(w / src_w, h / src_h)
    nw, nh = max(1, int(src_w * s)), max(1, int(src_h * s))
    im = np.asarray(Image.fromarray(g.astype("uint8")).resize((nw, nh)), np.float32)
    out = np.full((h, w), fill, np.float32)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = im
    return out


def width_crop(g: np.ndarray, w: int, h: int) -> np.ndarray:
    """Crop the lateral edges to a target aspect ratio and resize.

    The central region matching the target aspect ratio is retained, the edges
    are discarded, and the result is resized to the frame.

    Parameters
    ----------
    g : numpy.ndarray
        Grey-scale image in the range ``[0, 255]``.
    w, h : int
        Target width and height in pixels.

    Returns
    -------
    numpy.ndarray
        Image of shape ``(h, w)``.
    """
    src_h, src_w = g.shape
    cw = int(src_h * (w / h))
    if cw < src_w:
        x0 = (src_w - cw) // 2
        g = g[:, x0:x0 + cw]
    return np.asarray(Image.fromarray(g.astype("uint8")).resize((w, h)), np.float32)


def fit_frame(g: np.ndarray, out_w: int, out_h: int, fit: str = "squish") -> np.ndarray:
    """Map an image to the target frame using a width-fitting strategy.

    Parameters
    ----------
    g : numpy.ndarray
        Grey-scale image in the range ``[0, 255]``.
    out_w, out_h : int
        Target width and height in pixels.
    fit : {'squish', 'letterbox', 'width_crop'}, optional
        Strategy for reconciling the source aspect ratio with the target frame.

    Returns
    -------
    numpy.ndarray
        Image of shape ``(out_h, out_w)`` in the range ``[0, 255]``.

    Raises
    ------
    ValueError
        If ``fit`` is not one of :data:`FITS`.
    """
    if fit == "squish":
        return squish(g, out_w, out_h)
    if fit == "letterbox":
        return letterbox(g, out_w, out_h)
    if fit == "width_crop":
        return width_crop(g, out_w, out_h)
    raise ValueError(f"unknown fit {fit!r}, expected one of {FITS}")


def preprocess(
    path: str | Path,
    out_w: int = INPUT_W,
    out_h: int = INPUT_H,
    *,
    crop: bool = True,
    curvature: bool = True,
    fit: str = "squish",
) -> np.ndarray:
    """Run the full preprocessing pipeline on a single file.

    The keyword flags select the regime. In the cropped regime the image is
    reduced to the retinal band and optionally flattened. In the full-frame
    regime the whole image is kept and its padding is replaced by the background
    level. In both regimes the contrast is then stretched and the result is
    mapped to the target frame.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the image file.
    out_w, out_h : int, optional
        Target width and height of the output frame, in pixels.
    crop : bool, optional
        If ``True`` use the cropped regime, otherwise the full-frame regime.
    curvature : bool, optional
        If ``True`` and ``crop`` is ``True``, flatten the band. Ignored in the
        full-frame regime.
    fit : {'squish', 'letterbox', 'width_crop'}, optional
        Width-fitting strategy applied when mapping to the target frame.

    Returns
    -------
    numpy.ndarray
        Single-channel image of shape ``(out_h, out_w)`` and dtype ``float32``
        with values in ``[0, 1]``. A model expecting three channels can repeat
        this channel three times.
    """
    g = load_gray(path)
    if crop:
        g = crop_to_retina(g)
        if curvature:
            g = curvature_correct(g)
    else:
        g = background_fill(g)
    g = stretch(g) * 255.0
    frame = fit_frame(g, out_w, out_h, fit)
    return frame.astype(np.float32) / 255.0
