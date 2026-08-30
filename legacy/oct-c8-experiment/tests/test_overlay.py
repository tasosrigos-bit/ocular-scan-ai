import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

from oct_cds.explainability.overlay import colorize, save_overlay  # noqa: E402


def test_colorize_shape_and_range():
    cam = np.linspace(0, 1, 16 * 16).reshape(16, 16).astype(np.float32)
    rgb = colorize(cam)
    assert rgb.shape == (16, 16, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_save_overlay_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.random((24, 24, 3)).astype(np.float32)
    cam = rng.random((24, 24)).astype(np.float32)

    dest = tmp_path / "sub" / "overlay.png"
    out = save_overlay(img, cam, dest, side_by_side=True)
    assert out.exists()

    from PIL import Image

    with Image.open(out) as im:
        assert im.size == (24 * 2 + 4, 24)   # [orig | gap | overlay]


def test_save_overlay_single_panel(tmp_path):
    img = np.zeros((10, 10, 3), dtype=np.float32)
    cam = np.ones((10, 10), dtype=np.float32)
    out = save_overlay(img, cam, tmp_path / "o.png", side_by_side=False)
    from PIL import Image

    with Image.open(out) as im:
        assert im.size == (10, 10)
