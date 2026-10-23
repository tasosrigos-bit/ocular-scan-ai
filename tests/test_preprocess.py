"""Image preprocessing: the frame fitting and the full pipeline on a synthetic scan."""
from __future__ import annotations

import numpy as np
import pytest

from ocular.classifier import preprocess as pp
from ocular.classifier.data import PreConfig
from ocular.rag import tools


@pytest.fixture
def wide():
    """A 100 x 300 ramp, three times wider than tall."""
    return np.tile(np.linspace(0, 255, 300), (100, 1)).astype(np.float32)


def test_every_fit_produces_the_target_frame(wide):
    for fit in pp.FITS:
        assert pp.fit_frame(wide, 384, 256, fit).shape == (256, 384)


def test_letterbox_keeps_the_aspect_ratio_and_pads(wide):
    out = pp.letterbox(wide, 384, 256, fill=0)
    filled_rows = np.flatnonzero(out.max(axis=1) > 0)
    # 300 x 100 scaled to width 384 is 128 rows tall, centred in the 256-row frame.
    assert len(filled_rows) == 128
    assert filled_rows[0] == 64


def test_unknown_fit_is_rejected(wide):
    with pytest.raises(ValueError, match="unknown fit"):
        pp.fit_frame(wide, 384, 256, "stretch")


def test_stretch_maps_to_the_unit_range():
    g = np.random.default_rng(0).normal(100, 30, (50, 50))
    out = pp.stretch(g)
    assert out.min() == 0.0 and out.max() == 1.0


@pytest.mark.parametrize("cfg", [
    tools.PREPROCESS,
    PreConfig(384, 256, crop=False, curvature=False),
    PreConfig(256, 256, crop=True, curvature=False, fit="letterbox"),
])
def test_pipeline_output_matches_what_the_model_expects(synthetic_scan, cfg):
    out = cfg.apply(synthetic_scan)
    assert out.shape == (cfg.out_h, cfg.out_w)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_the_model_of_record_uses_the_selected_preprocessing():
    assert tools.PREPROCESS == PreConfig(384, 256, crop=True, curvature=True, fit="squish")
