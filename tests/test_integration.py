"""End-to-end checks with the delivered checkpoint and a real clinic scan.

These need ``experiments/convnext_final_e1.pt`` and the clinic B-scans, which are
not in the repository, so they are marked ``slow``, skipped when the files are
missing, and run locally with ``uv run pytest -m slow``.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ocular import config, service
from ocular.classifier.data import CLINIC_DIR

SCAN = CLINIC_DIR / "DME_2__L.png"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (config.CKPT.exists() and SCAN.exists()),
        reason="needs the delivered checkpoint and the clinic B-scans",
    ),
]


def test_the_delivered_model_classifies_a_clinic_dme_scan():
    result = service.analyse(SCAN)
    assert result["label"] == "DME"
    assert sum(result["probs"].values()) == pytest.approx(1.0, abs=1e-4)
    assert result["scan"].shape == (256, 384)
    assert result["overlay"].shape == (256, 384, 3)
    assert 0.0 <= result["overlay"].min() and result["overlay"].max() <= 1.0
    assert not np.allclose(result["overlay"], result["overlay"][0, 0])


def test_predict_endpoint_end_to_end(monkeypatch, tmp_path):
    from app import api

    monkeypatch.setattr(api, "UPLOADS", tmp_path)
    with TestClient(api.app) as client:  # runs the lifespan, loading the real model
        res = client.post("/api/predict", files={"file": ("scan.png", SCAN.read_bytes(), "image/png")})
    assert res.status_code == 200
    assert res.json()["label"] == "DME"
