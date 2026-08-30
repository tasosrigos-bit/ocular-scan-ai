import numpy as np

from oct_cds.evaluation.metrics import (
    classification_report_dict,
    confusion_matrix_dict,
    expected_calibration_error,
    sensitivity_specificity,
)

CLASSES = ["AMD", "CNV", "CSR", "DME", "DR", "Drusen", "Macular Hole", "Normal"]


def _fake_preds(n_per_class=20, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.repeat(np.arange(8), n_per_class)
    y_pred = y_true.copy()
    # corrupt ~15% of predictions
    flip = rng.random(len(y_true)) < 0.15
    y_pred[flip] = rng.integers(0, 8, flip.sum())
    probs = np.full((len(y_true), 8), 0.02)
    probs[np.arange(len(y_true)), y_pred] = 0.86
    probs /= probs.sum(axis=1, keepdims=True)
    return y_true, y_pred, probs


def test_report_has_all_requested_metrics():
    y_true, y_pred, probs = _fake_preds()
    rep = classification_report_dict(y_true, y_pred, probs, class_names=CLASSES)

    for key in ("accuracy", "macro_f1", "quadratic_weighted_kappa",
                "auroc_macro", "auprc_macro", "ece", "balanced_accuracy"):
        assert key in rep
    assert set(rep["per_class"]) == set(CLASSES)
    for m in rep["per_class"].values():
        for key in ("sensitivity", "specificity", "precision", "f1", "auroc", "auprc", "support"):
            assert key in m
    assert 0.0 <= rep["accuracy"] <= 1.0
    assert rep["per_class"]["AMD"]["support"] == 20


def test_confusion_matrix_shape_and_diagonal():
    y_true, y_pred, _ = _fake_preds()
    cm = confusion_matrix_dict(y_true, y_pred, CLASSES)
    m = np.array(cm["matrix"])
    assert m.shape == (8, 8)
    assert m.sum() == len(y_true)
    assert np.trace(m) == int((y_true == y_pred).sum())
    assert cm["labels"] == CLASSES


def test_sensitivity_specificity_perfect():
    y = np.array([0, 1, 2, 3])
    sens, spec = sensitivity_specificity(y, y, 4)
    assert all(v == 1.0 for v in sens.values())
    assert all(v == 1.0 for v in spec.values())


def test_ece_zero_for_perfectly_calibrated():
    # confidence exactly matches accuracy in every bin -> ECE 0
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert expected_calibration_error(probs, y_true) == 0.0


def test_report_handles_absent_classes():
    # only 3 of 8 classes present (mimics the 37-image external set)
    y_true = np.array([1, 1, 3, 3, 7, 7])
    y_pred = np.array([1, 3, 3, 3, 7, 1])
    probs = np.full((6, 8), 0.02)
    probs[np.arange(6), y_pred] = 0.86
    probs /= probs.sum(axis=1, keepdims=True)
    rep = classification_report_dict(y_true, y_pred, probs, class_names=CLASSES)
    assert rep["per_class"]["AMD"]["support"] == 0
    assert np.isnan(rep["per_class"]["AMD"]["auroc"])
    assert np.array(confusion_matrix_dict(y_true, y_pred, CLASSES)["matrix"]).shape == (8, 8)
