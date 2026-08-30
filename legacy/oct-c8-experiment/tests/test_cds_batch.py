from oct_cds.cds.batch import summarize_recommendations


def _row(true, pred, deferred, urgency="soon", abstained=None, ood=False):
    if abstained is None:
        abstained = deferred and not ood
    return {
        "true": true, "pred": pred, "correct": true == pred,
        "abstained": abstained, "ood_rejected": ood,
        "deferred_to_specialist": deferred, "urgency": urgency,
    }


def test_headline_misclassified_breakdown():
    rows = [
        _row("Drusen", "Drusen", deferred=False, urgency="routine"),   # correct, confident
        _row("Drusen", "AMD", deferred=True, urgency="soon"),          # wrong -> deferred (good)
        _row("Drusen", "CNV", deferred=False, urgency="urgent"),       # wrong -> confident urgent (bad)
        _row("CNV", "CNV", deferred=False, urgency="urgent"),          # correct
        _row("DME", "AMD", deferred=False, urgency="soon"),            # wrong -> confident (bad)
    ]
    s = summarize_recommendations(rows, "external_test")

    assert s["n_images"] == 5
    assert s["model_accuracy"] == 0.4
    m = s["on_misclassified"]
    assert m["n"] == 3
    assert m["deferred_to_specialist"] == 1
    assert m["confident_call"] == 2
    assert m["confident_call_urgencies"] == {"urgent": 1, "soon": 1}

    by_cls = s["misclassified_by_true_class"]
    assert set(by_cls) == {"Drusen", "DME"}
    assert by_cls["Drusen"]["n"] == 2
    assert by_cls["Drusen"]["deferred_to_specialist"] == 1
    assert by_cls["Drusen"]["confident_call"] == 1


def test_rates_and_correct_overcaution():
    rows = [
        _row("Normal", "Normal", deferred=True, abstained=True),   # correct but abstained
        _row("Normal", "Normal", deferred=False),
        _row("CNV", "AMD", deferred=True, ood=True),               # wrong, ood-rejected
    ]
    s = summarize_recommendations(rows)
    assert s["abstention_rate"] == round(1 / 3, 4)
    assert s["ood_reject_rate"] == round(1 / 3, 4)
    assert s["on_correct"]["deferred_to_specialist"] == 1     # over-cautious count
    assert s["on_misclassified"]["deferred_to_specialist"] == 1


def test_empty():
    s = summarize_recommendations([], "test")
    assert s["n_images"] == 0
    assert s["model_accuracy"] is None
    assert s["on_misclassified"]["n"] == 0
