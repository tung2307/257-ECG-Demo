
import pytest

from scripts.evaluate_predictions import evaluate_rows


def test_perfect_predictions():
    rows = [
        {"true_label": "Normal", "pred_label": "Normal"},
        {"true_label": "AFib", "pred_label": "AFib"},
        {"true_label": "Other", "pred_label": "Other"},
        {"true_label": "Noisy", "pred_label": "Noisy"},
    ]

    metrics = evaluate_rows(rows)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["paper_style_f1"] == 1.0
    assert metrics["afib_recall"] == 1.0
    assert metrics["afib_f1"] == 1.0


def test_partial_afib_recall():
    rows = [
        {"true_label": "AFib", "pred_label": "AFib"},
        {"true_label": "AFib", "pred_label": "Normal"},
        {"true_label": "Normal", "pred_label": "Normal"},
    ]

    metrics = evaluate_rows(rows)

    assert metrics["afib_recall"] == 0.5
    assert metrics["afib_f1"] == pytest.approx(2 / 3)


def test_unknown_label():
    rows = [
        {"true_label": "Unknown", "pred_label": "Normal"}
    ]

    with pytest.raises(ValueError):
        evaluate_rows(rows)


def test_empty_predictions():
    with pytest.raises(ValueError):
        evaluate_rows([])
