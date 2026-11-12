"""Metric functions in eval/metrics.py, checked against small hand-computed cases."""
import math

import pytest

from eval.metrics import (
    ABSTAIN_CORRECT,
    ABSTAIN_MISSED,
    CORRECT_MATCH,
    WRONG_MATCH,
    apply_threshold,
    classify,
    compute_metrics,
    evaluate,
    f1_score,
    fmt_ratio,
    gold_set,
    metric_cells,
    metrics_by_group,
    pick_threshold,
    sweep_thresholds,
)


def test_classify_covers_all_four_outcomes():
    assert classify("Boots", "Boots") == CORRECT_MATCH
    assert classify("Boots", "Booths") == WRONG_MATCH
    assert classify(None, "Boots") == WRONG_MATCH        # matched something that has no master
    assert classify(None, None) == ABSTAIN_CORRECT
    assert classify("Boots", None) == ABSTAIN_MISSED


def test_empty_string_and_nan_mean_no_true_master():
    assert gold_set("") == set()
    assert gold_set("   ") == set()
    assert gold_set(float("nan")) == set()
    assert classify("", None) == ABSTAIN_CORRECT
    assert classify(float("nan"), "X") == WRONG_MATCH


def test_nan_prediction_is_an_abstain_not_a_match():
    # pandas can turn None into NaN in a column of predictions.
    assert classify("Boots", float("nan")) == ABSTAIN_MISSED
    assert classify(None, float("nan")) == ABSTAIN_CORRECT


def test_several_acceptable_answers():
    # Abt-Buy: one Abt product can have more than one correct Buy product.
    assert classify({101, 102}, 102) == CORRECT_MATCH
    assert classify({101, 102}, 103) == WRONG_MATCH
    assert classify(set(), None) == ABSTAIN_CORRECT


def test_mixed_case_hand_computed():
    golds = ["A", "A", "B", None, None, "C"]
    preds = ["A", "B", None, None, "X", "C"]
    outcomes, m = evaluate(golds, preds)
    assert outcomes == [CORRECT_MATCH, WRONG_MATCH, ABSTAIN_MISSED,
                        ABSTAIN_CORRECT, WRONG_MATCH, CORRECT_MATCH]
    assert (m[CORRECT_MATCH], m[WRONG_MATCH], m[ABSTAIN_CORRECT], m[ABSTAIN_MISSED]) == (2, 2, 1, 1)
    assert m["n"] == 6 and m["n_with_true"] == 4
    assert m["precision"] == pytest.approx(2 / 4)
    assert m["recall"] == pytest.approx(2 / 4)
    assert m["f1"] == pytest.approx(0.5)
    assert m["abstain_rate"] == pytest.approx(2 / 6)


def test_high_precision_low_recall():
    _, m = evaluate(["A", "B", "C"], ["A", None, None])
    assert m["precision"] == 1.0
    assert m["recall"] == pytest.approx(1 / 3)
    # 2 * 1 * (1/3) / (1 + 1/3) = 0.5
    assert m["f1"] == pytest.approx(0.5)


def test_undefined_ratios_are_none_not_zero():
    # Tool abstains everywhere: precision is 0/0.
    _, m = evaluate(["A", "B"], [None, None])
    assert m["precision"] is None
    assert m["recall"] == 0.0
    assert m["f1"] is None
    assert m["abstain_rate"] == 1.0

    # No item has a true answer: recall is 0/0.
    _, m = evaluate([None, None], [None, None])
    assert m["recall"] is None and m["precision"] is None
    _, m = evaluate([], [])
    assert m["abstain_rate"] is None


def test_f1_edge_cases():
    assert f1_score(0.0, 0.0) == 0.0
    assert f1_score(None, 0.5) is None
    assert f1_score(0.5, 1.0) == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_compute_metrics_rejects_inconsistent_input():
    with pytest.raises(ValueError):
        compute_metrics([ABSTAIN_MISSED], [False])
    with pytest.raises(ValueError):
        compute_metrics([CORRECT_MATCH], [False])
    with pytest.raises(ValueError):
        compute_metrics(["bogus"], [True])
    with pytest.raises(ValueError):
        compute_metrics([CORRECT_MATCH], [True, False])


def test_metrics_by_group():
    groups = ["typo", "typo", "lookalike_negative", "lookalike_negative"]
    golds = ["Adidas", "Sainsburys", None, None]
    preds = ["Adidas", None, "Boots", None]
    by = metrics_by_group(groups, golds, preds)
    assert list(by) == ["lookalike_negative", "typo"]
    assert by["typo"]["precision"] == 1.0 and by["typo"]["recall"] == 0.5
    neg = by["lookalike_negative"]
    assert neg[WRONG_MATCH] == 1 and neg[ABSTAIN_CORRECT] == 1
    assert neg["precision"] == 0.0 and neg["recall"] is None


def test_apply_threshold_is_inclusive():
    assert apply_threshold(["A", "B", None], [80.0, 79.99, 100.0], 80) == ["A", None, None]


def test_sweep_and_pick_threshold_hand_computed():
    golds = ["A", "X", "C", None]
    cands = ["A", "B", "C", "D"]
    scores = [90, 80, 70, 60]
    sweep = dict(sweep_thresholds(golds, cands, scores, [55, 65, 75, 85]))
    # t=85: A only            -> P 1/1, R 1/3, F1 0.5
    # t=75: A, B              -> P 1/2, R 1/3, F1 0.4
    # t=65: A, B, C           -> P 2/3, R 2/3, F1 2/3
    # t=55: A, B, C, D        -> P 2/4, R 2/3, F1 4/7
    assert sweep[85]["f1"] == pytest.approx(0.5)
    assert sweep[75]["f1"] == pytest.approx(0.4)
    assert sweep[65]["f1"] == pytest.approx(2 / 3)
    assert sweep[55]["f1"] == pytest.approx(4 / 7)
    t, m = pick_threshold(list(sweep.items()))
    assert t == 65 and m["precision"] == pytest.approx(2 / 3)


def test_pick_threshold_tie_breaks():
    a = {"f1": 0.5, "precision": 0.5}
    b = {"f1": 0.5, "precision": 0.8}
    assert pick_threshold([(10, a), (20, b)])[0] == 20      # higher precision wins
    assert pick_threshold([(30, a), (10, a)])[0] == 10      # then lower threshold
    assert pick_threshold([(10, {"f1": None, "precision": None}), (20, a)])[0] == 20


def test_formatting_keeps_exact_fraction():
    assert fmt_ratio(2 / 3, 2, 3) == "0.6667 (2/3)"
    assert fmt_ratio(None, 0, 0) == "n/a (0/0)"
    assert fmt_ratio(None) == "n/a"
    _, m = evaluate(["A", "B", None], ["A", "C", None])
    cells = metric_cells(m)
    assert cells["precision"] == "0.5000 (1/2)"
    assert cells["recall"] == "0.5000 (1/2)"
    assert cells["abstain_rate"] == "0.3333 (1/3)"
    assert not math.isnan(m["f1"])
