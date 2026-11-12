"""Helpers of the Abt-Buy evaluation that do not need the dataset or the embedding model."""
import numpy as np
import pytest

from eval.product_llm import estimate_cost, parse_answer, select_borderline
from eval.product_rules import extract_model_numbers, make_record, model_signal, rule_score
from eval.run_product_eval import gold_rank, split_ids, top1


def test_model_numbers_are_letter_digit_codes():
    assert extract_model_numbers("Sony Turntable - PSLX350H") == {"pslx350h"}
    assert extract_model_numbers("Pioneer DEH-P3000IB Car Audio Player") == {"dehp3000ib"}
    assert extract_model_numbers("Pioneer PDP-5020FD 50' Plasma TV") == {"pdp5020fd"}


def test_units_and_specs_are_not_model_numbers():
    assert extract_model_numbers("Samsung 1080p 2.4GHz 32GB 5.1ch 200W 8MB") == set()
    assert extract_model_numbers("LG 5.6 cu.ft. Freestanding Electric Range") == set()
    assert extract_model_numbers("") == set()


def test_model_signal_agree_conflict_unknown():
    a = make_record(1, "Sony Turntable - PSLX350H")
    assert model_signal(a, make_record(2, "Sony PS-LX350H Belt-Drive Turntable")) == 1
    assert model_signal(a, make_record(3, "Sony PS-LX250H Turntable")) == -1
    assert model_signal(a, make_record(4, "Sony Belt Drive Turntable")) == 0


def test_code_written_with_a_space_still_agrees():
    a = make_record(1, "Pioneer Car Audio - DEHP3000IB")
    b = make_record(2, "Pioneer DEH P3000IB Car Audio Player")
    assert model_signal(a, b) == 1


def test_rule_score_hand_computed():
    a = make_record(1, "Sony Turntable PSLX350H")
    b = make_record(2, "Sony Turntable PSLX350H")
    # agree (100), identical tokens (Dice 100), same first word (100)
    assert rule_score(a, b) == pytest.approx(0.50 * 100 + 0.35 * 100 + 0.15 * 100)
    c = make_record(3, "Bose Speaker")
    # no code on c (40), no shared tokens (0), different first word (0)
    assert rule_score(a, c) == pytest.approx(0.50 * 40)


def test_split_is_deterministic_and_disjoint():
    ids = list(range(100, 200))
    v1, t1 = split_ids(ids)
    v2, t2 = split_ids(list(reversed(ids)))
    assert (v1, t1) == (v2, t2)
    assert len(v1) == 30 and len(t1) == 70
    assert not set(v1) & set(t1) and set(v1) | set(t1) == set(ids)


def test_top1_and_gold_rank():
    m = np.array([[0.1, 0.9, 0.5], [0.7, 0.7, 0.2]])
    idx, sc = top1(m)
    assert idx.tolist() == [1, 0]            # tie goes to the lower index
    assert sc.tolist() == [0.9, 0.7]
    assert gold_rank(m[0], [2]) == 2
    assert gold_rank(m[0], [0, 2]) == 2      # best gold counts
    assert gold_rank(m[0], [1]) == 1


def test_borderline_selection_is_capped_and_closest_first():
    scores = [40, 53, 48, 51, 90, 49]
    # distances to 50: 10, 3, 2, 1, 40, 1 -> inside margin 5: rows 1, 2, 3, 5
    # closest first, ties by index: 3, 5, 2, 1 -> capped at 3
    assert select_borderline(scores, 50, max_calls=3, margin=5) == [3, 5, 2]
    assert select_borderline(scores, 50, max_calls=0, margin=5) == []


def test_llm_answer_parsing():
    assert parse_answer("YES") is True
    assert parse_answer(" no.") is False
    assert parse_answer("Maybe") is None
    assert parse_answer("") is None


def test_cost_estimate_uses_price_table():
    tin, tout, cost = estimate_cost([("a" * 40, "b" * 40)], "claude-haiku-5-5")
    assert tin == 120 + 20 and tout == 400
    assert cost == pytest.approx((140 * 0.10 + 400 * 0.50) / 1_000_000)
    assert estimate_cost([("a", "b")], "unknown-model")[2] is None
