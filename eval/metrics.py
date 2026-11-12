"""
Match-quality metrics shared by the brand and product evaluations.

Every evaluated item gets one of four outcomes:

    correct_match    the tool proposed an answer and it is one of the correct ones
    wrong_match      the tool proposed an answer that is not correct
                     (including any answer when no correct answer exists)
    abstain_correct  no correct answer exists and the tool proposed nothing
    abstain_missed   a correct answer exists but the tool proposed nothing

From these:

    precision    = correct_match / (correct_match + wrong_match)
    recall       = correct_match / items that have a correct answer
    F1           = harmonic mean of precision and recall
    abstain rate = (abstain_correct + abstain_missed) / all items

A ratio with a zero denominator is undefined and returned as None (printed
"n/a"), never silently turned into 0 or 1. F1 is None when precision or recall
is None, and 0.0 when both are 0.
"""
from collections import OrderedDict
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

CORRECT_MATCH = "correct_match"
WRONG_MATCH = "wrong_match"
ABSTAIN_CORRECT = "abstain_correct"
ABSTAIN_MISSED = "abstain_missed"
OUTCOMES = (CORRECT_MATCH, WRONG_MATCH, ABSTAIN_CORRECT, ABSTAIN_MISSED)


def gold_set(gold) -> Set[Hashable]:
    """
    Normalise a gold label to a set of acceptable answers.
    None, NaN and "" mean "no correct answer". A string or number is one answer.
    Any other iterable is a set of acceptable answers.
    """
    if gold is None:
        return set()
    if isinstance(gold, float) and gold != gold:  # NaN
        return set()
    if isinstance(gold, str):
        return {gold} if gold.strip() else set()
    if isinstance(gold, (int, float)):
        return {gold}
    return {g for g in gold if not (g is None or (isinstance(g, str) and not g.strip()))}


def classify(gold, predicted) -> str:
    """
    Return the outcome for one item. predicted is None when the tool abstained.
    NaN also counts as an abstain, because pandas turns None into NaN in some columns.
    """
    answers = gold_set(gold)
    if predicted is None or (isinstance(predicted, float) and predicted != predicted):
        return ABSTAIN_MISSED if answers else ABSTAIN_CORRECT
    return CORRECT_MATCH if predicted in answers else WRONG_MATCH


def _ratio(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def f1_score(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(outcomes: Sequence[str], has_true: Sequence[bool]) -> Dict[str, object]:
    """
    Aggregate outcomes into counts and ratios.
    has_true[i] says whether item i has at least one correct answer
    (needed because wrong_match can happen with or without one).
    """
    if len(outcomes) != len(has_true):
        raise ValueError("outcomes and has_true must have the same length")
    counts = {o: 0 for o in OUTCOMES}
    for o in outcomes:
        if o not in counts:
            raise ValueError(f"unknown outcome {o!r}")
        counts[o] += 1
    for o, t in zip(outcomes, has_true):
        if o == ABSTAIN_MISSED and not t:
            raise ValueError("abstain_missed on an item without a correct answer")
        if o == ABSTAIN_CORRECT and t:
            raise ValueError("abstain_correct on an item with a correct answer")
        if o == CORRECT_MATCH and not t:
            raise ValueError("correct_match on an item without a correct answer")
    n = len(outcomes)
    n_true = sum(1 for t in has_true if t)
    cm, wm = counts[CORRECT_MATCH], counts[WRONG_MATCH]
    abstain = counts[ABSTAIN_CORRECT] + counts[ABSTAIN_MISSED]
    precision = _ratio(cm, cm + wm)
    recall = _ratio(cm, n_true)
    return {
        "n": n,
        "n_with_true": n_true,
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "abstain_rate": _ratio(abstain, n),
    }


def evaluate(golds: Sequence, predictions: Sequence) -> Tuple[List[str], Dict[str, object]]:
    """Classify every item and aggregate. Returns (outcomes, metrics)."""
    if len(golds) != len(predictions):
        raise ValueError("golds and predictions must have the same length")
    outcomes = [classify(g, p) for g, p in zip(golds, predictions)]
    has_true = [bool(gold_set(g)) for g in golds]
    return outcomes, compute_metrics(outcomes, has_true)


def metrics_by_group(groups: Sequence[str], golds: Sequence, predictions: Sequence) -> "OrderedDict[str, Dict[str, object]]":
    """evaluate() separately for each group label, groups sorted by name."""
    idx: Dict[str, List[int]] = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    out: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
    for g in sorted(idx):
        ii = idx[g]
        _, m = evaluate([golds[i] for i in ii], [predictions[i] for i in ii])
        out[g] = m
    return out


def apply_threshold(candidates: Sequence, scores: Sequence[float], threshold: float) -> List:
    """Keep a candidate when its score is >= threshold, otherwise abstain (None)."""
    return [c if (c is not None and s >= threshold) else None for c, s in zip(candidates, scores)]


def sweep_thresholds(golds: Sequence, candidates: Sequence, scores: Sequence[float],
                     thresholds: Iterable[float]) -> List[Tuple[float, Dict[str, object]]]:
    """Metrics at every threshold, in the order given."""
    return [(t, evaluate(golds, apply_threshold(candidates, scores, t))[1]) for t in thresholds]


def pick_threshold(sweep: Sequence[Tuple[float, Dict[str, object]]]) -> Tuple[float, Dict[str, object]]:
    """
    Choose the threshold with the highest F1. Ties go to higher precision, then
    to the lower threshold. Undefined F1 counts as lower than any number.
    Only ever call this on tuning data, never on the test split.
    """
    if not sweep:
        raise ValueError("empty sweep")

    def key(item):
        t, m = item
        f1 = m["f1"] if m["f1"] is not None else -1.0
        p = m["precision"] if m["precision"] is not None else -1.0
        return (f1, p, -t)

    return max(sweep, key=key)


def fmt_ratio(value: Optional[float], num: Optional[int] = None, den: Optional[int] = None) -> str:
    """Format a ratio to 4 decimals, with the exact fraction when given. None is 'n/a'."""
    if value is None:
        return "n/a" if den is None else "n/a (0/0)"
    s = f"{value:.4f}"
    if num is not None and den is not None:
        s += f" ({num}/{den})"
    return s


def metric_cells(m: Dict[str, object]) -> Dict[str, str]:
    """Display strings for one metrics dict, each with its exact fraction."""
    cm, wm = m[CORRECT_MATCH], m[WRONG_MATCH]
    ab = m[ABSTAIN_CORRECT] + m[ABSTAIN_MISSED]
    return {
        "precision": fmt_ratio(m["precision"], cm, cm + wm),
        "recall": fmt_ratio(m["recall"], cm, m["n_with_true"]),
        "f1": fmt_ratio(m["f1"]),
        "abstain_rate": fmt_ratio(m["abstain_rate"], ab, m["n"]),
    }
