"""The synthetic brand set and the brand evaluation runner."""
import pandas as pd

from eval import make_synthetic_brands as gen
from eval.metrics import ABSTAIN_CORRECT, CORRECT_MATCH, WRONG_MATCH, evaluate
from eval.run_brand_eval import SYN_EVAL, SYN_MASTER, load_labels, load_master, run_pipeline


def test_committed_synthetic_csvs_match_the_generator():
    assert gen.main(["--check"]) == 0


def test_synthetic_set_is_well_formed():
    labels = load_labels(SYN_EVAL)
    master = load_master(SYN_MASTER)
    assert 150 <= len(labels) <= 300
    assert labels["raw_brand"].is_unique
    assert set(t for t in labels["true_master"] if t) <= set(master)
    negatives = labels[labels["case_type"].isin(["lookalike_negative", "not_in_master"])]
    assert (negatives["true_master"] == "").all()


def test_runner_uses_the_real_pipeline_and_classifies_rows():
    labels = pd.DataFrame({
        "raw_brand": ["Adidas UK", "Booths", "Office Depot", "Reebok"],
        "true_master": ["Adidas", "", "", ""],
        "case_type": ["suffix", "lookalike_negative", "lookalike_negative", "not_in_master"],
    })
    pred = run_pipeline(labels, ["Adidas", "Boots", "Office"], threshold=80)
    outcomes, _ = evaluate([t or None for t in labels["true_master"]], pred["predicted"])
    assert outcomes[0] == CORRECT_MATCH
    assert outcomes[1] == ABSTAIN_CORRECT       # Booths is left alone
    assert outcomes[2] == WRONG_MATCH           # known weakness: prefix collision
    assert outcomes[3] == ABSTAIN_CORRECT
