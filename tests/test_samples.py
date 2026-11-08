"""End-to-end run on the synthetic files in samples/ (the same ones shown in the README)."""
import os

from src.processor import clean_file_with_brands

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def run_samples():
    df, _ = clean_file_with_brands(
        os.path.join(SAMPLES, "clean_brands.csv"),
        os.path.join(SAMPLES, "messy_catalog.csv"),
        min_similarity=80,
        keep_original=True,
    )
    return dict(zip(df["Original Brand"], df["brand"])), df


def test_all_adidas_variants_collapse_to_one_brand():
    result, _ = run_samples()
    for raw in ["adidas", "Adidas GB", "ADIDAS UK LTD", "adidas-uk", "Adidas (U.K.)", "Addidas"]:
        assert result[raw] == "Adidas", raw


def test_long_retail_names_reduce_to_the_brand():
    result, _ = run_samples()
    assert result["Great British Pub UK eGift Voucher"] == "Great British Pub"
    assert result["Global Hotel Card Powered by Expedia eGift"] == "Global Hotel Card"
    assert result["Sainsbuyrs"] == "Sainsburys"


def test_lookalikes_and_known_limits_are_left_unchanged():
    result, _ = run_samples()
    assert result["Booths"] == "Booths"  # not "Boots"
    # Known limitation: "and" vs "&" is not normalized, so this stays for review.
    assert result["Marks and Spencer E-Gift Voucher"] == "Marks and Spencer E-Gift Voucher"


def test_output_columns_and_other_data_are_preserved():
    _, df = run_samples()
    assert list(df.columns) == ["Original Brand", "brand", "Group code", "denomination"]
    assert df.loc[df["Original Brand"] == "Addidas", "Group code"].item() == "tuck_adidas"
    assert df["denomination"].sum() == 625
