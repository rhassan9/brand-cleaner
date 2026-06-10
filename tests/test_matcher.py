import pytest
from src.matcher import (
    strip_accents,
    normalize_alnum,
    tokens,
    strong_tokens,
    leading_token_matches,
    common_prefix_len,
    best_match
)

def test_strip_accents():
    assert strip_accents("Caffè") == "Caffe"
    assert strip_accents("Müller") == "Muller"
    assert strip_accents(None) == ""

def test_normalize_alnum():
    assert normalize_alnum("Caffè! @Brand") == "caffebrand"
    assert normalize_alnum("Müller-UK") == "mulleruk"
    assert normalize_alnum(None) == ""

def test_tokens():
    assert tokens("Nike UK Ltd.") == ["nike", "uk", "ltd"]
    assert tokens("H&M") == ["h", "m"]

def test_strong_tokens():
    assert strong_tokens("Nike UK Ltd.") == ["nike", "ltd"]
    assert strong_tokens("H&M") == []

def test_leading_token_matches():
    assert leading_token_matches("Nike Shoes", "Nike Shoes UK") == 2
    assert leading_token_matches("Nike UK", "Adidas UK") == 0

def test_common_prefix_len():
    assert common_prefix_len("nikeshoes", "nike") == 4
    assert common_prefix_len("adidas", "nike") == 0

def test_best_match():
    clean_list = ["Nike", "Adidas", "Puma"]
    
    # Exact match
    match, score = best_match("Nike", clean_list)
    assert match == "Nike"
    assert score == 100.0

    # Match with different casing and special chars
    match, score = best_match("  nIkE!!!  ", clean_list)
    assert match == "Nike"
    assert score == 100.0

    # Partial match
    match, score = best_match("Adida", clean_list)
    assert match == "Adidas"
    assert score > 50.0

    # No match
    match, score = best_match("Reebok", clean_list)
    assert score == 0.0
