import pytest
import pandas as pd
from src.processor import (
    remove_trailing_uk,
    to_group_code,
    clean_file_with_brands,
    is_excel,
    is_csv
)

def test_remove_trailing_uk():
    assert remove_trailing_uk("Nike UK") == "Nike"
    assert remove_trailing_uk("Adidas-uk.") == "Adidas"
    assert remove_trailing_uk("Puma_UK") == "Puma"
    assert remove_trailing_uk("Reebok") == "Reebok"
    assert remove_trailing_uk(None) == ""

def test_to_group_code():
    assert to_group_code("Nike") == "tuck_nike"
    assert to_group_code("H&M") == "tuck_hm"
    assert to_group_code("Levi's") == "tuck_levis"
    assert to_group_code(None) == "tuck_"

def test_is_excel():
    assert is_excel("data.xlsx") is True
    assert is_excel("data.xls") is True
    assert is_excel("data.csv") is False

def test_is_csv():
    assert is_csv("data.csv") is True
    assert is_csv("data.txt") is True
    assert is_csv("data.xlsx") is False
