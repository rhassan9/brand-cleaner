"""
processor.py

Contains functions to read/write files and apply brand cleaning logic to datasets.
Uses pandas for data manipulation.
"""
import os
import re
from typing import List, Tuple, Dict, Optional

import pandas as pd

from src.matcher import best_match, enhanced_pair_score_words, normalize_alnum

DEFAULT_SIMILARITY = 80
UK_SUFFIX_REGEX = re.compile(r"[\s\-_]*uk\b\.?$", re.IGNORECASE)


def is_excel(path: str) -> bool:
    """Checks if a file path points to an Excel file."""
    return os.path.splitext(path)[1].lower() in [".xlsx", ".xlsm", ".xls", ".xlsb"]


def is_csv(path: str) -> bool:
    """Checks if a file path points to a CSV/TXT file."""
    return os.path.splitext(path)[1].lower() in [".csv", ".txt"]


def read_single_column_series(path: str) -> pd.Series:
    """Reads the first column of an Excel or CSV file as a pandas Series."""
    if is_excel(path):
        df = pd.read_excel(path, engine="openpyxl") if path.lower().endswith(".xlsx") else pd.read_excel(path)
    elif is_csv(path):
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file type. Please provide an Excel or CSV file.")
    
    if df.shape[1] < 1:
        raise ValueError("The provided file appears to have no columns.")
    
    s = df[df.columns[0]]
    s = s.dropna()
    s = s.astype(str).map(lambda x: x.strip()).replace("", pd.NA).dropna()
    return s


def write_single_column_series(path: str, series: pd.Series, header: Optional[str] = None):
    """Writes a pandas Series to an Excel or CSV file."""
    header_name = header if header else (series.name if series.name else "brand")
    df_out = pd.DataFrame({header_name: series})
    if is_excel(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".xlsx":
            df_out.to_excel(path, index=False, engine="openpyxl")
        else:
            df_out.to_excel(path, index=False)
    elif is_csv(path):
        df_out.to_csv(path, index=False)
    else:
        raise ValueError("Unsupported file type for writing.")


def load_clean_brands(path: str) -> List[str]:
    """Loads a list of clean brands from a file, deduplicating them case-insensitively."""
    s = read_single_column_series(path)
    seen = set()
    result = []
    for val in s:
        if not isinstance(val, str):
            val = str(val)
        key = val.casefold().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(val.strip())
    return result


def remove_trailing_uk(value: str) -> str:
    """Removes a trailing 'uk' or 'UK' from a string."""
    if not isinstance(value, str):
        value = "" if pd.isna(value) else str(value)
    return UK_SUFFIX_REGEX.sub("", value).strip()


def to_group_code(value: str) -> str:
    """Converts a brand string to a standardized group code format: tuck_<brand>"""
    if not isinstance(value, str):
        value = "" if pd.isna(value) else str(value)
    base = normalize_alnum(value)
    return f"tuck_{base}" if base else "tuck_"


def apply_enhanced_scores_and_update(df: pd.DataFrame,
                                     brand_col: str,
                                     mapping: Dict[str, Tuple[str, float, str]],
                                     threshold: int) -> Dict[str, Tuple[str, float, str]]:
    """
    Applies the word-based enhanced scoring to the mappings and updates the DataFrame.
    """
    updated: Dict[str, Tuple[str, float, str]] = {}
    for key, (final, score, sugg) in mapping.items():
        new_score = enhanced_pair_score_words(key, sugg, score)
        new_final = final
        if final == key and new_score >= float(threshold):
            new_final = sugg
        updated[key] = (new_final, new_score, sugg)

    def apply_map_updated(x: str) -> str:
        k = "" if pd.isna(x) else str(x).strip()
        k = remove_trailing_uk(k)
        if k in updated:
            return updated[k][0]
        return k

    df[brand_col] = df[brand_col].map(apply_map_updated)
    return updated


def clean_file_with_brands(
    clean_brand_path: str,
    file_to_clean_path: str,
    min_similarity: int = DEFAULT_SIMILARITY,
    keep_original: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, float, str]]]:
    """
    Cleans a file containing brand names using a clean brand catalog.
    Returns the cleaned DataFrame and a mapping of the changes.
    """
    clean_list = load_clean_brands(clean_brand_path)

    # Load target file
    if is_excel(file_to_clean_path):
        df = pd.read_excel(file_to_clean_path, engine="openpyxl") if file_to_clean_path.lower().endswith(".xlsx") else pd.read_excel(file_to_clean_path)
    elif is_csv(file_to_clean_path):
        df = pd.read_csv(file_to_clean_path)
    else:
        raise ValueError("Unsupported file type to clean. Provide an Excel or CSV file.")

    if df.shape[1] < 1:
        raise ValueError("The file to clean appears to have no columns.")
    brand_col = df.columns[0]

    # Save original for optional column
    original_series = df[brand_col].copy()

    # Pre-clean: remove trailing 'uk'
    cleaned_series = df[brand_col].astype(str).map(lambda x: x.strip()).map(remove_trailing_uk)

    # Build initial mapping using the ranking logic
    unique_vals = pd.unique(cleaned_series.fillna("").astype(str))
    mapping: Dict[str, Tuple[str, float, str]] = {}
    for v in unique_vals:
        suggestion, score = best_match(v, clean_list)
        final_val = suggestion if score >= float(min_similarity) else v
        mapping[v] = (final_val, score, suggestion)

    # Apply initial mapping
    def apply_map(x: str) -> str:
        key = "" if pd.isna(x) else str(x).strip()
        key = remove_trailing_uk(key)
        if key in mapping:
            return mapping[key][0]
        return key

    df[brand_col] = df[brand_col].map(apply_map)

    # Insert Original Brand if requested
    if keep_original:
        df.insert(0, "Original Brand", original_series)

    # Post-processing: boost scores
    mapping = apply_enhanced_scores_and_update(df, brand_col, mapping, threshold=min_similarity)

    # Group code right after Brand column
    group_series = df[brand_col].map(to_group_code)
    if "Group code" in df.columns:
        df.drop(columns=["Group code"], inplace=True)
    brand_idx = df.columns.get_loc(brand_col)
    df.insert(brand_idx + 1, "Group code", group_series)

    return df, mapping
