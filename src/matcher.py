"""
matcher.py

Contains all fuzzy matching, tokenization, and scoring logic for brand names.
Provides robust algorithms for matching messy brand names against a clean catalog.
"""
import re
import unicodedata
from typing import List, Tuple

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False
    import difflib

ALNUM_REGEX = re.compile(r"[^0-9A-Za-z]+")
WORD_RE = re.compile(r"[0-9A-Za-z]+")

def strip_accents(s: str) -> str:
    """Removes diacritics/accents from a string."""
    if s is None:
        return ""
    normalized = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))

def normalize_alnum(s: str) -> str:
    """Strips accents, non-alphanumeric characters, and converts to lowercase."""
    if s is None:
        return ""
    s = strip_accents(s)
    return ALNUM_REGEX.sub("", s).lower().strip()

def tokens(s: str) -> List[str]:
    """Extracts alphanumeric words from a string as lowercase tokens."""
    if s is None:
        return []
    return [strip_accents(t).lower() for t in WORD_RE.findall(str(s))]

def strong_tokens(s: str) -> List[str]:
    """Returns tokens with a length of 3 or more."""
    return [t for t in tokens(s) if len(t) >= 3]

def leading_token_matches(value: str, candidate: str) -> int:
    """
    Count how many leading tokens match in order from the start.
    Token comparison is case/diacritic-insensitive, exact equality.
    """
    vt = tokens(value)
    ct = tokens(candidate)
    k = 0
    for i in range(min(len(vt), len(ct))):
        if vt[i] == ct[i]:
            k += 1
        else:
            break
    return k

def contiguous_span_index(value_tokens: List[str], cand_tokens: List[str]) -> int:
    """
    Return the starting index where candidate tokens appear as a contiguous subsequence
    within the value tokens. Return -1 if not found.
    """
    n, m = len(value_tokens), len(cand_tokens)
    if m == 0 or m > n:
        return -1
    for i in range(0, n - m + 1):
        if value_tokens[i : i + m] == cand_tokens:
            return i
    return -1

def common_prefix_len(a_norm: str, b_norm: str) -> int:
    """Returns the length of the common prefix between two strings."""
    n = min(len(a_norm), len(b_norm))
    i = 0
    while i < n and a_norm[i] == b_norm[i]:
        i += 1
    return i

def subseq_match_after_prefix(nv: str, nc: str, lcp: int) -> Tuple[int, int]:
    """
    After consuming the first lcp characters, try to match the rest of the candidate (nc)
    in order within the value (nv).
    Returns (matched_chars, segments_count).
    """
    j = lcp  # index in nv
    matched = 0
    segments = 0
    in_run = False
    prev_j = j - 1

    for i in range(lcp, len(nc)):
        found = False
        while j < len(nv):
            if nv[j] == nc[i]:
                if not in_run or j != prev_j + 1:
                    segments += 1
                    in_run = True
                matched += 1
                prev_j = j
                j += 1
                found = True
                break
            j += 1
        if not found:
            in_run = False
            break
    return matched, segments

def ratio(a: str, b: str) -> float:
    """Return a 0..100 similarity ratio on strings."""
    if HAVE_RAPIDFUZZ:
        return float(rf_fuzz.ratio(a, b))
    else:
        if not a and not b:
            return 100.0
        return float(difflib.SequenceMatcher(None, a, b).ratio() * 100.0)

def candidate_rank_and_score(value: str, candidate: str) -> Tuple[Tuple[int, int, float, int, float], float]:
    """
    Robust, token-anchored, prefix-first with hard-filter and span handling.
    Returns (ranking_key, composite_score).
    """
    nv = normalize_alnum(value)
    nc = normalize_alnum(candidate)

    vt = tokens(value)
    ct = tokens(candidate)

    strong_v = set(strong_tokens(value))
    strong_c = set(strong_tokens(candidate))
    span_idx = contiguous_span_index(vt, ct)
    prefix_relation = nv.startswith(nc) or nc.startswith(nv)
    
    if not (strong_v & strong_c) and not prefix_relation and span_idx == -1:
        return (0, 0, 0.0, 0, 0.0), 0.0

    lcp_std = common_prefix_len(nv, nc)
    span_case = span_idx >= 0 and not nv.startswith(nc)
    effective_lcp = len(nc) if span_case else lcp_std

    if span_case:
        matched_after, segments_after = 0, 0
        coverage_pct = 100.0
        lead_tok = len(ct)
    else:
        matched_after, segments_after = subseq_match_after_prefix(nv, nc, effective_lcp)
        coverage_pct = ((effective_lcp + matched_after) / max(1, len(nc))) * 100.0
        lead_tok = leading_token_matches(value, candidate)

    global_ratio = ratio(nv, nc)

    composite = (
        0.65 * (effective_lcp / max(1, len(nc))) * 100.0
        + 0.20 * global_ratio
        + 0.15 * coverage_pct
        - 10.0 * max(0, segments_after - 1)
        - (30.0 if (effective_lcp == 0 and not span_case) else 0.0)
    )
    composite = max(0.0, min(100.0, composite))

    key = (effective_lcp, lead_tok, coverage_pct, len(nc), global_ratio)
    return key, composite

def best_match(value: str, clean_list: List[str]) -> Tuple[str, float]:
    """
    Finds the best matching candidate from the clean_list for a given value.
    Returns (best_candidate, score).
    """
    if not clean_list:
        return value, 0.0

    v_norm = normalize_alnum(value)
    for cl in clean_list:
        if normalize_alnum(cl) == v_norm and v_norm:
            return cl, 100.0

    best_cand = None
    best_key = None
    best_score = -1.0

    for cand in clean_list:
        key, score = candidate_rank_and_score(value, cand)
        if (best_key is None) or (key > best_key) or (key == best_key and score > best_score):
            best_key = key
            best_score = score
            best_cand = cand

    return (best_cand if best_cand is not None else value), (best_score if best_score >= 0 else 0.0)

def lcs_length(a: List[str], b: List[str]) -> int:
    """Word-level Longest Common Subsequence length."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]

def subseq_match_tokens(value_tokens: List[str], cand_tokens: List[str]) -> Tuple[int, int]:
    """
    Greedy word-level subsequence match.
    Returns (matched_tokens_count, segments_count).
    """
    i, j = 0, 0
    matched = 0
    segments = 0
    prev_pos = -2
    while i < len(value_tokens) and j < len(cand_tokens):
        if value_tokens[i] == cand_tokens[j]:
            matched += 1
            if i != prev_pos + 1:
                segments += 1
            prev_pos = i
            i += 1
            j += 1
        else:
            i += 1
    return matched, segments

def expand_tokens_for_similarity(a_tokens: List[str], b_tokens: List[str]) -> Tuple[List[str], List[str]]:
    """
    Expand fused tokens when they equal concatenation of adjacent tokens on the other side.
    Example: 'giftcard' <-> ['gift','card'].
    """
    def expand(base: List[str], other: List[str]) -> List[str]:
        out = list(base)
        other_pairs = {other[i] + other[i + 1] for i in range(len(other) - 1)}
        for t in base:
            if t in other_pairs:
                for i in range(len(other) - 1):
                    if t == other[i] + other[i + 1]:
                        out.append(other[i])
                        out.append(other[i + 1])
                        break
        return out

    a_exp = expand(a_tokens, b_tokens)
    b_exp = expand(b_tokens, a_tokens)
    return a_exp, b_exp

def dice_coefficient(a_tokens: List[str], b_tokens: List[str]) -> float:
    """Dice coefficient on token sets (0..100)."""
    sa, sb = set(a_tokens), set(b_tokens)
    if not sa and not sb:
        return 100.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return (2.0 * inter / (len(sa) + len(sb))) * 100.0

def enhanced_pair_score_words(value: str, suggestion: str, base_score: float) -> float:
    """
    Pure word-based boost for scoring similarity.
    Calculates improved score, returns max of base_score and boosted score.
    """
    vt = tokens(value)
    ct = tokens(suggestion)

    vtx, ctx = expand_tokens_for_similarity(vt, ct)

    token_dice = dice_coefficient(vtx, ctx)

    lead = 0
    for i in range(min(len(vt), len(ct))):
        if vt[i] == ct[i]:
            lead += 1
        else:
            break
    lead_cov = (lead / max(1, len(ct))) * 100.0

    lcs = lcs_length(vt, ct)
    lcs_cov = (lcs / max(1, len(ct))) * 100.0

    matched_subseq, segments = subseq_match_tokens(vt, ct)

    boosted = 0.4 * token_dice + 0.3 * lead_cov + 0.3 * lcs_cov

    if segments > 1:
        boosted -= 8.0 * (segments - 1)
    if lead == 0:
        boosted -= 15.0

    boosted = max(0.0, min(100.0, boosted))
    return max(float(base_score), float(boosted))
