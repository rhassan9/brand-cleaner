"""
Approach A: a rules-based product title matcher built from the brand matcher's ideas.

Reused from src/matcher.py (imported, not modified): normalize_alnum, tokens,
leading_token_matches and dice_coefficient. Added for product titles: model
number extraction, which is the strongest single signal in this domain.

Score (0 to 100) for an (Abt title, Buy title) pair:

    0.50 * model_signal   100 if a model number agrees, 40 if either side has
                          none (no evidence), 0 if both have codes and none agree
  + 0.35 * token Dice     word overlap on the two titles (0 to 100)
  + 0.15 * brand_lead     100 if the first word agrees (brand usually comes first)

The weights were set before looking at any results and are not tuned. Only
the match threshold is tuned, on the validation split.
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from src.matcher import dice_coefficient, normalize_alnum, tokens

# Split titles into candidate code tokens on whitespace and common separators,
# keeping hyphens and dots inside a token ("KDL-40V4100", "DEH-P3000IB").
SPLIT_RE = re.compile(r"[\s,;:()\[\]{}\"'/|+]+")
HAS_DIGIT = re.compile(r"\d")
HAS_ALPHA = re.compile(r"[a-z]")
# Measurements and specs that look like codes but are not ("8mb", "1080p", "50in").
UNIT_RE = re.compile(
    r"^\d+(?:x\d+)*(?:mm|cm|m|in|inch|inches|ft|gb|mb|tb|kb|ghz|mhz|hz|khz|w|watt|watts|"
    r"v|mah|lb|lbs|oz|p|i|x|k|cu|cuft|qt|ch|bit|mp|pk|pc|pcs|hr|hrs|min|rpm|db|th|nd|rd|"
    r"st|s|d|mbps|gbps|g)$"
)
MIN_CODE_LEN = 4
MIN_SUBSTRING_LEN = 5   # shorter codes must match exactly, not as a substring

W_MODEL, W_DICE, W_LEAD = 0.50, 0.35, 0.15
MODEL_AGREE, MODEL_UNKNOWN, MODEL_CONFLICT = 100.0, 40.0, 0.0


def extract_model_numbers(title: str) -> Set[str]:
    """
    Alphanumeric product codes in a title, normalised to lowercase letters and digits.
    A code has at least one letter and one digit, at least 4 characters, and is not
    a unit or spec such as "8MB" or "1080p".
    """
    out: Set[str] = set()
    if not title:
        return out
    for raw in SPLIT_RE.split(str(title)):
        norm = normalize_alnum(raw)
        if len(norm) < MIN_CODE_LEN:
            continue
        if not (HAS_DIGIT.search(norm) and HAS_ALPHA.search(norm)):
            continue
        if UNIT_RE.match(norm):
            continue
        out.add(norm)
    return out


@dataclass(frozen=True)
class Record:
    """Precomputed features for one product title."""
    id: object
    title: str
    alnum: str
    toks: Tuple[str, ...]
    tokset: frozenset
    codes: frozenset


def make_record(rec_id, title: str) -> Record:
    t = "" if title is None else str(title)
    toks = tuple(tokens(t))
    return Record(rec_id, t, normalize_alnum(t), toks, frozenset(toks), frozenset(extract_model_numbers(t)))


def model_signal(a: Record, b: Record) -> int:
    """+1 a code agrees, 0 no evidence, -1 both sides have codes and none agree."""
    if not a.codes or not b.codes:
        # A code on one side can still appear inside the other title written
        # differently (for example split by a space), so check substrings too.
        for c in a.codes:
            if len(c) >= MIN_SUBSTRING_LEN and c in b.alnum:
                return 1
        for c in b.codes:
            if len(c) >= MIN_SUBSTRING_LEN and c in a.alnum:
                return 1
        return 0
    if a.codes & b.codes:
        return 1
    for c in a.codes:
        if len(c) >= MIN_SUBSTRING_LEN and c in b.alnum:
            return 1
    for c in b.codes:
        if len(c) >= MIN_SUBSTRING_LEN and c in a.alnum:
            return 1
    return -1


def rule_score(a: Record, b: Record, signal: Optional[int] = None) -> float:
    s = model_signal(a, b) if signal is None else signal
    model_part = MODEL_AGREE if s > 0 else (MODEL_CONFLICT if s < 0 else MODEL_UNKNOWN)
    dice = dice_coefficient(list(a.tokset), list(b.tokset))
    lead = 100.0 if (a.toks and b.toks and a.toks[0] == b.toks[0]) else 0.0
    return W_MODEL * model_part + W_DICE * dice + W_LEAD * lead


def score_matrix(queries: List[Record], pool: List[Record]):
    """
    Rule scores and model signals for every (query, pool) pair.
    Returns two numpy arrays of shape (len(queries), len(pool)).
    """
    import numpy as np
    scores = np.zeros((len(queries), len(pool)), dtype=np.float64)
    signals = np.zeros((len(queries), len(pool)), dtype=np.int8)
    for i, q in enumerate(queries):
        for j, p in enumerate(pool):
            s = model_signal(q, p)
            signals[i, j] = s
            scores[i, j] = rule_score(q, p, s)
    return scores, signals

