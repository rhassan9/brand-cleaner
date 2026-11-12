"""
Part 2: rules vs embeddings on the Abt-Buy product matching benchmark.

Task: for each Abt product, retrieve the best Buy product (top-1 over all 1092
Buy products, using the title only) and decide match or no match with a
threshold. Outcomes, precision, recall and F1 are defined in eval/metrics.py.
Every Abt product has at least one gold match, so a miss is either a wrong
top-1 candidate or an abstain.

Protocol:
- Abt products are split once, seed 42: 30% validation, 70% test.
- Thresholds (and the A+B weight) are chosen on validation only, by best F1.
  Test is evaluated once with those choices.
- Approaches: A rules, B embeddings, A+B embeddings plus model-number
  agreement, and C (optional, --use-llm) an LLM on borderline A+B pairs.

Usage:
    python -m eval.download_abt_buy
    python -m eval.run_product_eval                  # A, B, A+B
    python -m eval.run_product_eval --validation-only
    python -m eval.run_product_eval --use-llm [--max-llm-calls 150] [--llm-model claude-opus-5-5]
"""
import argparse
import os
import random
import sys
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from eval.metrics import (
    ABSTAIN_CORRECT, ABSTAIN_MISSED, CORRECT_MATCH, WRONG_MATCH,
    apply_threshold, evaluate, metric_cells, pick_threshold, sweep_thresholds,
)
from eval.product_rules import make_record, model_signal, score_matrix

SEED = 42
VAL_FRACTION = 0.30
AB_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]   # weight of the model-number signal in A+B
N_FAILURES = 10
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(REPO_ROOT, "data", "public", "abt_buy")
RESULTS = os.path.join(HERE, "results")
LLM_CACHE = os.path.join(RESULTS, "llm_judgements.jsonl")


# ------------------------------------------------------------------ data

def load_abt_buy(data_dir: str = DATA_DIR):
    paths = {n: os.path.join(data_dir, n) for n in ("Abt.csv", "Buy.csv", "abt_buy_perfectMapping.csv")}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit("Abt-Buy not found. Run: python -m eval.download_abt_buy")
    abt = pd.read_csv(paths["Abt.csv"], encoding="latin-1").sort_values("id").reset_index(drop=True)
    buy = pd.read_csv(paths["Buy.csv"], encoding="utf-8").sort_values("id").reset_index(drop=True)
    mapping = pd.read_csv(paths["abt_buy_perfectMapping.csv"])
    gold: Dict[int, Set[int]] = {}
    for a, b in zip(mapping["idAbt"], mapping["idBuy"]):
        gold.setdefault(int(a), set()).add(int(b))
    abt["name"] = abt["name"].fillna("").astype(str)
    buy["name"] = buy["name"].fillna("").astype(str)
    return abt, buy, gold


def split_ids(ids: Sequence[int], seed: int = SEED, val_fraction: float = VAL_FRACTION) -> Tuple[List[int], List[int]]:
    """Deterministic split of Abt ids into (validation, test)."""
    ordered = sorted(int(i) for i in ids)
    random.Random(seed).shuffle(ordered)
    n_val = int(round(len(ordered) * val_fraction))
    return sorted(ordered[:n_val]), sorted(ordered[n_val:])


# ------------------------------------------------------------------ scoring helpers

def top1(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Best column index and score per row. Ties go to the lowest index (lowest Buy id)."""
    idx = np.argmax(matrix, axis=1)
    return idx, matrix[np.arange(matrix.shape[0]), idx]


def gold_rank(row: np.ndarray, gold_cols: Sequence[int]) -> int:
    """1-based rank of the best-scoring gold candidate in one row."""
    best = max(row[c] for c in gold_cols)
    return int((row > best).sum()) + 1


class Approach:
    """Scores for one approach over all Abt rows, plus the tuned threshold."""

    def __init__(self, name: str, matrix: np.ndarray, buy_ids: np.ndarray):
        self.name = name
        self.matrix = matrix
        idx, sc = top1(matrix)
        self.cand_col = idx
        self.cand = [int(buy_ids[i]) for i in idx]
        self.score = [float(s) for s in sc]
        self.threshold = None
        self.val_metrics = None

    def rows(self, row_idx: Sequence[int]):
        return [self.cand[i] for i in row_idx], [self.score[i] for i in row_idx]

    def tune(self, val_rows: Sequence[int], golds_val: Sequence[Set[int]]):
        cands, scores = self.rows(val_rows)
        thresholds = sorted(set(scores)) + [float("inf")]
        sweep = sweep_thresholds(golds_val, cands, scores, thresholds)
        self.threshold, self.val_metrics = pick_threshold(sweep)
        return self.threshold, self.val_metrics

    def predict(self, row_idx: Sequence[int]):
        cands, scores = self.rows(row_idx)
        return apply_threshold(cands, scores, self.threshold)


def top1_accuracy(appr: Approach, row_idx, golds) -> Tuple[int, int]:
    hits = sum(1 for i, g in zip(row_idx, golds) if appr.cand[i] in g)
    return hits, len(row_idx)


def recall_at_k(appr: Approach, row_idx, gold_cols_list, k: int) -> Tuple[int, int]:
    hits = 0
    for i, cols in zip(row_idx, gold_cols_list):
        hits += gold_rank(appr.matrix[i], cols) <= k
    return hits, len(row_idx)


# ------------------------------------------------------------------ reporting

def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


METRIC_COLS = ["approach", "threshold", "n", "correct", "wrong", "abstain_missed",
               "precision", "recall", "F1"]


def metric_row(name, threshold, m):
    c = metric_cells(m)
    return [name, f"{threshold:.4f}", m["n"], m[CORRECT_MATCH], m[WRONG_MATCH],
            m[ABSTAIN_MISSED], c["precision"], c["recall"], c["f1"]]


def failure_reason(appr: Approach, i: int, outcome: str, abt_recs, buy_recs, gold_cols) -> str:
    q, c = abt_recs[i], buy_recs[appr.cand_col[i]]
    s, t = appr.score[i], appr.threshold
    rank = gold_rank(appr.matrix[i], gold_cols)
    top_is_gold = appr.cand_col[i] in gold_cols
    if outcome == ABSTAIN_MISSED:
        if top_is_gold:
            return f"Correct product retrieved but score {s:.4f} is below threshold {t:.4f}."
        return f"Below threshold ({s:.4f} < {t:.4f}) and the top candidate was wrong anyway; gold ranked {rank}."
    sig = model_signal(q, c)
    if sig < 0:
        return (f"Model numbers differ ({', '.join(sorted(q.codes))} vs {', '.join(sorted(c.codes))}) "
                f"but score {s:.4f} passed the threshold; gold ranked {rank}.")
    if sig > 0:
        return (f"Model codes overlap (often a pack, bundle or colour variant), yet the gold mapping "
                f"names another product; gold ranked {rank}.")
    side = "either title" if not q.codes and not c.codes else ("the Abt title" if not q.codes else "the Buy title")
    return (f"No letter+digit model code extracted from {side}, so similar wording decided; "
            f"gold ranked {rank}.")


def llm_failure_reason(appr: Approach, i: int, outcome: str, verdict, base_reason: str, gold_cols) -> str:
    """Reason for approach C, where borderline rows were decided by the LLM instead of the threshold."""
    where = f"Borderline pair (score {appr.score[i]:.4f}, threshold {appr.threshold:.4f})"
    if verdict is None:
        return f"LLM answer could not be parsed, so the A+B decision stood. {base_reason}"
    if verdict:
        # YES keeps the candidate, so a failure here means the candidate was not a gold match.
        return f"{where}; the LLM said YES but the candidate is not a gold match."
    if appr.cand_col[i] in gold_cols:
        return f"{where}; the LLM said NO to the correct product."
    return f"{where}; the LLM rightly said NO, but the correct product was not the top candidate."


def failures_md(appr: Approach, test_rows, outcomes, abt, buy, abt_recs, buy_recs, gold, buy_col,
                title: str = None, llm_verdicts: dict = None) -> str:
    """
    10 random test failures with a reason each. For approach C pass the A+B
    approach, C's outcomes, a title and llm_verdicts {test position: True/False/None}.
    """
    fails = [k for k, o in enumerate(outcomes) if o in (WRONG_MATCH, ABSTAIN_MISSED)]
    picked = sorted(random.Random(SEED).sample(fails, min(N_FAILURES, len(fails))))
    lines = [f"### {title or appr.name}", "",
             f"{len(fails)} failures on the test split; {len(picked)} drawn at random (seed {SEED}), "
             "not hand-picked.", ""]
    rows = []
    for k in picked:
        i = test_rows[k]
        aid = int(abt.at[i, "id"])
        gcols = [buy_col[b] for b in sorted(gold[aid])]
        gold_titles = " / ".join(buy.at[c, "name"] for c in gcols)
        pred_title = buy.at[appr.cand_col[i], "name"]
        reason = failure_reason(appr, i, outcomes[k], abt_recs, buy_recs, gcols)
        if llm_verdicts is not None:
            if k in llm_verdicts:
                reason = llm_failure_reason(appr, i, outcomes[k], llm_verdicts[k], reason, gcols)
            else:
                reason = "Not borderline, so not sent to the LLM. " + reason
        rows.append([outcomes[k], f"`{abt.at[i, 'name']}`", f"`{pred_title}`", f"{appr.score[i]:.4f}",
                     f"`{gold_titles}`", reason])
    lines.append(md_table(["outcome", "Abt title", "top Buy candidate", "score", "gold Buy title", "reason"], rows))
    return "\n".join(lines)


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-only", action="store_true",
                    help="tune and print validation results only; do not touch the test split")
    ap.add_argument("--use-llm", action="store_true", help="run approach C (costs money, asks first)")
    ap.add_argument("--max-llm-calls", type=int, default=150)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--yes", action="store_true", help="do not ask before sending LLM calls")
    args = ap.parse_args(argv)

    abt, buy, gold = load_abt_buy()
    buy_ids = buy["id"].to_numpy()
    buy_col = {int(b): k for k, b in enumerate(buy_ids)}
    abt_row = {int(a): k for k, a in enumerate(abt["id"])}
    val_ids, test_ids = split_ids(abt["id"])
    val_rows = [abt_row[a] for a in val_ids]
    test_rows = [abt_row[a] for a in test_ids]
    golds_all = [gold.get(int(a), set()) for a in abt["id"]]
    gcols_all = [[buy_col[b] for b in g] for g in golds_all]
    golds_val = [golds_all[i] for i in val_rows]
    golds_test = [golds_all[i] for i in test_rows]
    print(f"Abt {len(abt)}, Buy {len(buy)}, gold pairs {sum(len(g) for g in gold.values())}. "
          f"Split (seed {SEED}): validation {len(val_rows)}, test {len(test_rows)}.")

    print("Scoring A (rules) for all pairs...")
    abt_recs = [make_record(i, t) for i, t in zip(abt["id"], abt["name"])]
    buy_recs = [make_record(i, t) for i, t in zip(buy["id"], buy["name"])]
    rule_m, signals = score_matrix(abt_recs, buy_recs)
    print("Scoring B (embeddings)...")
    from eval.product_embed import MODEL_NAME, MODEL_REVISION, cosine_matrix
    cos = cosine_matrix(abt["name"].tolist(), buy["name"].tolist()).astype(np.float64)

    A = Approach("A: rules", rule_m, buy_ids)
    B = Approach("B: embeddings", cos, buy_ids)
    A.tune(val_rows, golds_val)
    B.tune(val_rows, golds_val)

    # A+B: cosine + w * model-number signal (+1 agree, 0 unknown, -1 conflict); w chosen on validation.
    w_rows, best = [], None
    for w in AB_WEIGHTS:
        cand = Approach(f"A+B (w={w})", cos + w * signals, buy_ids)
        t, m = cand.tune(val_rows, golds_val)
        w_rows.append([f"{w:.2f}", f"{t:.4f}", metric_cells(m)["precision"], metric_cells(m)["recall"],
                       metric_cells(m)["f1"]])
        key = (m["f1"] if m["f1"] is not None else -1.0, -w)
        if best is None or key > best[0]:
            best = (key, w, cand)
    ab_w, AB = best[1], best[2]
    AB.name = "A+B: embeddings + model number"
    approaches = [A, B, AB]

    print("\nValidation (used for tuning):")
    val_table = md_table(METRIC_COLS, [metric_row(x.name, x.threshold, x.val_metrics) for x in approaches])
    print(val_table)
    print(f"A+B weight chosen on validation: {ab_w}")
    if args.validation_only:
        print("\n--validation-only: test split not evaluated.")
        return 0

    # ---- test, evaluated once with the choices above
    test_results = {}
    for x in approaches:
        outcomes, m = evaluate(golds_test, x.predict(test_rows))
        test_results[x.name] = (outcomes, m, x.threshold)

    llm_section = ""
    llm_failures = None
    if args.use_llm:
        from eval.product_llm import (DEFAULT_MODEL, LLM_MARGIN, apply_verdicts, judge_pairs,
                                      select_borderline)
        model = args.llm_model or DEFAULT_MODEL
        test_scores = [AB.score[i] for i in test_rows]
        border = select_borderline(test_scores, AB.threshold, args.max_llm_calls)
        items = [{"abt_id": int(abt.at[test_rows[k], "id"]), "buy_id": AB.cand[test_rows[k]],
                  "abt_title": abt.at[test_rows[k], "name"],
                  "buy_title": buy.at[AB.cand_col[test_rows[k]], "name"]} for k in border]
        verdicts = judge_pairs(items, model, LLM_CACHE, args.yes)
        preds, n_unparsed = apply_verdicts(AB.predict(test_rows), border, items, verdicts)
        outcomes, m = evaluate(golds_test, preds)
        cname = f"C: A+B + LLM on borderline ({model})"
        test_results[cname] = (outcomes, m, AB.threshold)
        llm_failures = (cname, {k: verdicts[(str(it["abt_id"]), str(it["buy_id"]))]
                                for k, it in zip(border, items)})
        llm_section = (f"\nApproach C sent {len(border)} test pairs with an A+B score within "
                       f"{LLM_MARGIN} of the A+B threshold to `{model}` (cap {args.max_llm_calls}); "
                       f"{n_unparsed} answers could not be parsed and kept the A+B decision. "
                       "Answers are cached in `eval/results/llm_judgements.jsonl`.\n")

    test_table = md_table(METRIC_COLS, [metric_row(n, t, m) for n, (o, m, t) in test_results.items()])
    retr_rows = []
    for x in approaches:
        h1, n1 = top1_accuracy(x, test_rows, golds_test)
        h10, _ = recall_at_k(x, test_rows, [gcols_all[i] for i in test_rows], 10)
        retr_rows.append([x.name, f"{h1 / n1:.4f} ({h1}/{n1})", f"{h10 / n1:.4f} ({h10}/{n1})"])
    print("\nTest (held out, evaluated once):")
    print(test_table)

    os.makedirs(RESULTS, exist_ok=True)
    n_codes_abt = sum(1 for r in abt_recs if r.codes)
    n_codes_buy = sum(1 for r in buy_recs if r.codes)
    md = [
        "# Abt-Buy product matching: rules vs embeddings",
        "",
        "Generated by `python -m eval.run_product_eval`. Do not edit by hand.",
        "",
        f"- Data: Abt {len(abt)} products, Buy {len(buy)} products, {sum(len(g) for g in gold.values())} "
        f"gold pairs. Every Abt product has at least one gold match. Titles only (`name`).",
        f"- Split of Abt products (seed {SEED}): validation {len(val_rows)}, test {len(test_rows)}. "
        "All 1092 Buy products are candidates for both.",
        f"- Model numbers found: {n_codes_abt} of {len(abt)} Abt titles, {n_codes_buy} of {len(buy)} Buy titles.",
        f"- Embedding model: `{MODEL_NAME}` at revision `{MODEL_REVISION}`, CPU.",
        f"- Thresholds and the A+B weight were chosen on validation by best F1, then applied once to test.",
        "- Ratios to 4 decimals with exact fractions. `abstain_ok` is always 0 here because every "
        "Abt product has a true match, so it is not shown.",
        "",
        "## Test split (held out)",
        "",
        test_table,
        llm_section,
        "## Retrieval quality on the test split (before any threshold)",
        "",
        "Top-1 accuracy is the share of Abt products whose best candidate is a gold match; it is "
        "the recall the approach would reach if it never abstained. Recall@10 says whether a gold "
        "match is anywhere in the top 10.",
        "",
        md_table(["approach", "top-1 accuracy", "recall@10"], retr_rows),
        "",
        "## Validation split (tuning only)",
        "",
        val_table,
        "",
        "A+B weight search on validation (score = cosine + w * model-number signal, signal +1 agree, "
        "0 unknown, -1 conflict). w = 0 is plain B.",
        "",
        md_table(["w", "best threshold", "precision", "recall", "F1"], w_rows),
        "",
        f"Chosen w = {ab_w}.",
        "",
    ]
    with open(os.path.join(RESULTS, "product_metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    fmd = ["# Abt-Buy failure examples (test split)", "",
           "Generated by `python -m eval.run_product_eval`. Do not edit by hand. "
           "Reasons are produced by code from the scores and model numbers, not written by hand.", ""]
    for x in approaches:
        fmd.append(failures_md(x, test_rows, test_results[x.name][0], abt, buy, abt_recs, buy_recs, gold, buy_col))
        fmd.append("")
    if llm_failures is not None:
        cname, llm_verdicts = llm_failures
        fmd.append(failures_md(AB, test_rows, test_results[cname][0], abt, buy, abt_recs, buy_recs, gold,
                               buy_col, title=cname, llm_verdicts=llm_verdicts))
        fmd.append("")
    with open(os.path.join(RESULTS, "product_failures.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(fmd))
    print(f"\nWrote product_metrics.md and product_failures.md to {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
