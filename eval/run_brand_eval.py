"""
Part 1: precision and recall of the existing brand matcher.

The thing measured is the real pipeline: src.processor.clean_file_with_brands
(UK stripping, best_match, the word boost and the threshold), called exactly as
the app calls it. Nothing in src/ is re-implemented here.

For every threshold in 60, 70, 80, 90, 100 each row is classified:
    correct_match    output is a master name and equals true_master
    wrong_match      output is a master name but not true_master
                     (or true_master is empty)
    abstain_correct  output is not a master name and true_master is empty
    abstain_missed   output is not a master name but true_master exists

Outputs (synthetic set only):
    eval/results/brand_metrics.md        overall and per case_type
    eval/results/brand_pr_curve.png      precision, recall, F1 by threshold
    eval/results/brand_wrong_matches.md  every wrong_match with its score

If data/private/brand_eval_private.csv and data/private/brand_master_private.csv
exist, the same evaluation runs on them and ONLY aggregate metrics are printed.
No row-level value from the private files is printed or written.

Usage:
    python -m eval.run_brand_eval
    python -m eval.run_brand_eval --skip-private
"""
import argparse
import os
import sys
import tempfile
from typing import Dict, List

import pandas as pd

from eval.metrics import (
    ABSTAIN_CORRECT, ABSTAIN_MISSED, CORRECT_MATCH, WRONG_MATCH,
    evaluate, metric_cells, metrics_by_group,
)
from src.processor import clean_file_with_brands, remove_trailing_uk

THRESHOLDS = [60, 70, 80, 90, 100]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SYN_EVAL = os.path.join(HERE, "data", "brand_eval_synthetic.csv")
SYN_MASTER = os.path.join(HERE, "data", "brand_master_synthetic.csv")
PRIV_EVAL = os.path.join(REPO_ROOT, "data", "private", "brand_eval_private.csv")
PRIV_MASTER = os.path.join(REPO_ROOT, "data", "private", "brand_master_private.csv")
RESULTS = os.path.join(HERE, "results")


def load_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"raw_brand", "true_master", "case_type"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    df["raw_brand"] = df["raw_brand"].str.strip()
    df["true_master"] = df["true_master"].str.strip()
    df["case_type"] = df["case_type"].str.strip().replace("", "unlabelled")
    df = df[df["raw_brand"] != ""].reset_index(drop=True)
    return df


def load_master(path: str) -> List[str]:
    s = pd.read_csv(path, dtype=str, keep_default_na=False).iloc[:, 0].str.strip()
    return [x for x in s if x]


def run_pipeline(labels: pd.DataFrame, master: List[str], threshold: int) -> Dict[str, list]:
    """Run the real cleaning pipeline on the raw names and return prediction + score per row."""
    with tempfile.TemporaryDirectory() as tmp:
        mpath = os.path.join(tmp, "master.csv")
        cpath = os.path.join(tmp, "catalog.csv")
        pd.DataFrame({"brand": master}).to_csv(mpath, index=False)
        pd.DataFrame({"brand": labels["raw_brand"]}).to_csv(cpath, index=False)
        df, mapping = clean_file_with_brands(mpath, cpath, min_similarity=threshold, keep_original=True)
    if len(df) != len(labels):
        raise RuntimeError("pipeline changed the number of rows")
    master_set = set(master)
    outputs = [str(o) for o in df["brand"].tolist()]
    keys = [remove_trailing_uk(str(r).strip()) for r in labels["raw_brand"]]
    # Plain lists on purpose: a pandas column can turn None into NaN.
    return {
        "output": outputs,
        "predicted": [o if o in master_set else None for o in outputs],
        "suggestion": [mapping[k][2] for k in keys],
        "score": [float(mapping[k][1]) for k in keys],
    }


def evaluate_set(labels: pd.DataFrame, master: List[str]) -> Dict[int, dict]:
    golds = [t if t else None for t in labels["true_master"]]
    results = {}
    for t in THRESHOLDS:
        pred = run_pipeline(labels, master, t)
        outcomes, overall = evaluate(golds, pred["predicted"])
        by_type = metrics_by_group(labels["case_type"].tolist(), golds, pred["predicted"])
        results[t] = {"pred": pred, "outcomes": outcomes, "overall": overall, "by_type": by_type}
    return results


def check_label_master_consistency(labels: pd.DataFrame, master: List[str], show_values: bool) -> None:
    unknown = sorted(set(t for t in labels["true_master"] if t) - set(master))
    if unknown:
        detail = f": {unknown}" if show_values else ""
        raise SystemExit(f"{len(unknown)} true_master values are not in the master list{detail}")


# ---------------------------------------------------------------- reporting

COLS = ["threshold", "n", "correct", "wrong", "abstain_ok", "abstain_missed",
        "precision", "recall", "F1", "abstain rate"]


def metrics_row(label, m) -> List[str]:
    c = metric_cells(m)
    return [str(label), str(m["n"]), str(m[CORRECT_MATCH]), str(m[WRONG_MATCH]),
            str(m[ABSTAIN_CORRECT]), str(m[ABSTAIN_MISSED]),
            c["precision"], c["recall"], c["f1"], c["abstain_rate"]]


def md_table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def overall_table(results) -> str:
    return md_table(COLS, [metrics_row(t, results[t]["overall"]) for t in THRESHOLDS])


def by_type_tables(results) -> str:
    parts = []
    for t in THRESHOLDS:
        rows = [metrics_row(ct, m) for ct, m in results[t]["by_type"].items()]
        parts.append(f"#### Threshold {t}\n\n" + md_table(["case_type"] + COLS[1:], rows))
    return "\n\n".join(parts)


def write_metrics_md(results, labels, master, path) -> None:
    counts = labels["case_type"].value_counts().sort_index()
    text = [
        "# Brand matcher: precision and recall (SYNTHETIC data)",
        "",
        "Generated by `python -m eval.run_brand_eval`. Do not edit by hand.",
        "",
        f"Dataset: `eval/data/brand_eval_synthetic.csv` ({len(labels)} rows, "
        f"{sum(1 for t in labels['true_master'] if t)} with a true master), "
        f"master list `eval/data/brand_master_synthetic.csv` ({len(master)} brands).",
        "Ratios are shown to 4 decimals with the exact fraction in brackets. "
        "`n/a (0/0)` means the ratio is undefined.",
        "",
        "Rows per case type: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + ".",
        "",
        "## Overall",
        "",
        overall_table(results),
        "",
        "## By case type",
        "",
        "Case types with no true master (`lookalike_negative`, `not_in_master`) "
        "have undefined recall. Their precision is 0 whenever the tool matches "
        "anything, so read them through the wrong and abstain_ok counts.",
        "",
        by_type_tables(results),
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))


def write_wrong_md(results, labels, path) -> None:
    """Every row that is a wrong_match at any threshold, listed once with its score."""
    rows = []
    for i in range(len(labels)):
        wrong_at = [t for t in THRESHOLDS if results[t]["outcomes"][i] == WRONG_MATCH]
        if not wrong_at:
            continue
        pred = results[wrong_at[0]]["pred"]
        rows.append({
            "raw": labels.at[i, "raw_brand"],
            "true": labels.at[i, "true_master"] or "(none)",
            "pred": pred["predicted"][i],
            "score": pred["score"][i],
            "case_type": labels.at[i, "case_type"],
            "wrong_at": ", ".join(str(t) for t in wrong_at),
        })
    rows.sort(key=lambda r: (-r["score"], r["raw"]))
    table = md_table(
        ["raw_brand", "true_master", "matched to", "score", "case_type", "wrong at thresholds"],
        [[f"`{r['raw']}`", r["true"], r["pred"], f"{r['score']:.2f}", r["case_type"], r["wrong_at"]]
         for r in rows])
    text = [
        "# Every wrong match on the SYNTHETIC set",
        "",
        "Generated by `python -m eval.run_brand_eval`. Do not edit by hand.",
        "",
        "Score is the matcher's final score (after the word boost) for its best "
        "candidate. It does not depend on the threshold, so a pair is a wrong "
        "match at every threshold at or below its score.",
        "",
        f"{len(rows)} distinct rows are a wrong match at one or more thresholds.",
        "",
        table,
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))


def plot_pr(results, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    series = [("precision", "Precision", "#2a78d6", "o"),
              ("recall", "Recall", "#eb6834", "s"),
              ("f1", "F1", "#1baf7a", "^")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    xs = THRESHOLDS
    for key, label, color, marker in series:
        ys = [results[t]["overall"][key] for t in xs]
        px = [x for x, y in zip(xs, ys) if y is not None]
        py = [y for y in ys if y is not None]
        ax.plot(px, py, color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor=surface, markeredgewidth=1.5, label=label, zorder=3)
        if py:
            ax.annotate(f"{label} {py[-1]:.4f}", (px[-1], py[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8, color=ink2)
    ax.set_xticks(xs)
    ax.set_xlim(57, 109)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Similarity threshold", color=ink2, fontsize=9)
    ax.set_ylabel("Score (0 to 1)", color=ink2, fontsize=9)
    ax.set_title("Brand matcher on the synthetic set: precision, recall and F1 by threshold",
                 color=ink, fontsize=10, loc="left")
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.tick_params(colors=ink2, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink2, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def print_aggregate(title: str, results) -> None:
    print(f"\n{title}\n")
    print(overall_table(results))
    print()
    print(by_type_tables(results))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-private", action="store_true")
    args = ap.parse_args(argv)

    labels = load_labels(SYN_EVAL)
    master = load_master(SYN_MASTER)
    check_label_master_consistency(labels, master, show_values=True)
    results = evaluate_set(labels, master)

    os.makedirs(RESULTS, exist_ok=True)
    write_metrics_md(results, labels, master, os.path.join(RESULTS, "brand_metrics.md"))
    write_wrong_md(results, labels, os.path.join(RESULTS, "brand_wrong_matches.md"))
    plot_pr(results, os.path.join(RESULTS, "brand_pr_curve.png"))
    print_aggregate(f"SYNTHETIC set ({len(labels)} rows)", results)
    print(f"\nWrote brand_metrics.md, brand_wrong_matches.md, brand_pr_curve.png to {RESULTS}")

    if args.skip_private:
        return 0
    if os.path.exists(PRIV_EVAL):
        if not os.path.exists(PRIV_MASTER):
            print(f"\nPrivate labels found but no master list at {PRIV_MASTER}; skipping private run.")
            return 1
        plabels = load_labels(PRIV_EVAL)
        pmaster = load_master(PRIV_MASTER)
        check_label_master_consistency(plabels, pmaster, show_values=False)
        presults = evaluate_set(plabels, pmaster)
        n_true = sum(1 for t in plabels["true_master"] if t)
        print_aggregate(f"PRIVATE set, aggregate only ({len(plabels)} rows, {n_true} with a true master, "
                        f"{len(pmaster)} master brands)", presults)
    else:
        print(f"\nNo private labels at {PRIV_EVAL}; private run skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
