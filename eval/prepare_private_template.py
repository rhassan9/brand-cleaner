"""
Prepare files for labelling real client brand names locally.

Writes, under data/private/ (gitignored):
    brand_master_private.csv           the client's master list (column: brand)
    brand_eval_private_TO_LABEL.csv    every unique raw brand name from the
                                       catalogs, with empty true_master and
                                       case_type columns to fill in by hand

The template deliberately has no suggestion column: showing the tool's guess
while labelling would bias the labels towards the tool. When labelling is done,
rename the file to brand_eval_private.csv and run python -m eval.run_brand_eval.
Leave true_master empty only for names that truly have no correct master.

Brand names are read from the FIRST column of each file, as the app does, with
the same trimming. Only counts are printed.

Usage:
    python -m eval.prepare_private_template --master MASTER.xlsx CATALOG1.xlsx CATALOG2.xlsx ...
"""
import argparse
import os
import sys

import pandas as pd

from src.processor import load_clean_brands, read_single_column_series

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "data", "private")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, help="client master brand list (first column)")
    ap.add_argument("catalogs", nargs="+", help="supplier catalogs (brand in first column)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing template")
    args = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    template = os.path.join(OUT_DIR, "brand_eval_private_TO_LABEL.csv")
    if os.path.exists(template) and not args.force:
        print(f"{template} exists; pass --force to overwrite (this would erase labels in it).")
        return 1

    master = load_clean_brands(args.master)
    pd.DataFrame({"brand": master}).to_csv(os.path.join(OUT_DIR, "brand_master_private.csv"), index=False)

    seen, raws = set(), []
    for path in args.catalogs:
        for v in read_single_column_series(path):
            if v not in seen:
                seen.add(v)
                raws.append(v)
    pd.DataFrame({"raw_brand": raws, "true_master": "", "case_type": ""}).to_csv(template, index=False)
    print(f"Master list: {len(master)} brands. Template: {len(raws)} unique raw names "
          f"from {len(args.catalogs)} catalogs. Written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
