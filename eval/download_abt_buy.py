"""
Download the Abt-Buy product matching benchmark into data/public/abt_buy/.

Source: Database Group, Leipzig University (Prof. Erhard Rahm),
https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution
Released under a Creative Commons license (the page does not name the variant).
Cite: Koepcke, Thor, Rahm. "Evaluation of entity resolution approaches on
real-world match problems." PVLDB 3(1), 2010.

The zip is checked against a pinned SHA-256 and the extracted files are checked
against the published record counts. Nothing is redistributed by this repo.

Usage:
    python -m eval.download_abt_buy
"""
import argparse
import hashlib
import io
import os
import sys
import urllib.request
import zipfile

URL = "https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip"
SHA256 = "b9ff2937d97371b7a00b0a97033213bfbdee24e8f1a0718a41ea8364472a8ff8"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(REPO_ROOT, "data", "public", "abt_buy")

# file name -> (encoding, expected number of data rows)
EXPECTED = {
    "Abt.csv": ("latin-1", 1081),
    "Buy.csv": ("utf-8", 1092),
    "abt_buy_perfectMapping.csv": ("utf-8", 1097),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def count_csv_rows(path: str, encoding: str) -> int:
    """Count data rows (header excluded) with a real CSV parser, since fields can contain newlines."""
    import csv
    with open(path, newline="", encoding=encoding) as f:
        return sum(1 for _ in csv.reader(f)) - 1


def verify(out_dir: str) -> None:
    for name, (enc, rows) in EXPECTED.items():
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}")
        got = count_csv_rows(path, enc)
        if got != rows:
            raise SystemExit(f"{name}: expected {rows} rows, found {got}")
        print(f"  ok  {name}: {got} rows")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--force", action="store_true", help="download again even if files exist")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.force and all(os.path.exists(os.path.join(args.out_dir, n)) for n in EXPECTED):
        print(f"Files already present in {args.out_dir}, verifying.")
        verify(args.out_dir)
        return 0

    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=60) as resp:
        data = resp.read()
    digest = sha256_bytes(data)
    if digest != SHA256:
        raise SystemExit(f"SHA-256 mismatch: expected {SHA256}, got {digest}")
    print(f"  ok  sha256 {digest} ({len(data)} bytes)")

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in EXPECTED:
            # Extract only the known file names, so no path from the archive is trusted.
            with zf.open(name) as src, open(os.path.join(args.out_dir, name), "wb") as dst:
                dst.write(src.read())
    verify(args.out_dir)
    print(f"Done. Files are in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
