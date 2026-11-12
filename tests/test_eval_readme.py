"""
Every results table quoted in a README must match the generated files.

A quoted block looks like:
    <!-- results:brand_metrics.md -->
    | ...table lines... |
    <!-- /results -->
Each line inside the block must appear verbatim in eval/results/<file>.
Regenerate with python -m eval.run_brand_eval / python -m eval.run_product_eval.
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "eval", "results")
READMES = [os.path.join(ROOT, "README.md"), os.path.join(ROOT, "eval", "README.md")]
BLOCK_RE = re.compile(r"<!-- results:(\S+) -->\n(.*?)<!-- /results -->", re.S)


def quoted_blocks():
    for path in READMES:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for m in BLOCK_RE.finditer(text):
            yield os.path.relpath(path, ROOT), m.group(1), [l for l in m.group(2).splitlines() if l.strip()]


BLOCKS = list(quoted_blocks())


def test_readmes_quote_some_results():
    assert len(BLOCKS) >= 5


@pytest.mark.parametrize("readme,results_file,lines", BLOCKS,
                         ids=[f"{b[0]}:{b[1]}:{i}" for i, b in enumerate(BLOCKS)])
def test_quoted_table_matches_generated_results(readme, results_file, lines):
    with open(os.path.join(RESULTS, results_file), encoding="utf-8") as f:
        generated = set(f.read().splitlines())
    missing = [l for l in lines if l not in generated]
    assert not missing, f"{readme} quotes lines not in {results_file}: {missing}"


def test_no_dashes_in_docs():
    for path in READMES:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        assert "—" not in text and "–" not in text, path
