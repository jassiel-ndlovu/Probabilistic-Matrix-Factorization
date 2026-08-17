"""Render every figure from the saved results.

Reads ``results/*.json`` and writes PDF (for LaTeX) plus PNG (for the README and
the HTML summary) into ``report/figures/``. Figures whose results file is
missing are skipped with a note rather than failing the run.

Usage::

    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pmf.viz import FIGURE_DIR, make_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIGURE_DIR)
    parser.add_argument("--dataset", default="ml-100k")
    args = parser.parse_args()

    make_all(out_dir=args.out, dataset=args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
