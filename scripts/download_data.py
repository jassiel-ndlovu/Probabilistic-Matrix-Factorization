"""Fetch MovieLens releases into ``data/``.

Usage::

    python scripts/download_data.py            # ml-100k and ml-1m
    python scripts/download_data.py ml-10m     # a specific release
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pmf.data import DATASETS, describe, download, load_movielens  # noqa: E402


def main(keys: list[str]) -> int:
    for key in keys:
        if key not in DATASETS:
            print(f"unknown dataset {key!r}; choose from {sorted(DATASETS)}")
            return 2
        download(key)
        stats = describe(load_movielens(key))
        print(f"[data] {key}: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["ml-100k", "ml-1m"]))
