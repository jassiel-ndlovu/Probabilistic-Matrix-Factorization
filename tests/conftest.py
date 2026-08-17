"""Shared fixtures.

The ML-100K split is module-scoped and built once: loading and splitting 100k
rows for every test would dominate the suite's runtime.
"""

from __future__ import annotations

import pytest

from pmf.data import Split, load_movielens, train_val_test_split


@pytest.fixture(scope="session")
def ml100k_split() -> Split:
    data = load_movielens("ml-100k", auto_download=False)
    return train_val_test_split(data, val_size=0.1, test_size=0.2, seed=0)
