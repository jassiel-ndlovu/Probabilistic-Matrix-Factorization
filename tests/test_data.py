"""Data loading, re-indexing and splitting."""

from __future__ import annotations

import numpy as np
import pytest

from pmf.data import (
    RatingData,
    describe,
    load_movielens,
    train_test_split,
    train_val_test_split,
)


@pytest.fixture(scope="module")
def ml100k() -> RatingData:
    return load_movielens("ml-100k", auto_download=False)


def _toy(n: int = 500, seed: int = 0) -> RatingData:
    rng = np.random.default_rng(seed)
    triples = np.column_stack(
        [rng.integers(0, 20, n), rng.integers(0, 30, n), rng.integers(1, 6, n)]
    ).astype(np.float64)
    return RatingData("toy", triples, 20, 30, 1.0, 5.0)


def test_ml100k_shape(ml100k: RatingData) -> None:
    assert len(ml100k) == 100_000
    assert ml100k.n_users == 943
    assert ml100k.n_items == 1682
    assert (ml100k.rating_min, ml100k.rating_max) == (1.0, 5.0)


def test_indices_are_contiguous_and_zero_based(ml100k: RatingData) -> None:
    """The original implementation sized factors as max(id)+1, wasting row 0 and
    silently depending on the raw id space. Re-indexing must be exact."""
    assert set(np.unique(ml100k.users)) == set(range(ml100k.n_users))
    assert set(np.unique(ml100k.items)) == set(range(ml100k.n_items))


def test_split_is_deterministic_given_seed(ml100k: RatingData) -> None:
    a = train_test_split(ml100k, test_size=0.2, seed=7)
    b = train_test_split(ml100k, test_size=0.2, seed=7)
    assert np.array_equal(a.train.triples, b.train.triples)
    assert np.array_equal(a.test.triples, b.test.triples)


def test_different_seeds_give_different_splits(ml100k: RatingData) -> None:
    a = train_test_split(ml100k, test_size=0.2, seed=1)
    b = train_test_split(ml100k, test_size=0.2, seed=2)
    assert not np.array_equal(a.test.triples, b.test.triples)


def test_split_partitions_exactly(ml100k: RatingData) -> None:
    split = train_test_split(ml100k, test_size=0.2, seed=0)
    assert len(split.train) + len(split.test) == len(ml100k)
    assert len(split.test) == pytest.approx(20_000, abs=1)

    combined = np.vstack([split.train.triples, split.test.triples])
    order = np.lexsort(combined.T)
    original = np.lexsort(ml100k.triples.T)
    assert np.array_equal(combined[order], ml100k.triples[original])


def test_three_way_split_is_disjoint(ml100k: RatingData) -> None:
    split = train_val_test_split(ml100k, val_size=0.1, test_size=0.2, seed=0)
    assert len(split.train) + len(split.val) + len(split.test) == len(ml100k)
    assert len(split.val) == pytest.approx(10_000, abs=1)
    assert len(split.test) == pytest.approx(20_000, abs=1)


def test_split_preserves_ambient_index_space() -> None:
    """Subsetting must not renumber entities, or train and test factors would
    refer to different users."""
    data = _toy()
    split = train_test_split(data, test_size=0.3, seed=0)
    assert split.train.n_users == data.n_users == split.test.n_users
    assert split.train.n_items == data.n_items == split.test.n_items


def test_sparsity_matches_manual_computation() -> None:
    data = _toy(n=300)
    expected = 1.0 - len(data) / (data.n_users * data.n_items)
    assert data.sparsity == pytest.approx(expected)


def test_describe_reports_known_ml100k_facts(ml100k: RatingData) -> None:
    stats = describe(ml100k)
    assert stats["n_ratings"] == 100_000
    assert stats["rating_mean"] == pytest.approx(3.5299, abs=1e-3)
    # GroupLens guarantees every ML-100K user has at least 20 ratings.
    assert stats["ratings_per_user_min"] >= 20
    assert stats["users_with_lt_20_ratings_pct"] == 0.0


def test_titles_are_loaded_and_aligned(ml100k: RatingData) -> None:
    assert ml100k.titles is not None
    assert len(ml100k.titles) > 1600
    assert all(0 <= idx < ml100k.n_items for idx in ml100k.titles)
