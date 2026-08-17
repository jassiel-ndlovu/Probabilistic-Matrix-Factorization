"""Metric correctness."""

from __future__ import annotations

import numpy as np
import pytest

from pmf.metrics import mae, ranking_metrics, rmse, rmse_by_group


def test_rmse_is_zero_for_perfect_predictions() -> None:
    y = np.array([1.0, 3.0, 5.0])
    assert rmse(y, y) == 0.0


def test_rmse_matches_hand_computation() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 4.0, 3.0])
    # errors 1, 2, 0 -> sqrt((1 + 4 + 0) / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(5 / 3))


def test_rmse_contains_no_regularisation_term() -> None:
    """The specific defect being corrected: RMSE must depend only on the
    residuals, never on parameter magnitudes."""
    y_true = np.array([3.0, 4.0])
    y_pred = np.array([3.0, 4.0])
    assert rmse(y_true, y_pred) == 0.0


def test_mae_matches_hand_computation() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 4.0, 3.0])
    assert mae(y_true, y_pred) == pytest.approx(1.0)


def test_rmse_penalises_large_errors_more_than_mae() -> None:
    y_true = np.zeros(4)
    spread = np.array([0.0, 0.0, 0.0, 4.0])
    even = np.ones(4)
    assert rmse(y_true, spread) > rmse(y_true, even)
    assert mae(y_true, spread) == mae(y_true, even)


class _Oracle:
    """Scores items by a fixed per-user preference table."""

    def __init__(self, table: dict[int, np.ndarray]) -> None:
        self.table = table

    def score_all_items(self, user: int, n_items: int) -> np.ndarray:
        return self.table[user]


def test_ranking_metrics_on_a_hand_worked_example() -> None:
    # 5 items. User 0 rated item 0 in training (must be excluded from ranking).
    # Model ranks items 0 > 1 > 2 > 3 > 4.
    model = _Oracle({0: np.array([9.0, 8.0, 7.0, 6.0, 5.0])})

    scores = ranking_metrics(
        model,
        train_users=np.array([0]),
        train_items=np.array([0]),
        test_users=np.array([0, 0]),
        test_items=np.array([1, 3]),
        test_ratings=np.array([5.0, 5.0]),
        k=2,
        relevance_threshold=4.0,
        n_items=5,
    )
    # Item 0 is masked, so the top-2 are items 1 and 2. Item 1 is relevant.
    assert scores.precision == pytest.approx(0.5)
    assert scores.recall == pytest.approx(0.5)
    # DCG = 1/log2(2) = 1 (hit at rank 1); IDCG = 1/log2(2) + 1/log2(3).
    assert scores.ndcg == pytest.approx(1.0 / (1.0 + 1.0 / np.log2(3)))
    assert scores.n_users_evaluated == 1


def test_ranking_excludes_training_items() -> None:
    """Without masking, the top-scored item is one the user already rated and
    would inflate the score. This is the bug in the original topK()."""
    model = _Oracle({0: np.array([9.0, 8.0, 7.0])})
    masked = ranking_metrics(
        model,
        train_users=np.array([0]),
        train_items=np.array([0]),
        test_users=np.array([0]),
        test_items=np.array([1]),
        test_ratings=np.array([5.0]),
        k=1,
        n_items=3,
    )
    assert masked.precision == pytest.approx(1.0)


def test_ranking_skips_users_with_no_relevant_items() -> None:
    model = _Oracle({0: np.array([1.0, 2.0, 3.0])})
    scores = ranking_metrics(
        model,
        train_users=np.array([0]),
        train_items=np.array([0]),
        test_users=np.array([0]),
        test_items=np.array([1]),
        test_ratings=np.array([2.0]),  # below threshold
        k=2,
        relevance_threshold=4.0,
        n_items=3,
    )
    assert scores.n_users_evaluated == 0


def test_rmse_by_group_partitions_observations() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 3.0, 3.0, 4.0])
    group = np.array([1, 1, 50, 50])

    rows = rmse_by_group(y_true, y_pred, group, [(0, 10), (10, np.inf)])
    assert [r["n"] for r in rows] == [2, 2]
    assert rows[0]["rmse"] == pytest.approx(1.0)
    assert rows[1]["rmse"] == pytest.approx(0.0)


def test_rmse_by_group_handles_empty_bins() -> None:
    rows = rmse_by_group(
        np.array([1.0]), np.array([1.0]), np.array([5]), [(0, 1), (0, 10)]
    )
    assert rows[0]["n"] == 0
    assert np.isnan(rows[0]["rmse"])
