"""Non-factorisation reference points.

These exist to make the PMF numbers interpretable. An RMSE is meaningless in
isolation: the relevant question is how much of it a model buys over predicting
the global mean, or over simple additive user/item effects. The original report
quoted a test RMSE of 1.0109 without any such reference, which is precisely how
a result barely better than an item average can look acceptable.
"""

from __future__ import annotations

import numpy as np

from .data import RatingData
from .models.base import Recommender

__all__ = ["GlobalMean", "UserMean", "ItemMean", "BiasBaseline"]


class GlobalMean(Recommender):
    """Predict the training-set mean rating for every pair. The floor."""

    name = "Global mean"

    def fit(self, train: RatingData, val: RatingData | None = None) -> GlobalMean:
        self.rating_min, self.rating_max = train.rating_min, train.rating_max
        self.mu = float(train.ratings.mean())
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(users).shape[0], self.mu, dtype=np.float64)


class UserMean(Recommender):
    """Predict each user's own training mean, backing off to the global mean."""

    name = "User mean"

    def fit(self, train: RatingData, val: RatingData | None = None) -> UserMean:
        self.rating_min, self.rating_max = train.rating_min, train.rating_max
        self.mu = float(train.ratings.mean())
        totals = np.bincount(train.users, weights=train.ratings, minlength=train.n_users)
        counts = np.bincount(train.users, minlength=train.n_users)
        self.means = np.where(counts > 0, totals / np.maximum(counts, 1), self.mu)
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        out = np.full(users.shape[0], self.mu, dtype=np.float64)
        known = users < self.means.shape[0]
        out[known] = self.means[users[known]]
        return self._clip(out)


class ItemMean(Recommender):
    """Predict each item's own training mean. The bar PMF must clear."""

    name = "Item mean"

    def fit(self, train: RatingData, val: RatingData | None = None) -> ItemMean:
        self.rating_min, self.rating_max = train.rating_min, train.rating_max
        self.mu = float(train.ratings.mean())
        totals = np.bincount(train.items, weights=train.ratings, minlength=train.n_items)
        counts = np.bincount(train.items, minlength=train.n_items)
        self.means = np.where(counts > 0, totals / np.maximum(counts, 1), self.mu)
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        items = np.asarray(items, dtype=np.int64)
        out = np.full(items.shape[0], self.mu, dtype=np.float64)
        known = items < self.means.shape[0]
        out[known] = self.means[items[known]]
        return self._clip(out)


class BiasBaseline(Recommender):
    """Additive user and item offsets, :math:`\\hat r_{ij} = \\mu + b_i + c_j`.

    Fitted by the standard damped closed form rather than by descent: each
    offset is a shrunk mean residual, with ``damping`` acting as the Gaussian
    prior's precision. Shrinkage is what stops an item with two ratings from
    claiming an extreme offset.

    This is the strongest model in the file and the one that matters — a
    factorisation that cannot beat it has not learned any interaction structure.
    """

    name = "Bias baseline"

    def __init__(self, damping: float = 10.0, n_iter: int = 15, clip: bool = True) -> None:
        self.damping = float(damping)
        self.n_iter = int(n_iter)
        self.clip_predictions = bool(clip)

    def fit(self, train: RatingData, val: RatingData | None = None) -> BiasBaseline:
        self.rating_min, self.rating_max = train.rating_min, train.rating_max
        users, items, ratings = train.users, train.items, train.ratings
        self.mu = float(ratings.mean())

        self.b_user = np.zeros(train.n_users)
        self.b_item = np.zeros(train.n_items)
        count_user = np.bincount(users, minlength=train.n_users)
        count_item = np.bincount(items, minlength=train.n_items)

        # Alternating shrunk least squares; converges in a handful of sweeps.
        for _ in range(self.n_iter):
            residual = ratings - self.mu - self.b_user[users]
            self.b_item = np.bincount(items, weights=residual, minlength=train.n_items) / (
                count_item + self.damping
            )
            residual = ratings - self.mu - self.b_item[items]
            self.b_user = np.bincount(users, weights=residual, minlength=train.n_users) / (
                count_user + self.damping
            )
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        out = np.full(users.shape[0], self.mu, dtype=np.float64)
        ok_u = users < self.b_user.shape[0]
        ok_i = items < self.b_item.shape[0]
        out[ok_u] += self.b_user[users[ok_u]]
        out[ok_i] += self.b_item[items[ok_i]]
        return self._clip(out) if self.clip_predictions else out
