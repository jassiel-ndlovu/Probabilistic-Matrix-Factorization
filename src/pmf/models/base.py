"""Shared interface and training bookkeeping for every model in the study."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from ..data import RatingData

__all__ = ["Recommender", "History", "scatter_add"]


@dataclass
class History:
    """Per-epoch training trace.

    ``objective`` is the MAP objective actually being minimised; ``train_rmse``
    is the unpenalised RMSE on the training partition. Keeping them as separate
    series is the fix for the original implementation's conflation of the two.
    """

    epoch: list[int] = field(default_factory=list)
    objective: list[float] = field(default_factory=list)
    train_rmse: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)
    elapsed_s: list[float] = field(default_factory=list)

    def record(
        self,
        epoch: int,
        objective: float,
        train_rmse: float,
        val_rmse: float | None,
        elapsed_s: float,
    ) -> None:
        self.epoch.append(epoch)
        self.objective.append(objective)
        self.train_rmse.append(train_rmse)
        self.val_rmse.append(val_rmse if val_rmse is not None else float("nan"))
        self.elapsed_s.append(elapsed_s)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "epoch": list(self.epoch),
            "objective": list(self.objective),
            "train_rmse": list(self.train_rmse),
            "val_rmse": list(self.val_rmse),
            "elapsed_s": list(self.elapsed_s),
        }


class Recommender(ABC):
    """Minimal contract every model satisfies so evaluation code is model-agnostic."""

    name: str = "recommender"
    rating_min: float = 1.0
    rating_max: float = 5.0

    @abstractmethod
    def fit(self, train: RatingData, val: RatingData | None = None) -> Recommender:
        """Estimate parameters from ``train``, optionally monitoring ``val``."""

    @abstractmethod
    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Predicted ratings for aligned ``(user, item)`` index arrays."""

    def predict_data(self, data: RatingData) -> np.ndarray:
        return self.predict(data.users, data.items)

    def score_all_items(self, user: int, n_items: int) -> np.ndarray:
        """Scores for every item for one user; used by the ranking metrics."""
        return self.predict(np.full(n_items, user, dtype=np.int64), np.arange(n_items))

    def _clip(self, predictions: np.ndarray) -> np.ndarray:
        return np.clip(predictions, self.rating_min, self.rating_max)


#: Batch size above which the sparse-matmul reduction beats ``np.add.at``.
#: Measured on this repo's shapes (d=20, n_rows 943-3706): at batch 256
#: ``np.add.at`` costs 0.09 ms against 0.16 ms for the matmul, whose CSR
#: construction dominates; by batch 1024 the ordering reverses (0.32 vs 0.21 ms)
#: and by batch 4096 the matmul is 4x faster (1.28 vs 0.31 ms).
SPARSE_SCATTER_THRESHOLD = 512


def scatter_add(n_rows: int, index: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Sum ``values`` rows into a ``(n_rows, d)`` array by ``index``.

    Semantically ``out = zeros((n_rows, d)); np.add.at(out, index, values)``.
    This sits in the innermost training loop, so it switches strategy on batch
    size: ``np.add.at`` is an unbuffered ufunc with low fixed cost but poor
    scaling, while building a sparse indicator matrix and multiplying hands the
    reduction to BLAS at the price of constructing the matrix. See
    :data:`SPARSE_SCATTER_THRESHOLD` for the measured crossover.
    """
    batch = index.shape[0]
    if batch < SPARSE_SCATTER_THRESHOLD:
        out = np.zeros((n_rows, values.shape[1]), dtype=np.float64)
        np.add.at(out, index, values)
        return out

    indicator = sparse.csr_matrix(
        (np.ones(batch), (index, np.arange(batch))), shape=(n_rows, batch)
    )
    return np.asarray(indicator @ values)
