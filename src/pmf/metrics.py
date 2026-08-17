"""Evaluation metrics.

Two families are used. *Rating-prediction* metrics (RMSE, MAE) are what the PMF
paper and the original report quote. *Ranking* metrics (precision/recall/NDCG@k)
are reported alongside them because a low RMSE does not by itself imply a useful
top-N list — a point the re-execution study makes explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "rmse",
    "mae",
    "RankingScores",
    "ranking_metrics",
    "rmse_by_group",
]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error.

    Deliberately free of any regularisation term — the defect that made the
    original implementation's "training RMSE" incomparable to its test RMSE.
    """
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean(error**2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.mean(np.abs(error)))


@dataclass
class RankingScores:
    """Top-``k`` ranking quality, macro-averaged over evaluated users."""

    k: int
    precision: float
    recall: float
    ndcg: float
    n_users_evaluated: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            f"precision@{self.k}": round(self.precision, 5),
            f"recall@{self.k}": round(self.recall, 5),
            f"ndcg@{self.k}": round(self.ndcg, 5),
            "n_users_evaluated": self.n_users_evaluated,
        }


def ranking_metrics(
    model,
    train_users: np.ndarray,
    train_items: np.ndarray,
    test_users: np.ndarray,
    test_items: np.ndarray,
    test_ratings: np.ndarray,
    k: int = 10,
    relevance_threshold: float = 4.0,
    n_items: int | None = None,
    max_users: int | None = None,
    seed: int = 0,
) -> RankingScores:
    """Precision/recall/NDCG at ``k`` under a *held-out relevance* protocol.

    For each user the model scores every item, items the user rated **in
    training** are masked out (the original implementation omitted this masking,
    which lets already-seen items count as hits), and the top ``k`` survivors are
    compared against that user's test items whose rating is at least
    ``relevance_threshold``.

    Users with no relevant test item are skipped rather than scored as zero —
    including them would make the absolute numbers a function of the threshold
    more than of the model.
    """
    n_items = int(n_items if n_items is not None else max(train_items.max(), test_items.max()) + 1)

    seen: dict[int, list[int]] = {}
    for u, i in zip(train_users, train_items, strict=True):
        seen.setdefault(int(u), []).append(int(i))

    relevant: dict[int, list[int]] = {}
    for u, i, r in zip(test_users, test_items, test_ratings, strict=True):
        if r >= relevance_threshold:
            relevant.setdefault(int(u), []).append(int(i))

    users = np.array(sorted(relevant), dtype=np.int64)
    if max_users is not None and users.size > max_users:
        users = np.random.default_rng(seed).choice(users, size=max_users, replace=False)

    if users.size == 0:
        return RankingScores(k, 0.0, 0.0, 0.0, 0)

    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    precision_sum = recall_sum = ndcg_sum = 0.0
    for user in users:
        scores = np.asarray(model.score_all_items(int(user), n_items), dtype=np.float64)
        already = seen.get(int(user))
        if already:
            scores[already] = -np.inf

        top = np.argpartition(-scores, kth=min(k, scores.size - 1))[:k]
        top = top[np.argsort(-scores[top])]

        truth = set(relevant[int(user)])
        hits = np.fromiter((int(i) in truth for i in top), dtype=bool, count=top.size)

        precision_sum += hits.sum() / k
        recall_sum += hits.sum() / len(truth)

        dcg = float((hits * discounts[: hits.size]).sum())
        ideal = float(discounts[: min(len(truth), k)].sum())
        ndcg_sum += dcg / ideal if ideal > 0 else 0.0

    n = users.size
    return RankingScores(
        k=k,
        precision=precision_sum / n,
        recall=recall_sum / n,
        ndcg=ndcg_sum / n,
        n_users_evaluated=int(n),
    )


def rmse_by_group(
    y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray, bins: list[tuple[float, float]]
) -> list[dict[str, float | int | str]]:
    """RMSE stratified by a per-observation covariate.

    Used for the cold-start analysis, where ``group`` is the number of training
    ratings belonging to each test observation's user. Bins are half-open
    ``[lo, hi)``.
    """
    out: list[dict[str, float | int | str]] = []
    for lo, hi in bins:
        mask = (group >= lo) & (group < hi)
        label = f"[{int(lo)}, {int(hi)})" if np.isfinite(hi) else f"[{int(lo)}, inf)"
        if not mask.any():
            out.append({"bin": label, "n": 0, "rmse": float("nan")})
            continue
        out.append(
            {
                "bin": label,
                "n": int(mask.sum()),
                "rmse": round(rmse(y_true[mask], y_pred[mask]), 5),
            }
        )
    return out
