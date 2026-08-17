"""A reproducible re-execution study of Probabilistic Matrix Factorization.

Public surface::

    from pmf import load_movielens, train_val_test_split, PMF, BPMF, rmse

See ``scripts/run_experiments.py`` for the full experimental protocol and
``report/`` for the write-up of what the numbers turned out to be.
"""

from __future__ import annotations

from .baselines import BiasBaseline, GlobalMean, ItemMean, UserMean
from .data import (
    DATASETS,
    RatingData,
    Split,
    describe,
    download,
    load_movielens,
    train_test_split,
    train_val_test_split,
)
from .metrics import mae, ranking_metrics, rmse, rmse_by_group
from .models import BPMF, PMF, Recommender

__version__ = "1.0.0"

__all__ = [
    "BPMF",
    "DATASETS",
    "PMF",
    "BiasBaseline",
    "GlobalMean",
    "ItemMean",
    "RatingData",
    "Recommender",
    "Split",
    "UserMean",
    "__version__",
    "describe",
    "download",
    "load_movielens",
    "mae",
    "ranking_metrics",
    "rmse",
    "rmse_by_group",
    "train_test_split",
    "train_val_test_split",
]
