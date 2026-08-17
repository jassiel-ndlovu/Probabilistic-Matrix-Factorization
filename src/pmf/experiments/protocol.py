"""Shared experimental protocol.

Every number reported in the paper goes through this module, so the protocol is
stated once and applied uniformly:

* **Repeated splits.** Each configuration is fitted over several seeds, each of
  which reshuffles the train/validation/test partition. A single split on 100k
  ratings has enough sampling noise (roughly +/- 0.006 RMSE, measured) to
  reorder models that differ by less than that, so single-run comparisons are
  not reported.
* **Selection on validation only.** Early stopping and hyperparameter choice use
  the validation partition. The test partition is scored once per fitted model
  and never optimised against.
* **Uncertainty quoted.** Results carry a standard deviation and a Student-t
  95% interval over seeds, so a difference can be judged against the noise
  rather than eyeballed.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ..data import RatingData, Split, load_movielens, train_val_test_split
from ..metrics import mae, rmse
from ..models.base import Recommender

__all__ = [
    "RESULTS_DIR",
    "Measurement",
    "EvaluationResult",
    "aggregate",
    "evaluate",
    "make_split",
    "save_result",
    "environment",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass
class Measurement:
    """A scalar summarised across seeds."""

    mean: float
    std: float
    ci95: float
    values: list[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 5),
            "std": round(self.std, 5),
            "ci95": round(self.ci95, 5),
            "values": [round(v, 5) for v in self.values],
        }

    def __str__(self) -> str:
        return f"{self.mean:.4f} +/- {self.ci95:.4f}"


def aggregate(values: Sequence[float]) -> Measurement:
    """Mean, standard deviation and a Student-t 95% interval over seeds.

    A normal interval would understate the spread at the 3-10 seeds used here,
    so the t distribution's critical value is used instead.
    """
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return Measurement(float("nan"), float("nan"), float("nan"), list(values))
    if array.size == 1:
        return Measurement(float(array[0]), 0.0, 0.0, [float(array[0])])

    mean = float(array.mean())
    std = float(array.std(ddof=1))
    critical = float(stats.t.ppf(0.975, df=array.size - 1))
    return Measurement(mean, std, critical * std / np.sqrt(array.size), array.tolist())


@dataclass
class EvaluationResult:
    """One model configuration, evaluated across seeds."""

    label: str
    dataset: str
    test_rmse: Measurement
    test_mae: Measurement
    train_rmse: Measurement
    fit_seconds: Measurement
    epochs_run: Measurement
    params: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "dataset": self.dataset,
            "test_rmse": self.test_rmse.mean,
            "test_rmse_ci95": self.test_rmse.ci95,
            "test_rmse_std": self.test_rmse.std,
            "test_mae": self.test_mae.mean,
            "train_rmse": self.train_rmse.mean,
            "fit_seconds": self.fit_seconds.mean,
            "epochs_run": self.epochs_run.mean,
            **{f"param_{k}": v for k, v in self.params.items()},
            **self.extra,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "dataset": self.dataset,
            "params": self.params,
            "test_rmse": self.test_rmse.as_dict(),
            "test_mae": self.test_mae.as_dict(),
            "train_rmse": self.train_rmse.as_dict(),
            "fit_seconds": self.fit_seconds.as_dict(),
            "epochs_run": self.epochs_run.as_dict(),
            "extra": self.extra,
        }


def make_split(
    dataset: str = "ml-100k", seed: int = 0, val_size: float = 0.1, test_size: float = 0.2
) -> Split:
    """Load a dataset and partition it. Cached per dataset within a process."""
    data = _load_cached(dataset)
    return train_val_test_split(data, val_size=val_size, test_size=test_size, seed=seed)


_CACHE: dict[str, RatingData] = {}


def _load_cached(dataset: str) -> RatingData:
    if dataset not in _CACHE:
        _CACHE[dataset] = load_movielens(dataset, auto_download=True)
    return _CACHE[dataset]


def evaluate(
    build: Callable[[int], Recommender],
    label: str,
    dataset: str = "ml-100k",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    use_validation: bool = True,
    params: dict[str, Any] | None = None,
    verbose: bool = True,
) -> EvaluationResult:
    """Fit ``build(seed)`` once per seed and summarise held-out performance.

    ``build`` receives the seed so that both the model's own randomness and the
    data split move together, giving genuinely independent replicates rather
    than the same split re-fitted.
    """
    test_rmse, test_mae, train_rmse, seconds, epochs = [], [], [], [], []

    for seed in seeds:
        split = make_split(dataset, seed=seed)
        model = build(seed)

        started = time.perf_counter()
        model.fit(split.train, split.val if use_validation else None)
        elapsed = time.perf_counter() - started

        predicted = model.predict_data(split.test)
        test_rmse.append(rmse(split.test.ratings, predicted))
        test_mae.append(mae(split.test.ratings, predicted))
        train_rmse.append(rmse(split.train.ratings, model.predict_data(split.train)))
        seconds.append(elapsed)
        epochs.append(float(getattr(model, "n_epochs_run", 0) or 0))

    result = EvaluationResult(
        label=label,
        dataset=dataset,
        test_rmse=aggregate(test_rmse),
        test_mae=aggregate(test_mae),
        train_rmse=aggregate(train_rmse),
        fit_seconds=aggregate(seconds),
        epochs_run=aggregate(epochs),
        params=params or {},
    )
    if verbose:
        print(
            f"  {label:38s} test RMSE {result.test_rmse}  "
            f"(train {result.train_rmse.mean:.4f}, {result.fit_seconds.mean:.1f}s)",
            flush=True,
        )
    return result


def environment() -> dict[str, Any]:
    """Provenance stamped into every result file."""
    import numpy

    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "numpy": numpy.__version__,
    }
    try:
        import torch

        record["torch"] = torch.__version__
    except ImportError:
        record["torch"] = None
    return record


def save_result(name: str, payload: dict[str, Any], directory: Path | None = None) -> Path:
    """Write one experiment's output to ``results/<name>.json`` with provenance."""
    directory = directory or RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    document = {"experiment": name, "environment": environment(), **payload}
    path.write_text(json.dumps(document, indent=2, default=_jsonable), encoding="utf-8")
    print(f"[saved] {path.relative_to(REPO_ROOT)}")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Measurement | EvaluationResult):
        return value.as_dict()
    raise TypeError(f"not JSON serialisable: {type(value)}")
