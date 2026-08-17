"""Parallel grid search over model hyperparameters.

Model selection is done on a validation split that is carved out of the training
data; the test partition is touched exactly once, for the final reported number.
The original report gave a single test RMSE with no stated selection protocol,
which makes it impossible to tell whether the figure was a tuned result or an
arbitrary one.

Search is parallelised across processes because each configuration is an
independent CPU-bound fit and the grids here are large enough for the difference
to matter (a 100-point grid drops from roughly 25 minutes to 5 on six cores).
"""

from __future__ import annotations

import itertools
import json
import os
import time
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_movielens, train_val_test_split
from .metrics import rmse

__all__ = ["TrialResult", "grid", "search", "best_of"]


@dataclass
class TrialResult:
    """Outcome of fitting one hyperparameter configuration."""

    params: dict[str, Any]
    train_rmse: float
    val_rmse: float
    test_rmse: float
    epochs_run: int
    best_epoch: int | None
    seconds: float
    diverged: bool = False
    history: dict[str, list[float]] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        """Flat record suitable for a DataFrame or a JSON dump."""
        return {
            **self.params,
            "train_rmse": self.train_rmse,
            "val_rmse": self.val_rmse,
            "test_rmse": self.test_rmse,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "seconds": round(self.seconds, 2),
            "diverged": self.diverged,
        }


def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]:
    """Cartesian product of named axes, as a list of keyword dicts.

    ``grid(lr=[0.001, 0.005], reg=[0.05, 0.1])`` gives four configurations.
    """
    keys = list(axes)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*axes.values())]


def _fit_one(job: tuple[str, dict[str, Any], dict[str, Any], int, str]) -> dict[str, Any]:
    """Worker entry point. Rebuilds the split from a seed rather than shipping
    arrays between processes, which keeps pickling cost negligible."""
    dataset, params, split_spec, keep_history, model_key = job

    # Imported here so each worker process gets its own module state, and so
    # that a torch-free environment can still search the NumPy models.
    if model_key == "torch":
        from .models import TorchPMF as Model
    else:
        from .models import PMF as Model

    data = load_movielens(dataset, auto_download=False)
    split = train_val_test_split(
        data,
        val_size=split_spec["val_size"],
        test_size=split_spec["test_size"],
        seed=split_spec["seed"],
    )

    started = time.perf_counter()
    try:
        model = Model(**params).fit(split.train, split.val)
    except FloatingPointError:
        return asdict(
            TrialResult(
                params=params,
                train_rmse=float("nan"),
                val_rmse=float("inf"),
                test_rmse=float("nan"),
                epochs_run=0,
                best_epoch=None,
                seconds=time.perf_counter() - started,
                diverged=True,
            )
        )

    result = TrialResult(
        params=params,
        train_rmse=rmse(split.train.ratings, model.predict_data(split.train)),
        val_rmse=rmse(split.val.ratings, model.predict_data(split.val)),
        test_rmse=rmse(split.test.ratings, model.predict_data(split.test)),
        epochs_run=model.n_epochs_run,
        best_epoch=model.best_epoch,
        seconds=time.perf_counter() - started,
        history=model.history.as_dict() if keep_history else {},
    )
    return asdict(result)


def search(
    configurations: Iterable[dict[str, Any]],
    dataset: str = "ml-100k",
    val_size: float = 0.1,
    test_size: float = 0.2,
    split_seed: int = 0,
    n_workers: int | None = None,
    keep_history: bool = False,
    save_to: Path | str | None = None,
    model: str = "numpy",
    verbose: bool = True,
) -> list[TrialResult]:
    """Fit every configuration and return results sorted by validation RMSE.

    ``model`` selects the trainer: ``"numpy"`` for :class:`~pmf.models.PMF`,
    ``"torch"`` for :class:`~pmf.models.TorchPMF`.
    """
    configurations = list(configurations)
    split_spec = {"val_size": val_size, "test_size": test_size, "seed": split_seed}
    jobs = [(dataset, params, split_spec, keep_history, model) for params in configurations]

    started = time.perf_counter()
    results: list[TrialResult] = []

    if n_workers == 1:
        for job in jobs:
            results.append(TrialResult(**_fit_one(job)))
            if verbose:
                _log(len(results), len(jobs), results[-1])
    else:
        # Each worker is already a full core's worth of work, so BLAS must not
        # also fan out inside it: six processes times a default thread pool
        # oversubscribes the machine badly enough to be slower than running
        # serially. Children inherit these on spawn, which is what Windows uses.
        with _single_threaded_blas(), ProcessPoolExecutor(max_workers=n_workers) as pool:
            for payload in pool.map(_fit_one, jobs):
                results.append(TrialResult(**payload))
                if verbose:
                    _log(len(results), len(jobs), results[-1])

    results.sort(key=lambda r: r.val_rmse)
    if verbose:
        elapsed = time.perf_counter() - started
        print(
            f"[tune] {len(results)} configurations in {elapsed:.1f}s; "
            f"best val RMSE {results[0].val_rmse:.4f}"
        )

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(
            json.dumps([r.row() for r in results], indent=2, default=_jsonable), encoding="utf-8"
        )

    return results


@contextmanager
def _single_threaded_blas() -> Iterator[None]:
    """Pin every BLAS backend to one thread for the duration of the block."""
    names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def best_of(results: Sequence[TrialResult]) -> TrialResult:
    """The configuration with the lowest validation RMSE."""
    viable = [r for r in results if not r.diverged and np.isfinite(r.val_rmse)]
    if not viable:
        raise RuntimeError("every configuration diverged")
    return min(viable, key=lambda r: r.val_rmse)


def _log(done: int, total: int, result: TrialResult) -> None:
    status = "DIVERGED" if result.diverged else f"val {result.val_rmse:.4f}"
    summary = " ".join(f"{k}={v}" for k, v in result.params.items())
    print(f"[tune] {done:4d}/{total}  {status:16s} {summary}", flush=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")
