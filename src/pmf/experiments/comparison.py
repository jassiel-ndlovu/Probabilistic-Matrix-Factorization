"""Head-to-head model comparison, on rating prediction and on ranking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..baselines import BiasBaseline, GlobalMean, ItemMean, UserMean
from ..config import tuned
from ..metrics import ranking_metrics
from ..models import BPMF, PMF
from .protocol import DEFAULT_SEEDS, EvaluationResult, aggregate, evaluate, make_split, save_result

__all__ = ["model_comparison", "ranking_comparison"]


def _builders(dataset: str) -> dict[str, Any]:
    """Model constructors keyed by display label.

    Hyperparameters come from :mod:`pmf.config`, which records what the
    validation search selected, so this table stays a description of *which*
    models are compared rather than a second, drifting copy of *how* they are
    configured.
    """
    pmf_kwargs = tuned(dataset, "pmf")
    bias_kwargs = tuned(dataset, "pmf_bias")
    logistic_kwargs = tuned(dataset, "pmf_logistic")

    return {
        "Global mean": lambda seed: GlobalMean(),
        "User mean": lambda seed: UserMean(),
        "Item mean": lambda seed: ItemMean(),
        "Bias baseline": lambda seed: BiasBaseline(damping=10.0),
        "PMF": lambda seed: PMF(seed=seed, **pmf_kwargs),
        "PMF + logistic link": lambda seed: PMF(seed=seed, link="logistic", **logistic_kwargs),
        "PMF + biases": lambda seed: PMF(seed=seed, use_bias=True, **bias_kwargs),
        "BPMF": lambda seed: BPMF(
            n_factors=pmf_kwargs["n_factors"], n_iter=120, burn_in=25, max_samples=60, seed=seed
        ),
    }


def model_comparison(
    dataset: str = "ml-100k",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    include_torch: bool = True,
    only: Sequence[str] | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Fit every model on the same splits and report test RMSE and MAE.

    The baselines are not decoration: they set the scale against which a
    factorisation's RMSE has to be read.

    ``only`` restricts the roster to the named labels. This exists for ML-1M,
    where a single PMF fit takes ~10 minutes and Gibbs sampling is slower still,
    so the full roster over five seeds is not affordable; the reduced roster
    still answers the question that dataset is there to answer, namely whether
    the model *ordering* survives an order of magnitude more data.
    """
    print(f"\n=== model comparison on {dataset} ({len(seeds)} seeds) ===")
    builders = _builders(dataset)

    if include_torch:
        try:
            from ..models import TorchPMF

            adam_kwargs = tuned(dataset, "pmf_adam")
            builders["PMF (Adam)"] = lambda seed: TorchPMF(seed=seed, **adam_kwargs)
        except ImportError:
            print("  [skip] PyTorch unavailable; omitting the Adam variant")

    if only is not None:
        unknown = set(only) - set(builders)
        if unknown:
            raise KeyError(f"unknown model labels {sorted(unknown)}; have {sorted(builders)}")
        builders = {label: build for label, build in builders.items() if label in only}

    results: list[EvaluationResult] = []
    for label, build in builders.items():
        params = _describe(build(0))
        results.append(
            evaluate(build, label=label, dataset=dataset, seeds=seeds, params=params)
        )

    results.sort(key=lambda r: r.test_rmse.mean)
    payload = {
        "dataset": dataset,
        "seeds": list(seeds),
        "roster": sorted(builders),
        "reduced_roster": only is not None,
        "results": [r.as_dict() for r in results],
        "table": [r.row() for r in results],
    }
    if save:
        save_result(f"model_comparison_{dataset}", payload)
    return payload


def ranking_comparison(
    dataset: str = "ml-100k",
    seeds: Sequence[int] = (0, 1, 2),
    k: int = 10,
    relevance_threshold: float = 4.0,
    max_users: int | None = 1500,
    save: bool = True,
) -> dict[str, Any]:
    """Top-``k`` precision, recall and NDCG for each model.

    Reported alongside RMSE because the two need not agree: a model can win on
    squared error while ranking worse, and a recommender is deployed for its
    ranking. Items seen in training are masked before ranking.
    """
    print(f"\n=== ranking comparison on {dataset} (k={k}, {len(seeds)} seeds) ===")
    builders = _builders(dataset)

    collected: dict[str, dict[str, list[float]]] = {
        label: {"precision": [], "recall": [], "ndcg": []} for label in builders
    }

    for seed in seeds:
        split = make_split(dataset, seed=seed)
        for label, build in builders.items():
            model = build(seed).fit(split.train, split.val)
            scores = ranking_metrics(
                model,
                train_users=split.train.users,
                train_items=split.train.items,
                test_users=split.test.users,
                test_items=split.test.items,
                test_ratings=split.test.ratings,
                k=k,
                relevance_threshold=relevance_threshold,
                n_items=split.train.n_items,
                max_users=max_users,
                seed=seed,
            )
            collected[label]["precision"].append(scores.precision)
            collected[label]["recall"].append(scores.recall)
            collected[label]["ndcg"].append(scores.ndcg)

    table = []
    for label, series in collected.items():
        row = {"label": label}
        for metric, values in series.items():
            summary = aggregate(values)
            row[f"{metric}@{k}"] = round(summary.mean, 5)
            row[f"{metric}@{k}_ci95"] = round(summary.ci95, 5)
        table.append(row)

    table.sort(key=lambda r: -r[f"ndcg@{k}"])
    payload = {
        "dataset": dataset,
        "k": k,
        "relevance_threshold": relevance_threshold,
        "seeds": list(seeds),
        "max_users_per_seed": max_users,
        "table": table,
    }
    if save:
        save_result(f"ranking_comparison_{dataset}", payload)
    return payload


def _describe(model: Any) -> dict[str, Any]:
    """Extract the hyperparameters worth recording next to a result."""
    keys = ("n_factors", "lr", "reg", "reg_mode", "momentum", "batch_size", "link", "use_bias",
            "n_iter", "burn_in", "alpha", "damping")
    return {
        key: _plain(getattr(model, key))
        for key in keys
        if hasattr(model, key)
    }


def _plain(value: Any) -> Any:
    if isinstance(value, np.integer | np.floating):
        return value.item()
    return value
