"""Ablations: what each modelling choice is actually worth."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..config import tuned
from ..metrics import rmse
from ..models import PMF
from .protocol import DEFAULT_SEEDS, evaluate, make_split, save_result

__all__ = ["latent_dimension", "regularisation", "link_function", "optimiser", "convergence"]


#: Epoch budget for the arms that run with early stopping switched off.
#:
#: The tuned configurations carry a large ``n_epochs`` so that early stopping
#: has room to find its own minimum. Inheriting that budget for a deliberately
#: unstopped run is pure waste: validation error turns upward inside the first
#: hundred epochs at every setting examined, so the remainder only makes the
#: sweep take hours while adding nothing to the conclusion.
UNSTOPPED_EPOCHS = 150


def latent_dimension(
    dataset: str = "ml-100k",
    dims: Sequence[int] = (2, 5, 10, 20, 30, 50, 100),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    save: bool = True,
) -> dict[str, Any]:
    """Test RMSE as a function of :math:`d`, with and without early stopping.

    The paper reports monotone gains from larger :math:`d` on Netflix. On a
    dataset three orders of magnitude smaller the capacity/data balance is
    completely different, so this measures where the curve actually turns —
    and whether early stopping is what flattens it.
    """
    print(f"\n=== latent dimension sweep on {dataset} ===")
    base = tuned(dataset, "pmf")
    unstopped_params = {**base, "n_epochs": UNSTOPPED_EPOCHS}
    rows = []

    for d in dims:
        params = {**base, "n_factors": d}
        stopped = evaluate(
            lambda seed, p=params: PMF(seed=seed, **p),
            label=f"d={d} (early stopping)",
            dataset=dataset,
            seeds=seeds,
            params=params,
        )
        unstopped = evaluate(
            lambda seed, p={**unstopped_params, "n_factors": d}: PMF(
                seed=seed, early_stopping=False, **p
            ),
            label=f"d={d} (no early stopping)",
            dataset=dataset,
            seeds=seeds,
            params=params,
        )
        rows.append(
            {
                "n_factors": d,
                "test_rmse": round(stopped.test_rmse.mean, 5),
                "test_rmse_ci95": round(stopped.test_rmse.ci95, 5),
                "train_rmse": round(stopped.train_rmse.mean, 5),
                "epochs_run": round(stopped.epochs_run.mean, 1),
                "fit_seconds": round(stopped.fit_seconds.mean, 2),
                "test_rmse_no_early_stop": round(unstopped.test_rmse.mean, 5),
                "train_rmse_no_early_stop": round(unstopped.train_rmse.mean, 5),
                "generalisation_gap": round(
                    stopped.test_rmse.mean - stopped.train_rmse.mean, 5
                ),
            }
        )

    best = min(rows, key=lambda r: r["test_rmse"])
    payload = {
        "dataset": dataset,
        "seeds": list(seeds),
        "base_params": base,
        "table": rows,
        "best_n_factors": best["n_factors"],
        "best_test_rmse": best["test_rmse"],
    }
    if save:
        save_result(f"latent_dimension_{dataset}", payload)
    return payload


def regularisation(
    dataset: str = "ml-100k",
    regs: Sequence[float] = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
    n_factors: int = 30,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    save: bool = True,
) -> dict[str, Any]:
    """Train and test RMSE against :math:`\\lambda`, with early stopping off.

    Early stopping is disabled deliberately: it is itself a regulariser, and
    leaving it on hides exactly the overfitting this sweep is meant to expose.
    This is the experiment that tests the report's claim that PMF "requires
    careful parameter tuning to avoid overfitting".
    """
    print(f"\n=== regularisation sweep on {dataset} (d={n_factors}, no early stopping) ===")
    base = {**tuned(dataset, "pmf"), "n_factors": n_factors, "n_epochs": UNSTOPPED_EPOCHS}
    rows = []

    for reg in regs:
        params = {**base, "reg": reg}
        result = evaluate(
            lambda seed, p=params: PMF(seed=seed, early_stopping=False, **p),
            label=f"lambda={reg}",
            dataset=dataset,
            seeds=seeds,
            params=params,
        )
        rows.append(
            {
                "reg": reg,
                "train_rmse": round(result.train_rmse.mean, 5),
                "test_rmse": round(result.test_rmse.mean, 5),
                "test_rmse_ci95": round(result.test_rmse.ci95, 5),
                "generalisation_gap": round(
                    result.test_rmse.mean - result.train_rmse.mean, 5
                ),
            }
        )

    best = min(rows, key=lambda r: r["test_rmse"])
    unregularised = next(r for r in rows if r["reg"] == 0.0) if 0.0 in regs else None
    payload = {
        "dataset": dataset,
        "n_factors": n_factors,
        "seeds": list(seeds),
        "table": rows,
        "best_reg": best["reg"],
        "best_test_rmse": best["test_rmse"],
        "cost_of_no_regularisation": (
            round(unregularised["test_rmse"] - best["test_rmse"], 5) if unregularised else None
        ),
    }
    if save:
        save_result(f"regularisation_{dataset}", payload)
    return payload


def link_function(
    dataset: str = "ml-100k",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    save: bool = True,
) -> dict[str, Any]:
    """Identity versus logistic link, with and without prediction clipping.

    The report asserts the sigmoid variant "ensures the model's output respects
    the scale of the ratings and avoids extrapolation". That is structurally
    true, but the interesting question is whether it is *worth* anything once
    the far cheaper alternative — clipping the identity model's output to
    [1, 5] — is applied. This measures both.
    """
    print(f"\n=== link function on {dataset} ===")
    identity = tuned(dataset, "pmf")
    logistic = tuned(dataset, "pmf_logistic")

    variants = {
        "identity, clipped": lambda seed: PMF(seed=seed, clip=True, **identity),
        "identity, unclipped": lambda seed: PMF(seed=seed, clip=False, **identity),
        "logistic, clipped": lambda seed: PMF(seed=seed, link="logistic", clip=True, **logistic),
        "logistic, unclipped": lambda seed: PMF(seed=seed, link="logistic", clip=False, **logistic),
    }

    rows = []
    for label, build in variants.items():
        result = evaluate(build, label=label, dataset=dataset, seeds=seeds)

        # How often does the unbounded model leave the valid range at all?
        split = make_split(dataset, seed=seeds[0])
        model = build(seeds[0]).fit(split.train, split.val)
        raw = model.predict_data(split.test)
        below = raw < split.test.rating_min - 1e-9
        above = raw > split.test.rating_max + 1e-9
        out_of_range = float(np.mean(below | above))

        rows.append(
            {
                "variant": label,
                "test_rmse": round(result.test_rmse.mean, 5),
                "test_rmse_ci95": round(result.test_rmse.ci95, 5),
                "test_mae": round(result.test_mae.mean, 5),
                "out_of_range_fraction": round(out_of_range, 5),
                "min_prediction": round(float(raw.min()), 4),
                "max_prediction": round(float(raw.max()), 4),
            }
        )

    payload = {"dataset": dataset, "seeds": list(seeds), "table": rows}
    if save:
        save_result(f"link_function_{dataset}", payload)
    return payload


def optimiser(
    dataset: str = "ml-100k",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    save: bool = True,
) -> dict[str, Any]:
    """SGD with momentum (the paper) versus Adam (the original report).

    Same objective, same latent dimensionality, same splits — only the
    optimiser differs, which is what makes the comparison attributable.
    """
    print(f"\n=== optimiser comparison on {dataset} ===")
    rows = []

    sgd_params = tuned(dataset, "pmf")
    sgd = evaluate(
        lambda seed: PMF(seed=seed, **sgd_params),
        label="SGD + momentum (paper)",
        dataset=dataset,
        seeds=seeds,
        params=sgd_params,
    )
    rows.append(_optimiser_row("SGD + momentum", sgd))

    try:
        from ..models import TorchPMF
    except ImportError:
        print("  [skip] PyTorch unavailable")
    else:
        adam_params = tuned(dataset, "pmf_adam")
        adam = evaluate(
            lambda seed: TorchPMF(seed=seed, **adam_params),
            label="Adam (original report)",
            dataset=dataset,
            seeds=seeds,
            params=adam_params,
        )
        rows.append(_optimiser_row("Adam", adam))

    payload = {"dataset": dataset, "seeds": list(seeds), "table": rows}
    if save:
        save_result(f"optimiser_{dataset}", payload)
    return payload


def _optimiser_row(label: str, result) -> dict[str, Any]:
    return {
        "optimiser": label,
        "test_rmse": round(result.test_rmse.mean, 5),
        "test_rmse_ci95": round(result.test_rmse.ci95, 5),
        "train_rmse": round(result.train_rmse.mean, 5),
        "epochs_to_best": round(result.epochs_run.mean, 1),
        "fit_seconds": round(result.fit_seconds.mean, 2),
        "params": result.params,
    }


def convergence(
    dataset: str = "ml-100k",
    seed: int = 0,
    n_epochs: int = 200,
    save: bool = True,
) -> dict[str, Any]:
    """Full per-epoch traces for the learning-curve figure.

    Records the MAP objective and the *unpenalised* train and validation RMSE as
    three separate series. The original implementation reported a single blended
    quantity as "training RMSE", which makes a learning curve drawn from it
    uninterpretable.
    """
    print(f"\n=== convergence trace on {dataset} ===")
    split = make_split(dataset, seed=seed)
    traces = {}

    # The middle trace uses the lambda the validation search actually selected,
    # rather than a hard-coded value that might drift away from it.
    selected = tuned(dataset, "pmf")["reg"]
    regimes = {
        "under-regularised (lambda=0)": 0.0,
        f"selected (lambda={selected:g})": selected,
        f"over-regularised (lambda={5 * selected:g})": 5 * selected,
    }

    for label, reg in regimes.items():
        params = {**tuned(dataset, "pmf"), "n_factors": 30, "n_epochs": n_epochs, "reg": reg}
        model = PMF(seed=seed, early_stopping=False, **params).fit(split.train, split.val)
        history = model.history.as_dict()
        history["test_rmse_final"] = rmse(split.test.ratings, model.predict_data(split.test))
        traces[label] = history
        print(f"  {label:32s} final val RMSE {history['val_rmse'][-1]:.4f}")

    payload = {"dataset": dataset, "seed": seed, "n_epochs": n_epochs, "traces": traces}
    if save:
        save_result(f"convergence_{dataset}", payload)
    return payload
