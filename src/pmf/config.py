"""Selected hyperparameters, in one place.

These are not guesses. Each entry is the configuration that minimised
*validation* RMSE in the grid search recorded in
``results/hyperparameter_search_<dataset>.json``, which can be regenerated with::

    python scripts/tune.py --dataset ml-100k

Keeping them here rather than inline in each experiment means the paper, the
README, the notebook and the experiment scripts all quote the same numbers, and
that re-tuning is a one-file change.

A note on why the values differ so much between ``reg_mode`` settings: under
``"uniform"`` the prior competes with a data gradient that has accumulated
``n_i`` observations for user *i*, so the useful lambda is roughly ``n_i`` times
larger and does not transfer between datasets. All entries below use
``"per_observation"``, whose lambda does transfer.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

__all__ = ["TUNED", "tuned"]

TUNED: dict[str, dict[str, dict[str, Any]]] = {
    # Selected on validation RMSE; see results/hyperparameter_search_ml-100k_*.json.
    # Note how far apart the identity and logistic lambdas sit (0.1 vs 0.005):
    # the logistic gradient carries a factor g(x)(1-g(x)) <= 1/4 on the data term
    # but not on the prior, so the same nominal lambda regularises far harder.
    "ml-100k": {
        "pmf": {
            "n_factors": 20,
            "lr": 0.0005,
            "reg": 0.1,
            "momentum": 0.9,
            "batch_size": 256,
            "n_epochs": 600,
            "patience": 25,
        },
        "pmf_bias": {
            "n_factors": 20,
            "lr": 0.001,
            "reg": 0.1,
            "momentum": 0.9,
            "batch_size": 256,
            "n_epochs": 600,
            "patience": 25,
        },
        "pmf_logistic": {
            "n_factors": 20,
            "lr": 0.02,
            "reg": 0.005,
            "momentum": 0.9,
            "batch_size": 256,
            "n_epochs": 600,
            "patience": 25,
        },
        "pmf_adam": {
            "n_factors": 20,
            "lr": 0.01,
            "reg": 0.1,
            "batch_size": 1024,
            "n_epochs": 600,
            "patience": 25,
        },
    },
    # ML-1M entries are TRANSFERRED from the ML-100K search rather than
    # re-tuned, with two deliberate changes: batch_size is raised to 4096 and
    # eval_every to 5, both purely for speed (see PMF.batch_size). This is
    # stated as a limitation in the paper.
    #
    # The transfer is defensible specifically because reg_mode is
    # "per_observation": lambda then multiplies a per-rating penalty, so its
    # useful scale does not move with dataset size or density. Under the
    # uniform-lambda form it would not transfer and a re-search would be
    # mandatory. The learning rate is left unchanged because it is likewise a
    # per-observation rate under the summed-gradient update.
    "ml-1m": {
        "pmf": {
            "n_factors": 20,
            "lr": 0.0005,
            "reg": 0.1,
            "momentum": 0.9,
            "batch_size": 4096,
            "n_epochs": 120,
            "patience": 12,
            "eval_every": 5,
        },
        "pmf_bias": {
            "n_factors": 20,
            "lr": 0.001,
            "reg": 0.1,
            "momentum": 0.9,
            "batch_size": 4096,
            "n_epochs": 120,
            "patience": 12,
            "eval_every": 5,
        },
        "pmf_logistic": {
            "n_factors": 20,
            "lr": 0.02,
            "reg": 0.005,
            "momentum": 0.9,
            "batch_size": 4096,
            "n_epochs": 120,
            "patience": 12,
            "eval_every": 5,
        },
        "pmf_adam": {
            "n_factors": 20,
            "lr": 0.01,
            "reg": 0.1,
            "batch_size": 4096,
            "n_epochs": 150,
            "patience": 15,
        },
    },
}

#: Settings quoted in the original 2025 report, reproduced verbatim for the
#: claim-verification experiment. The report specifies only the latent
#: dimensionality, the optimiser and the test fraction.
ORIGINAL_REPORT = {
    "n_factors": 20,
    "optimiser": "adam",
    "test_size": 0.2,
    "claimed_test_rmse": 1.0109,
}


def tuned(dataset: str, model: str) -> dict[str, Any]:
    """Selected hyperparameters for ``model`` on ``dataset``.

    Returns a copy, so a caller adding ``seed`` or overriding ``n_factors``
    cannot mutate the shared table.
    """
    try:
        return deepcopy(TUNED[dataset][model])
    except KeyError:
        known = {d: sorted(m) for d, m in TUNED.items()}
        raise KeyError(f"no tuned settings for {model!r} on {dataset!r}; have {known}") from None
