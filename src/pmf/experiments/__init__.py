"""The experimental study.

Each submodule owns one family of questions:

``protocol``
    Shared machinery — repeated splits, aggregation with confidence intervals,
    result persistence with provenance.
``comparison``
    Model against model, on rating prediction and on ranking.
``ablations``
    What each modelling choice contributes: latent dimensionality,
    regularisation strength, link function, optimiser.
``diagnostics``
    Where models fail (cold start, sparsity) and how cost grows (scalability).
``claims``
    Direct verification of the assertions in the original report.
"""

from __future__ import annotations

from .ablations import convergence, latent_dimension, link_function, optimiser, regularisation
from .claims import verify_all
from .comparison import model_comparison, ranking_comparison
from .diagnostics import cold_start, scalability, sparsity_curve
from .protocol import RESULTS_DIR, Measurement, aggregate, evaluate, make_split, save_result

__all__ = [
    "RESULTS_DIR",
    "Measurement",
    "aggregate",
    "cold_start",
    "convergence",
    "evaluate",
    "latent_dimension",
    "link_function",
    "make_split",
    "model_comparison",
    "optimiser",
    "ranking_comparison",
    "regularisation",
    "save_result",
    "scalability",
    "sparsity_curve",
    "verify_all",
]
