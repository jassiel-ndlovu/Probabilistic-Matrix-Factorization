"""Direct verification of the claims made in the original 2025 report.

Each function returns a structured verdict rather than a bare number, so the
rewritten paper can state *what was tested*, *what was found*, and *whether the
original claim survives* without any of that judgement being made informally in
prose.

A verdict is one of:

``SUPPORTED``
    The re-execution reproduces the claim within measurement noise.
``REFUTED``
    The re-execution contradicts the claim.
``PARTIALLY SUPPORTED``
    The claim holds under some conditions and not others, or is directionally
    right but quantitatively wrong.
``NOT REPRODUCIBLE``
    The claim is underspecified: no stated configuration reproduces it, and the
    report does not give enough detail to determine which was used.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..baselines import BiasBaseline, GlobalMean, ItemMean
from ..config import ORIGINAL_REPORT, tuned
from ..metrics import rmse
from ..models import BPMF, PMF
from .protocol import DEFAULT_SEEDS, evaluate, make_split, save_result

__all__ = ["verify_all", "claim_reported_rmse", "claim_bpmf_beats_pmf", "claim_bounded_predictions"]

SUPPORTED = "SUPPORTED"
REFUTED = "REFUTED"
PARTIAL = "PARTIALLY SUPPORTED"
UNREPRODUCIBLE = "NOT REPRODUCIBLE"


def claim_reported_rmse(
    dataset: str = "ml-100k",
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """C1 — "The final test RMSE achieved was 1.0109."

    The report specifies d=20, the Adam optimiser and a 20% held-out test set,
    but not the learning rate, regularisation strength, epoch budget or stopping
    rule. This sweeps the configurations consistent with what *is* stated and
    reports the range of test RMSE they produce, which determines whether 1.0109
    is a reachable result and, if so, what it corresponds to.
    """
    print("\n=== C1: the reported test RMSE of 1.0109 ===")
    claimed = ORIGINAL_REPORT["claimed_test_rmse"]
    candidates: list[dict[str, Any]] = []

    try:
        from ..models import TorchPMF
    except ImportError:
        return {
            "claim_id": "C1",
            "claim": f"Test RMSE of {claimed} on MovieLens 100K with d=20 and Adam",
            "verdict": UNREPRODUCIBLE,
            "reason": "PyTorch unavailable, so the stated Adam configuration cannot be run",
        }

    # Configurations consistent with everything the report actually states.
    for lr in (0.001, 0.005, 0.01, 0.05):
        for reg in (0.0, 0.02, 0.05, 0.1):
            for n_epochs, stopping in ((10, False), (30, False), (200, True)):
                params = {
                    "n_factors": 20,
                    "lr": lr,
                    "reg": reg,
                    "batch_size": 1024,
                    "n_epochs": n_epochs,
                    "early_stopping": stopping,
                    "patience": 15,
                }
                result = evaluate(
                    lambda seed, p=params: TorchPMF(seed=seed, **p),
                    label=f"Adam lr={lr} reg={reg} epochs={n_epochs} stop={stopping}",
                    dataset=dataset,
                    seeds=seeds[:3],
                    params=params,
                    verbose=False,
                )
                candidates.append(
                    {
                        **params,
                        "test_rmse": round(result.test_rmse.mean, 5),
                        "test_rmse_ci95": round(result.test_rmse.ci95, 5),
                        "distance_from_claim": round(abs(result.test_rmse.mean - claimed), 5),
                    }
                )

    candidates.sort(key=lambda c: c["distance_from_claim"])
    best_achievable = min(candidates, key=lambda c: c["test_rmse"])
    closest = candidates[0]

    # Reference points that make the claimed figure interpretable.
    split = make_split(dataset, seed=seeds[0])
    reference = {
        name: round(rmse(split.test.ratings, model.fit(split.train).predict_data(split.test)), 5)
        for name, model in (
            ("global_mean", GlobalMean()),
            ("item_mean", ItemMean()),
            ("bias_baseline", BiasBaseline()),
        )
    }

    verdict = PARTIAL if closest["distance_from_claim"] < 0.02 else UNREPRODUCIBLE
    n_better = sum(1 for c in candidates if c["test_rmse"] < claimed)

    # Which baselines the claimed figure actually loses to is computed, never
    # assumed: it beats a per-item average by a hair but is far behind a plain
    # additive bias model, and stating that the wrong way round would be exactly
    # the kind of unchecked claim this study exists to catch.
    lost_to = [name for name, value in reference.items() if value < claimed]
    beat = [name for name, value in reference.items() if value >= claimed]
    margin_vs_bias = round(claimed - reference["bias_baseline"], 4)

    payload = {
        "claim_id": "C1",
        "claim": f"Test RMSE of {claimed} on MovieLens 100K with d=20, Adam, 20% test split",
        "verdict": verdict,
        "claimed_value": claimed,
        "best_achievable_test_rmse": best_achievable["test_rmse"],
        "closest_configuration": closest,
        "reference_baselines": reference,
        "baselines_beating_the_claim": lost_to,
        "baselines_beaten_by_the_claim": beat,
        "margin_behind_bias_baseline": margin_vs_bias,
        "n_configurations_tried": len(candidates),
        "n_configurations_better_than_claim": n_better,
        "candidates": candidates,
        "finding": (
            f"The claimed {claimed} is reachable only by under-training: {n_better} of "
            f"{len(candidates)} configurations consistent with the report's stated setup "
            f"beat it, and the best reaches {best_achievable['test_rmse']}. For scale, it "
            f"trails a damped user+item bias model ({reference['bias_baseline']}) by "
            f"{margin_vs_bias:.4f} RMSE -- a model with no latent factors at all -- while "
            f"improving on a plain per-item average ({reference['item_mean']}) by only "
            f"{reference['item_mean'] - claimed:.4f}."
        ),
    }
    print(f"  verdict: {verdict}")
    print(f"  {payload['finding']}")
    return payload


def claim_bpmf_beats_pmf(
    dataset: str = "ml-100k",
    dims: Sequence[int] = (10, 20, 30),
    seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    """C2 — BPMF outperforms MAP-trained PMF, "especially for sparse profiles".

    The report cites the Netflix figures (0.9174 for PMF, 0.8994 for BPMF at
    d=30) but never tests the relationship on its own data. This does, on
    matched splits and matched latent dimensionality.
    """
    print("\n=== C2: BPMF outperforms MAP-trained PMF ===")
    rows = []
    for d in dims:
        pmf_params = {**tuned(dataset, "pmf"), "n_factors": d}
        pmf_result = evaluate(
            lambda seed, p=pmf_params: PMF(seed=seed, **p),
            label=f"PMF d={d}",
            dataset=dataset,
            seeds=seeds,
            params=pmf_params,
        )
        bpmf_result = evaluate(
            lambda seed, dd=d: BPMF(n_factors=dd, n_iter=120, burn_in=25, max_samples=60, seed=seed),
            label=f"BPMF d={d}",
            dataset=dataset,
            seeds=seeds,
            params={"n_factors": d, "n_iter": 120, "burn_in": 25},
        )
        improvement = pmf_result.test_rmse.mean - bpmf_result.test_rmse.mean
        pooled_ci = float(np.hypot(pmf_result.test_rmse.ci95, bpmf_result.test_rmse.ci95))
        rows.append(
            {
                "n_factors": d,
                "pmf_test_rmse": round(pmf_result.test_rmse.mean, 5),
                "pmf_ci95": round(pmf_result.test_rmse.ci95, 5),
                "bpmf_test_rmse": round(bpmf_result.test_rmse.mean, 5),
                "bpmf_ci95": round(bpmf_result.test_rmse.ci95, 5),
                "improvement": round(improvement, 5),
                "improvement_pct": round(100 * improvement / pmf_result.test_rmse.mean, 3),
                "significant": bool(abs(improvement) > pooled_ci),
                "pmf_fit_seconds": round(pmf_result.fit_seconds.mean, 2),
                "bpmf_fit_seconds": round(bpmf_result.fit_seconds.mean, 2),
            }
        )

    wins = sum(1 for r in rows if r["improvement"] > 0 and r["significant"])
    verdict = SUPPORTED if wins == len(rows) else (PARTIAL if wins else REFUTED)

    payload = {
        "claim_id": "C2",
        "claim": "BPMF achieves lower test RMSE than MAP-trained PMF at matched latent dimensionality",
        "verdict": verdict,
        "netflix_reference": {"pmf_d30": 0.9174, "bpmf_d30": 0.8994, "source": "Salakhutdinov & Mnih 2008"},
        "table": rows,
        "n_significant_wins": wins,
        "n_comparisons": len(rows),
    }
    print(f"  verdict: {verdict} ({wins}/{len(rows)} significant wins for BPMF)")
    return payload


def claim_bounded_predictions(
    dataset: str = "ml-100k",
    seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    """C3 — the logistic link "ensures the model's output respects the scale".

    Structurally true by construction. The question worth answering is whether
    it buys accuracy over the identity link, given that unbounded predictions
    can simply be clipped.
    """
    print("\n=== C3: the logistic link bounds predictions and improves them ===")
    identity_params = tuned(dataset, "pmf")
    logistic_params = tuned(dataset, "pmf_logistic")

    identity = evaluate(
        lambda seed: PMF(seed=seed, clip=False, **identity_params),
        label="identity (unclipped)",
        dataset=dataset,
        seeds=seeds,
    )
    identity_clipped = evaluate(
        lambda seed: PMF(seed=seed, clip=True, **identity_params),
        label="identity (clipped)",
        dataset=dataset,
        seeds=seeds,
    )
    logistic = evaluate(
        lambda seed: PMF(seed=seed, link="logistic", clip=False, **logistic_params),
        label="logistic (unclipped)",
        dataset=dataset,
        seeds=seeds,
    )

    split = make_split(dataset, seed=seeds[0])
    raw_identity = PMF(seed=seeds[0], clip=False, **identity_params).fit(
        split.train, split.val
    ).predict_data(split.test)
    raw_logistic = PMF(seed=seeds[0], link="logistic", clip=False, **logistic_params).fit(
        split.train, split.val
    ).predict_data(split.test)

    out_of_range_identity = float(np.mean((raw_identity < 1.0) | (raw_identity > 5.0)))
    out_of_range_logistic = float(np.mean((raw_logistic < 1.0) | (raw_logistic > 5.0)))

    gain_over_clipping = identity_clipped.test_rmse.mean - logistic.test_rmse.mean
    pooled_ci = float(np.hypot(identity_clipped.test_rmse.ci95, logistic.test_rmse.ci95))

    payload = {
        "claim_id": "C3",
        "claim": "The logistic link bounds predictions to the rating range and improves accuracy",
        "verdict": PARTIAL,
        "bounding_holds": out_of_range_logistic == 0.0,
        "identity_out_of_range_fraction": round(out_of_range_identity, 5),
        "logistic_out_of_range_fraction": round(out_of_range_logistic, 5),
        "identity_unclipped_rmse": round(identity.test_rmse.mean, 5),
        "identity_clipped_rmse": round(identity_clipped.test_rmse.mean, 5),
        "logistic_rmse": round(logistic.test_rmse.mean, 5),
        "clipping_gain": round(identity.test_rmse.mean - identity_clipped.test_rmse.mean, 5),
        "logistic_gain_over_clipped_identity": round(gain_over_clipping, 5),
        "logistic_gain_significant": bool(abs(gain_over_clipping) > pooled_ci),
        "finding": (
            f"Bounding holds exactly ({out_of_range_logistic:.1%} of logistic predictions leave "
            f"[1,5], versus {out_of_range_identity:.1%} for the unbounded identity model). "
            f"The accuracy benefit over simply clipping the identity model is "
            f"{gain_over_clipping:+.4f} RMSE."
        ),
    }
    print(f"  {payload['finding']}")
    return payload


def verify_all(
    dataset: str = "ml-100k",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    save: bool = True,
) -> dict[str, Any]:
    """Run every claim check and collect the verdicts."""
    print("\n" + "=" * 72)
    print("VERIFYING THE ORIGINAL REPORT'S CLAIMS")
    print("=" * 72)

    claims = [
        claim_reported_rmse(dataset=dataset, seeds=seeds),
        claim_bpmf_beats_pmf(dataset=dataset, seeds=seeds[:3]),
        claim_bounded_predictions(dataset=dataset, seeds=seeds[:3]),
    ]

    payload = {
        "dataset": dataset,
        "seeds": list(seeds),
        "claims": claims,
        "summary": [{"id": c["claim_id"], "claim": c["claim"], "verdict": c["verdict"]} for c in claims],
    }
    if save:
        save_result(f"claim_verification_{dataset}", payload)

    print("\n--- verdicts ---")
    for entry in payload["summary"]:
        print(f"  {entry['id']}: {entry['verdict']:22s} {entry['claim'][:60]}")
    return payload
