"""Re-execute the original forked implementation, unmodified.

This drives ``legacy/ProbabilisticMatrixFactorization.py`` exactly as
``legacy/RunExample.py`` does, but non-interactively and over multiple seeds, so
the report can state what that code actually produces rather than what the
original submission claimed it produced.

Two things are recorded that the original run could not report:

1. The value the legacy class calls ``rmse_train`` alongside the *true*
   training RMSE recomputed from its own factors. The gap between them is the
   regularisation penalty that the legacy code folds into the metric.
2. Spread across random seeds. The legacy code seeds nothing, so its published
   single number carries an unquantified run-to-run variance.

Usage::

    python scripts/reproduce_legacy.py --seeds 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "legacy"))

from ProbabilisticMatrixFactorization import PMF as LegacyPMF  # noqa: E402

from pmf.data import load_movielens  # noqa: E402
from pmf.metrics import rmse  # noqa: E402


def legacy_split(triples: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The split used by ``RunExample.py`` (sklearn's ``train_test_split``),
    reproduced with a seeded permutation so the comparison is repeatable."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(triples.shape[0])
    n_test = int(round(test_size * triples.shape[0]))
    return triples[perm[n_test:]], triples[perm[:n_test]]


def run_once(seed: int, n_factors: int, epochs: int) -> dict[str, float]:
    """One legacy fit with the original example's hyperparameters."""
    # The legacy loader reads raw 1-based ids; the class then sizes its factor
    # matrices as max(id)+1. Feed it the raw ids it expects.
    data = load_movielens("ml-100k", auto_download=False, with_titles=False)
    raw = data.triples.copy()
    raw[:, 0] += 1
    raw[:, 1] += 1

    train, test = legacy_split(raw, test_size=0.2, seed=seed)

    np.random.seed(seed)  # the legacy class uses the global numpy RNG
    model = LegacyPMF()
    model.set_params(
        {
            "num_feat": n_factors,
            "epsilon": 1,
            "_lambda": 0.1,
            "momentum": 0.8,
            "maxepoch": epochs,
            "num_batches": 100,
            "batch_size": 1000,
        }
    )
    model.fit(train, test)

    # Recompute the true (unpenalised) training RMSE from the legacy factors.
    users = train[:, 0].astype(int)
    items = train[:, 1].astype(int)
    predicted = (
        np.sum(model.w_User[users] * model.w_Item[items], axis=1) + model.mean_inv
    )
    true_train_rmse = rmse(train[:, 2], predicted)

    return {
        "seed": seed,
        "reported_train_rmse": float(model.rmse_train[-1]),
        "true_train_rmse": float(true_train_rmse),
        "test_rmse": float(model.rmse_test[-1]),
        "penalty_inflation": float(model.rmse_train[-1] - true_train_rmse),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--factors", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "legacy_reproduction.json")
    args = parser.parse_args()

    runs = []
    for seed in range(args.seeds):
        print(f"--- seed {seed} ---")
        runs.append(run_once(seed, args.factors, args.epochs))

    test_scores = np.array([r["test_rmse"] for r in runs])
    reported = np.array([r["reported_train_rmse"] for r in runs])
    true_train = np.array([r["true_train_rmse"] for r in runs])

    summary = {
        "config": {
            "source": "legacy/ProbabilisticMatrixFactorization.py",
            "num_feat": args.factors,
            "epsilon": 1,
            "_lambda": 0.1,
            "momentum": 0.8,
            "maxepoch": args.epochs,
            "num_batches": 100,
            "batch_size": 1000,
            "n_seeds": args.seeds,
        },
        "test_rmse_mean": round(float(test_scores.mean()), 4),
        "test_rmse_std": round(float(test_scores.std(ddof=1)), 4),
        "test_rmse_min": round(float(test_scores.min()), 4),
        "test_rmse_max": round(float(test_scores.max()), 4),
        "reported_train_rmse_mean": round(float(reported.mean()), 4),
        "true_train_rmse_mean": round(float(true_train.mean()), 4),
        "penalty_inflation_mean": round(float((reported - true_train).mean()), 4),
        "runs": runs,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== legacy implementation, as published ===")
    print(f"test RMSE            {summary['test_rmse_mean']:.4f} +/- {summary['test_rmse_std']:.4f} "
          f"(min {summary['test_rmse_min']:.4f}, max {summary['test_rmse_max']:.4f})")
    print(f"'train RMSE' printed {summary['reported_train_rmse_mean']:.4f}")
    print(f"true train RMSE      {summary['true_train_rmse_mean']:.4f}")
    print(f"inflation from penalty {summary['penalty_inflation_mean']:.4f}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
