"""Run the full experimental study and write ``results/*.json``.

Every number in the paper, the README and the HTML summary comes from here.
Figures are generated separately by ``scripts/make_figures.py``, which reads the
JSON this produces — so a figure can never disagree with a table.

Usage::

    python scripts/run_experiments.py                 # everything, ML-100K + ML-1M
    python scripts/run_experiments.py --quick         # fewer seeds, ML-100K only
    python scripts/run_experiments.py --only claims scalability
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

warnings.filterwarnings("ignore", category=RuntimeWarning)

from pmf.data import describe, load_movielens  # noqa: E402
from pmf.experiments import (  # noqa: E402
    ablations,
    claims,
    comparison,
    diagnostics,
)
from pmf.experiments.protocol import save_result  # noqa: E402

STAGES = (
    "datasets",
    "convergence",
    "comparison",
    "latent_dimension",
    "regularisation",
    "link_function",
    "optimiser",
    "cold_start",
    "sparsity",
    "ranking",
    "scalability",
    "claims",
)


def stage_datasets(datasets: list[str], **_) -> None:
    """Dataset statistics table."""
    print("\n=== dataset statistics ===")
    rows = []
    for key in datasets:
        stats = describe(load_movielens(key, auto_download=True))
        rows.append(stats)
        print(f"  {key}: {stats['n_users']} users, {stats['n_items']} items, "
              f"{stats['n_ratings']} ratings, {100 * (1 - stats['sparsity']):.2f}% dense")
    save_result("dataset_statistics", {"table": rows})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="3 seeds, ML-100K only")
    parser.add_argument("--only", nargs="+", choices=STAGES, help="run just these stages")
    parser.add_argument("--skip", nargs="+", choices=STAGES, default=[])
    args = parser.parse_args()

    # The headline comparison carries the most weight, so it gets the most
    # replicates. Ablations are about the shape of a curve rather than about
    # separating two models by a few thousandths, and each point in them costs
    # a full fit, so three seeds buys the trend at a third of the compute.
    seeds = (0, 1, 2) if args.quick else (0, 1, 2, 3, 4)
    ablation_seeds = seeds[:3]
    datasets = ["ml-100k"] if args.quick else ["ml-100k", "ml-1m"]
    selected = set(args.only) if args.only else set(STAGES)
    selected -= set(args.skip)

    print(f"seeds={seeds}  ablation_seeds={ablation_seeds}  "
          f"datasets={datasets}  stages={sorted(selected)}")
    started = time.perf_counter()

    if "datasets" in selected:
        stage_datasets(datasets)

    if "convergence" in selected:
        ablations.convergence(dataset="ml-100k", seed=0)

    if "comparison" in selected:
        comparison.model_comparison(dataset="ml-100k", seeds=seeds)
        if "ml-1m" in datasets:
            # A single PMF fit on ML-1M takes ~10 minutes and Gibbs sampling far
            # longer, so this runs a reduced roster over fewer seeds. It is
            # there to show the ordering holds at scale, not to tune on 1M.
            comparison.model_comparison(
                dataset="ml-1m",
                seeds=seeds[:3],
                include_torch=False,
                only=["Item mean", "User mean", "Bias baseline", "PMF", "PMF + biases"],
            )

    if "latent_dimension" in selected:
        dims = (2, 5, 10, 20, 30, 50) if args.quick else (2, 5, 10, 20, 30, 50, 100)
        ablations.latent_dimension(dataset="ml-100k", dims=dims, seeds=ablation_seeds)

    if "regularisation" in selected:
        ablations.regularisation(dataset="ml-100k", seeds=ablation_seeds)

    if "link_function" in selected:
        ablations.link_function(dataset="ml-100k", seeds=ablation_seeds)

    if "optimiser" in selected:
        ablations.optimiser(dataset="ml-100k", seeds=seeds)

    if "cold_start" in selected:
        diagnostics.cold_start(dataset="ml-100k", seeds=ablation_seeds)

    if "sparsity" in selected:
        diagnostics.sparsity_curve(dataset="ml-100k", seeds=ablation_seeds)

    if "ranking" in selected:
        comparison.ranking_comparison(dataset="ml-100k", seeds=seeds[:3])

    if "scalability" in selected:
        diagnostics.scalability(datasets=tuple(datasets), seeds=seeds[:3])

    if "claims" in selected:
        claims.verify_all(dataset="ml-100k", seeds=seeds)

    elapsed = time.perf_counter() - started
    print(f"\n=== finished in {elapsed / 60:.1f} min; results in {REPO_ROOT / 'results'} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
