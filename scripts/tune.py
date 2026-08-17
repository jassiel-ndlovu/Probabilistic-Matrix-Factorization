"""Select hyperparameters on the validation split.

Writes ``results/hyperparameter_search_<dataset>.json`` and prints the top
configurations. The winning entries are what belongs in :mod:`pmf.config`.

Usage::

    python scripts/tune.py --dataset ml-100k
    python scripts/tune.py --dataset ml-1m --workers 6
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

warnings.filterwarnings("ignore", category=RuntimeWarning)

from pmf.tuning import best_of, grid, search  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"

#: Ranges are centred on where the coarse probe found the optimum. The learning
#: rate matters most: above ~0.01 the momentum update diverges, and below
#: ~0.0005 the epoch budget binds before convergence does.
SPACE = {
    "ml-100k": {
        "lr": [0.0002, 0.0005, 0.001, 0.002],
        "reg": [0.05, 0.1, 0.15, 0.2],
        "momentum": [0.9],
        "batch_size": [256, 1024],
        "n_epochs": [600],
        "patience": [25],
    },
    # ML-1M is searched over a tighter grid centred on the ML-100K optimum. The
    # per-observation regulariser makes lambda transferable across dataset
    # sizes (unlike the uniform form), so a wide re-search is not warranted.
    # batch_size starts at 4096: the momentum update is dense over all factor
    # rows, so per-epoch cost scales with batch *count*, and 1024 is 3.6x
    # slower per epoch here for no accuracy gain.
    "ml-1m": {
        "lr": [0.0005, 0.001, 0.002],
        "reg": [0.05, 0.1, 0.15],
        "momentum": [0.9],
        "batch_size": [4096, 8192],
        "n_epochs": [200],
        "patience": [15],
        "eval_every": [5],
    },
}

#: The logistic link needs its own ranges. Its gradient carries the factor
#: g(x)(1-g(x)) <= 1/4, which shrinks the data term without shrinking the prior
#: term, so the useful lambda is roughly an order of magnitude smaller and the
#: useful learning rate an order of magnitude larger. Searching it on the
#: identity ranges makes it look far worse than it is.
LOGISTIC_OVERRIDES = {
    "link": ["logistic"],
    "lr": [0.005, 0.01, 0.02, 0.05, 0.1],
    "reg": [0.0, 0.002, 0.005, 0.01, 0.02],
}

#: Adam adapts its own per-parameter step, so it tolerates (and needs) a much
#: larger nominal learning rate than plain momentum SGD.
ADAM_OVERRIDES = {
    "lr": [0.002, 0.005, 0.01, 0.02],
    "reg": [0.02, 0.05, 0.1, 0.2],
    "batch_size": [1024, 4096],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ml-100k", choices=sorted(SPACE))
    parser.add_argument("--factors", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    space = {**SPACE[args.dataset], "n_factors": [args.factors], "seed": [0]}
    everything = {}

    for variant, overrides, backend in (
        ("pmf", {}, "numpy"),
        ("pmf_bias", {"use_bias": [True]}, "numpy"),
        ("pmf_logistic", LOGISTIC_OVERRIDES, "numpy"),
        ("pmf_adam", {**ADAM_OVERRIDES, "momentum": [None]}, "torch"),
    ):
        print(f"\n{'=' * 78}\n{variant} on {args.dataset}\n{'=' * 78}", flush=True)
        axes = {**space, **overrides}
        if backend == "torch":
            # TorchPMF has no momentum parameter; Adam owns that behaviour.
            axes.pop("momentum", None)
        configurations = grid(**axes)
        results = search(
            configurations,
            dataset=args.dataset,
            n_workers=args.workers,
            save_to=RESULTS_DIR / f"hyperparameter_search_{args.dataset}_{variant}.json",
            model=backend,
            verbose=False,
        )

        header = f"{'lr':>8} {'reg':>6} {'bs':>6} {'ep':>4} {'best':>5} {'train':>7} {'val':>7} {'test':>7}"
        print(header, flush=True)
        for result in results[: args.top]:
            if result.diverged:
                continue
            p = result.params
            print(
                f"{p['lr']:8.4f} {p['reg']:6.3f} {p['batch_size']:6d} {result.epochs_run:4d} "
                f"{str(result.best_epoch):>5} {result.train_rmse:7.4f} {result.val_rmse:7.4f} "
                f"{result.test_rmse:7.4f}",
                flush=True,
            )

        winner = best_of(results)
        everything[variant] = winner.params
        print(f"\n  BEST {variant}: {winner.params}", flush=True)
        print(f"  val {winner.val_rmse:.4f}  test {winner.test_rmse:.4f}", flush=True)

    print(f"\n{'=' * 78}\nPaste into src/pmf/config.py under TUNED[{args.dataset!r}]:\n{'=' * 78}")
    for variant, params in everything.items():
        clean = {k: v for k, v in params.items() if k != "seed"}
        print(f"    {variant!r}: {clean},")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
