"""Figure generation.

Every figure is built from a saved ``results/*.json`` file rather than from a
live fit, so the figures in the paper and the numbers in the tables cannot drift
apart: regenerating one without re-running the other is impossible.

Sizing targets the IEEE two-column template — :data:`COLUMN_WIDTH` for a
single-column figure, :data:`PAGE_WIDTH` for a full-width one — so figures are
placed at their natural size and text in them matches the body text rather than
being scaled to something unreadable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

__all__ = ["PALETTE", "apply_style", "make_all", "FIGURE_DIR"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
FIGURE_DIR = REPO_ROOT / "report" / "figures"

#: IEEEtran column and text widths, in inches.
COLUMN_WIDTH = 3.45
PAGE_WIDTH = 7.16

#: Categorical slots, in fixed assignment order. Validated for adjacent-pair
#: CVD separation and normal-vision separation on a light surface; charts that
#: put every pair side by side (scatter) use only the first three.
PALETTE: dict[str, str] = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
SERIES = list(PALETTE.values())

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
CRITICAL = "#d03b3b"


def apply_style() -> None:
    """Recessive chrome, thin marks, no top/right spines."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 4,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _load(name: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{suffix}")
    plt.close(fig)
    print(f"  [figure] {name}.pdf / .png")


# --------------------------------------------------------------------- plots


def fig_learning_curves(out_dir: Path, dataset: str = "ml-100k") -> None:
    """Objective and unpenalised RMSE, as three separate series.

    The left panel is the point: the MAP objective falls monotonically while
    validation RMSE turns upward. A single blended "training RMSE" — what the
    original implementation reported — cannot show that divergence at all.
    """
    data = _load(f"convergence_{dataset}")
    if data is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.5))
    traces = data["traces"]

    # The left panel shows the regime that was actually selected; fall back to
    # the first trace if the naming ever changes.
    reference = next((k for k in traces if "selected" in k), next(iter(traces)))

    ax = axes[0]
    history = traces[reference]
    epochs = history["epoch"]
    ax.plot(epochs, history["train_rmse"], color=PALETTE["blue"], label="Train RMSE")
    ax.plot(epochs, history["val_rmse"], color=PALETTE["orange"], label="Validation RMSE")

    best = int(np.nanargmin(history["val_rmse"]))
    ax.axvline(epochs[best], color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.annotate(
        f"best epoch {epochs[best]}",
        xy=(epochs[best], history["val_rmse"][best]),
        xytext=(6, 14),
        textcoords="offset points",
        fontsize=6.5,
        color=INK_SECONDARY,
    )

    twin = ax.twiny()
    twin.set_xlim(ax.get_xlim())
    twin.set_xticks([])
    twin.spines["top"].set_visible(False)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Train and validation error diverge", loc="left")
    ax.legend(loc="upper right")

    ax = axes[1]
    for colour, (label, history) in zip(SERIES, traces.items(), strict=False):
        ax.plot(history["epoch"], history["val_rmse"], color=colour, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Regularisation controls the turn", loc="left")
    ax.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, "learning_curves", out_dir)


def fig_latent_dimension(out_dir: Path, dataset: str = "ml-100k") -> None:
    """Test RMSE against d, with and without early stopping."""
    data = _load(f"latent_dimension_{dataset}")
    if data is None:
        return

    rows = data["table"]
    dims = [r["n_factors"] for r in rows]
    stopped = np.array([r["test_rmse"] for r in rows])
    errors = np.array([r["test_rmse_ci95"] for r in rows])
    unstopped = np.array([r["test_rmse_no_early_stop"] for r in rows])
    train = np.array([r["train_rmse"] for r in rows])

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.35))
    ax.fill_between(dims, stopped - errors, stopped + errors, color=PALETTE["blue"], alpha=0.15, linewidth=0)
    ax.plot(dims, stopped, color=PALETTE["blue"], marker="o", label="Test (early stopping)")
    ax.plot(dims, unstopped, color=PALETTE["orange"], marker="s", linestyle=(0, (5, 2)),
            label="Test (no early stopping)")
    ax.plot(dims, train, color=PALETTE["aqua"], marker="^", linewidth=1.3, label="Train (early stopping)")

    best = int(np.argmin(stopped))
    ax.scatter([dims[best]], [stopped[best]], s=54, facecolor=SURFACE,
               edgecolor=PALETTE["blue"], zorder=5, linewidth=1.6)
    ax.annotate(
        f"d={dims[best]}\n{stopped[best]:.4f}",
        xy=(dims[best], stopped[best]),
        xytext=(4, -20),
        textcoords="offset points",
        fontsize=6.5,
        color=INK_SECONDARY,
    )

    ax.set_xscale("log")
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("Latent dimensionality $d$")
    ax.set_ylabel("RMSE")
    ax.set_title("Capacity without early stopping overfits", loc="left")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, "latent_dimension", out_dir)


def fig_regularisation(out_dir: Path, dataset: str = "ml-100k") -> None:
    """The overfitting gap as a function of lambda."""
    data = _load(f"regularisation_{dataset}")
    if data is None:
        return

    rows = data["table"]
    regs = np.array([r["reg"] for r in rows])
    train = np.array([r["train_rmse"] for r in rows])
    test = np.array([r["test_rmse"] for r in rows])
    errors = np.array([r["test_rmse_ci95"] for r in rows])

    # A log axis cannot show lambda = 0; place it at a labelled sentinel tick.
    positive = regs[regs > 0]
    sentinel = positive.min() / 3 if positive.size else 1e-3
    x = np.where(regs > 0, regs, sentinel)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.35))
    ax.fill_between(x, test - errors, test + errors, color=PALETTE["orange"], alpha=0.15, linewidth=0)
    ax.plot(x, train, color=PALETTE["blue"], marker="o", label="Train RMSE")
    ax.plot(x, test, color=PALETTE["orange"], marker="s", label="Test RMSE")

    best = int(np.argmin(test))
    ax.axvline(x[best], color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.annotate(
        f"$\\lambda$={regs[best]:g}",
        xy=(x[best], test[best]),
        xytext=(5, 10),
        textcoords="offset points",
        fontsize=6.5,
        color=INK_SECONDARY,
    )

    ax.set_xscale("log")
    ticks = [sentinel, *positive.tolist()]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", *[f"{v:g}" for v in positive]])
    ax.set_xlabel("Regularisation strength $\\lambda$")
    ax.set_ylabel("RMSE")
    ax.set_title("Without $\\lambda$, the model memorises", loc="left")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, "regularisation", out_dir)


def fig_model_comparison(out_dir: Path, dataset: str = "ml-100k") -> None:
    """Ranked test RMSE with confidence intervals, baselines included."""
    data = _load(f"model_comparison_{dataset}")
    if data is None:
        return

    rows = sorted(data["table"], key=lambda r: -r["test_rmse"])
    labels = [r["label"] for r in rows]
    values = np.array([r["test_rmse"] for r in rows])
    errors = np.array([r["test_rmse_ci95"] for r in rows])

    # Baselines recede; factorisation models carry the categorical hues.
    is_baseline = [
        any(token in label for token in ("mean", "Bias baseline")) for label in labels
    ]
    colours = [INK_MUTED if flag else PALETTE["blue"] for flag in is_baseline]
    best = int(np.argmin(values))
    colours[best] = PALETTE["aqua"]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.32 * len(rows) + 0.85))
    positions = np.arange(len(rows))
    ax.barh(positions, values, height=0.62, color=colours, linewidth=0)
    ax.errorbar(values, positions, xerr=errors, fmt="none", ecolor=INK_SECONDARY, elinewidth=0.9, capsize=2)

    for position, value in zip(positions, values, strict=True):
        ax.text(value + 0.012, position, f"{value:.4f}", va="center", fontsize=6.5, color=INK_SECONDARY)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7, color=INK)
    ax.set_xlim(0.85, max(values) + 0.09)
    ax.set_xlabel("Test RMSE (lower is better)")
    ax.set_title(f"Model comparison, {dataset}", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, f"model_comparison_{dataset}", out_dir)


#: Minimum test observations for a stratum's confidence interval to carry
#: information. MovieLens guarantees every user at least 20 ratings, so the
#: lowest activity bin is populated only when a split happens to place nearly
#: all of one user's ratings in test: it holds 3 observations, and its interval
#: (+/- 7 RMSE) is wider than the rating scale. Plotting that band on a shared
#: axis compresses every real difference into a few pixels.
MIN_RELIABLE_BIN = 50


def fig_cold_start(out_dir: Path, dataset: str = "ml-100k") -> None:
    """RMSE stratified by user activity, in two panels.

    The left panel keeps every stratum so nothing is hidden, with intervals
    clipped to the axes and under-powered strata drawn hollow. The right panel
    drops the under-powered strata, which is the only version in which the
    differences between models are legible.
    """
    data = _load(f"cold_start_{dataset}")
    if data is None:
        return

    rows = data["table"]
    models = [k for k in rows[0] if k not in ("bin", "n_test_ratings") and not k.endswith("ci95")]
    reliable = [r["n_test_ratings"] >= MIN_RELIABLE_BIN for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.6))

    for panel, ax in enumerate(axes):
        subset = rows if panel == 0 else [r for r, ok in zip(rows, reliable, strict=True) if ok]
        keep = reliable if panel == 0 else [True] * len(subset)
        positions = np.arange(len(subset))

        for colour, model in zip(SERIES, models, strict=False):
            values = np.array([r[model] for r in subset], dtype=float)
            errors = np.array([r.get(f"{model} ci95") or 0.0 for r in subset], dtype=float)

            ax.plot(positions, values, color=colour, label=model, zorder=3)
            # Bands only where they mean something.
            trusted = np.array(keep, dtype=bool)
            ax.fill_between(
                positions, np.where(trusted, values - errors, values),
                np.where(trusted, values + errors, values),
                color=colour, alpha=0.13, linewidth=0,
            )
            ax.scatter(
                positions[trusted], values[trusted], s=22, color=colour, zorder=4,
                edgecolor=SURFACE, linewidth=0.8,
            )
            if (~trusted).any():
                ax.scatter(
                    positions[~trusted], values[~trusted], s=26, facecolor=SURFACE,
                    edgecolor=colour, linewidth=1.2, zorder=4,
                )

        # Scale to the trustworthy strata so one tiny bin cannot flatten the rest.
        scaled = np.concatenate([
            [r[m] for r, ok in zip(subset, keep, strict=True) if ok] for m in models
        ])
        margin = 0.06 * (scaled.max() - scaled.min())
        ax.set_ylim(scaled.min() - margin, scaled.max() + margin)

        ax.set_xticks(positions)
        ax.set_xticklabels([r["bin"] for r in subset], fontsize=6.2, rotation=20, ha="right")
        ax.set_xlabel("Training ratings by the user")
        if panel == 0:
            ax.set_ylabel("Test RMSE")
            ax.set_title("All strata", loc="left")
            thin = [r for r, ok in zip(rows, reliable, strict=True) if not ok]
            if thin:
                ax.annotate(
                    f"$n$={thin[0]['n_test_ratings']}, off scale",
                    xy=(0, ax.get_ylim()[1]), xytext=(3, -8), textcoords="offset points",
                    fontsize=6, color=INK_MUTED, va="top",
                )
        else:
            ax.set_title(f"Strata with $n \\geq {MIN_RELIABLE_BIN}$", loc="left")
            ax.legend(loc="best", ncol=2, fontsize=6.2)

    fig.tight_layout()
    _save(fig, "cold_start", out_dir)


def fig_scalability(out_dir: Path) -> None:
    """Cost per epoch against N, with the fitted log-log slope."""
    data = _load("scalability")
    if data is None:
        return

    rows = data["table"]
    fit = data["loglog_fit"]
    n = np.array([r["n_train_ratings"] for r in rows], dtype=float)
    t = np.array([r["seconds_per_epoch"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.35))
    for colour, dataset in zip(SERIES, sorted({r["dataset"] for r in rows}), strict=False):
        mask = np.array([r["dataset"] == dataset for r in rows])
        ax.scatter(n[mask], t[mask], s=26, color=colour, label=dataset, zorder=3,
                   edgecolor=SURFACE, linewidth=0.8)

    grid_n = np.logspace(np.log10(n.min()), np.log10(n.max()), 50)
    ax.plot(
        grid_n,
        np.exp(fit["intercept"]) * grid_n ** fit["slope"],
        color=INK_MUTED, linewidth=1.2, linestyle=(0, (5, 2)), zorder=2,
        label=f"fit: slope {fit['slope']:.3f}",
    )
    ax.plot(grid_n, t.min() * grid_n / n.min(), color=AXIS, linewidth=1.0, linestyle=":", zorder=1,
            label="exact linear")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training ratings $N$")
    ax.set_ylabel("Seconds per epoch")

    # The within-dataset exponents are the ones that answer the claim; the
    # pooled figure additionally varies the entity counts, so it is subordinate.
    within = "  ".join(
        f"{name} {f['slope']:.3f}" for name, f in data.get("per_dataset_fit", {}).items()
    )
    title = f"Within dataset: {within}" if within else ""
    ax.set_title(title, loc="left", fontsize=7.5)
    ax.text(
        0.0, 1.005, f"pooled {fit['slope']:.3f} [{fit['slope_ci_low']:.3f}, "
        f"{fit['slope_ci_high']:.3f}], $R^2$={fit['r_squared']:.3f}",
        transform=ax.transAxes, fontsize=6.4, color=INK_MUTED, va="bottom",
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "scalability", out_dir)


def fig_sparsity(out_dir: Path, dataset: str = "ml-100k") -> None:
    """PMF's advantage over additive effects as data is thinned."""
    data = _load(f"sparsity_curve_{dataset}")
    if data is None:
        return

    rows = data["table"]
    n = np.array([r["n_train_ratings"] for r in rows], dtype=float)
    pmf_values = np.array([r["pmf_test_rmse"] for r in rows])
    pmf_errors = np.array([r["pmf_test_rmse_ci95"] for r in rows])
    bias_values = np.array([r["bias_test_rmse"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.3))

    ax = axes[0]
    ax.fill_between(n, pmf_values - pmf_errors, pmf_values + pmf_errors,
                    color=PALETTE["blue"], alpha=0.15, linewidth=0)
    ax.plot(n, pmf_values, color=PALETTE["blue"], marker="o", label="PMF")
    ax.plot(n, bias_values, color=PALETTE["orange"], marker="s", label="Bias baseline")
    ax.set_xlabel("Training ratings")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Both degrade as data thins", loc="left")
    ax.legend(loc="best")

    ax = axes[1]
    advantage = bias_values - pmf_values
    colours = [PALETTE["aqua"] if v > 0 else CRITICAL for v in advantage]
    ax.bar(np.arange(len(n)), advantage, width=0.6, color=colours, linewidth=0)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xticks(np.arange(len(n)))
    ax.set_xticklabels([f"{int(v):,}" for v in n], fontsize=6.2, rotation=20, ha="right")
    ax.set_xlabel("Training ratings")
    ax.set_ylabel("RMSE advantage of PMF")
    ax.set_title("The factorisation's margin shrinks", loc="left")

    fig.tight_layout()
    _save(fig, "sparsity_curve", out_dir)


def fig_claim_verification(out_dir: Path, dataset: str = "ml-100k") -> None:
    """Where the originally reported 1.0109 sits against what is reachable."""
    data = _load(f"claim_verification_{dataset}")
    if data is None:
        return

    claim = next((c for c in data["claims"] if c["claim_id"] == "C1"), None)
    if claim is None or "candidates" not in claim:
        return

    values = np.array([c["test_rmse"] for c in claim["candidates"]])
    claimed = claim["claimed_value"]
    reference = claim["reference_baselines"]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.2))
    ax.hist(values, bins=22, color=PALETTE["blue"], alpha=0.75, linewidth=0)

    for value, label, colour in (
        (claimed, f"reported {claimed}", CRITICAL),
        (reference["bias_baseline"], f"bias baseline {reference['bias_baseline']:.3f}", INK_MUTED),
        (float(values.min()), f"best reached {values.min():.3f}", PALETTE["aqua"]),
    ):
        ax.axvline(value, color=colour, linewidth=1.5, linestyle=(0, (4, 2)))
        ax.annotate(
            label, xy=(value, ax.get_ylim()[1] * 0.92), xytext=(3, 0),
            textcoords="offset points", fontsize=6.2, color=colour, rotation=90, va="top",
        )

    ax.set_xlabel("Test RMSE")
    ax.set_ylabel("Configurations")
    ax.set_title("The reported figure against what is reachable", loc="left")
    fig.tight_layout()
    _save(fig, "claim_verification", out_dir)


def fig_dataset_profile(out_dir: Path) -> None:
    """The long tail that makes this a sparse-data problem."""
    from .data import load_movielens

    fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 2.1))

    data = load_movielens("ml-100k", auto_download=False, with_titles=False)
    per_user = np.bincount(data.users)
    per_item = np.bincount(data.items)

    ax = axes[0]
    ax.hist(data.ratings, bins=np.arange(0.5, 6.5, 1.0), color=PALETTE["blue"], linewidth=0, rwidth=0.82)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Ratings skew positive", loc="left")
    thousands = FuncFormatter(lambda v, _: f"{int(v / 1000)}k" if v >= 1000 else f"{int(v)}")
    ax.yaxis.set_major_formatter(thousands)

    for ax, counts, label, colour in (
        (axes[1], np.sort(per_user)[::-1], "Users", PALETTE["orange"]),
        (axes[2], np.sort(per_item)[::-1], "Items", PALETTE["aqua"]),
    ):
        ax.plot(np.arange(1, counts.size + 1), counts, color=colour)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{label}, ranked")
        ax.set_ylabel("Ratings")
        ax.set_title(f"{label}: a long tail", loc="left")
        ax.axhline(np.median(counts), color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.annotate(
            f"median {int(np.median(counts))}", xy=(1.4, np.median(counts)), xytext=(0, 4),
            textcoords="offset points", fontsize=6.2, color=INK_SECONDARY,
        )

    fig.tight_layout()
    _save(fig, "dataset_profile", out_dir)


def fig_ranking(out_dir: Path, dataset: str = "ml-100k") -> None:
    """Top-k ranking quality, which RMSE does not predict."""
    data = _load(f"ranking_comparison_{dataset}")
    if data is None:
        return

    rows = data["table"]
    k = data["k"]
    labels = [r["label"] for r in rows]
    metrics = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}"]

    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 2.4))
    positions = np.arange(len(labels))
    width = 0.26

    for offset, (colour, metric) in enumerate(zip(SERIES, metrics, strict=False)):
        values = [r[metric] for r in rows]
        errors = [r.get(f"{metric}_ci95", 0.0) for r in rows]
        ax.bar(positions + (offset - 1) * width, values, width=width - 0.02, color=colour,
               label=metric, linewidth=0)
        ax.errorbar(positions + (offset - 1) * width, values, yerr=errors, fmt="none",
                    ecolor=INK_SECONDARY, elinewidth=0.8, capsize=1.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6.4, rotation=18, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Top-{k} ranking quality", loc="left")
    ax.legend(loc="upper left", ncol=3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "ranking", out_dir)


def make_all(out_dir: Path | str = FIGURE_DIR, dataset: str = "ml-100k") -> None:
    """Build every figure whose results file exists.

    Switches to the Agg backend here rather than at import time. Setting the
    backend from module scope would be a library hijacking the caller's
    environment: importing :func:`apply_style` in a notebook — which the
    walkthrough does — silently disabled inline figure rendering, so every plot
    cell executed successfully and displayed nothing.
    """
    mpl.use("Agg")
    apply_style()
    out_dir = Path(out_dir)
    print(f"\n=== figures -> {out_dir} ===")

    fig_dataset_profile(out_dir)
    fig_learning_curves(out_dir, dataset)
    fig_latent_dimension(out_dir, dataset)
    fig_regularisation(out_dir, dataset)
    fig_model_comparison(out_dir, dataset)
    fig_model_comparison(out_dir, "ml-1m")
    fig_cold_start(out_dir, dataset)
    fig_scalability(out_dir)
    fig_sparsity(out_dir, dataset)
    fig_claim_verification(out_dir, dataset)
    fig_ranking(out_dir, dataset)
