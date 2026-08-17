"""Assemble the self-contained HTML summary in ``site/index.html``.

Inlines ``site/src/styles.css`` and ``site/src/app.js`` into
``site/src/index.template.html``, and injects a single ``SITE_DATA`` JSON blob
built from ``results/*.json``. The output has no external requests, so it works
from ``file://``, from GitHub Pages, and as a published artifact alike.

The page's numbers therefore come from the same JSON the paper's tables and
figures do — there is no second copy to fall out of date.

Usage::

    python scripts/build_site.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from pmf.data import load_movielens  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
SITE_SRC = REPO_ROOT / "site" / "src"
SITE_OUT = REPO_ROOT / "site" / "index.html"

#: Results files the page reads. Missing ones degrade to `null` rather than
#: failing the build, so the site can be assembled from a partial run.
WANTED = (
    "dataset_statistics",
    "model_comparison_ml-100k",
    "model_comparison_ml-1m",
    "latent_dimension_ml-100k",
    "regularisation_ml-100k",
    "link_function_ml-100k",
    "optimiser_ml-100k",
    "cold_start_ml-100k",
    "sparsity_curve_ml-100k",
    "ranking_comparison_ml-100k",
    "convergence_ml-100k",
    "scalability",
    "claim_verification_ml-100k",
    "legacy_reproduction",
)


def collect_results() -> dict[str, Any]:
    """Load every results file the page knows how to render."""
    payload: dict[str, Any] = {}
    for name in WANTED:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            payload[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload[name] = None
            print(f"  [warn] missing {path.name}; the page will hide that section")
    return payload


def build_demo_subset(
    n_users: int = 240, n_items: int = 320, seed: int = 0
) -> dict[str, Any]:
    """A dense-ish MovieLens slice small enough to train in a browser tab.

    The most-rated films and most-active users are taken so the slice stays
    learnable: a uniform sample of a 6%-dense matrix would leave many rows
    empty and the live demo would look broken rather than instructive. The page
    says so, since it changes what the demo's RMSE means.
    """
    data = load_movielens("ml-100k", auto_download=True, with_titles=True)

    top_items = np.argsort(-np.bincount(data.items, minlength=data.n_items))[:n_items]
    top_users = np.argsort(-np.bincount(data.users, minlength=data.n_users))[:n_users]

    item_rank = {int(v): k for k, v in enumerate(top_items)}
    user_rank = {int(v): k for k, v in enumerate(top_users)}

    keep = np.fromiter(
        (int(u) in user_rank and int(i) in item_rank for u, i in zip(data.users, data.items, strict=True)),
        dtype=bool,
        count=len(data),
    )
    users = np.fromiter((user_rank[int(u)] for u in data.users[keep]), dtype=np.int32)
    items = np.fromiter((item_rank[int(i)] for i in data.items[keep]), dtype=np.int32)
    ratings = data.ratings[keep].astype(np.int8)

    order = np.random.default_rng(seed).permutation(users.size)
    titles = [
        (data.titles or {}).get(int(raw), f"item {int(raw)}") for raw in top_items
    ]

    density = users.size / (n_users * n_items)
    print(f"  [demo] {users.size:,} ratings, {n_users}x{n_items}, {100 * density:.1f}% dense")

    return {
        "nUsers": int(n_users),
        "nItems": int(n_items),
        "users": users[order].tolist(),
        "items": items[order].tolist(),
        "ratings": ratings[order].tolist(),
        "titles": titles,
        "density": round(density, 4),
    }


def build_matrix_preview(n_rows: int = 60, n_cols: int = 110, seed: int = 0) -> dict[str, Any]:
    """A small observed/unobserved mask for the hero animation.

    Users and items are sampled **uniformly at random**, not by popularity. The
    hero's whole job is to show how empty the matrix is, and a top-N slice is
    the densest corner of the data — it would show a nearly full grid while the
    caption claimed 93.7% missing. A uniform sample reproduces the true density.
    """
    data = load_movielens("ml-100k", auto_download=True, with_titles=False)
    rng = np.random.default_rng(seed)
    rows = rng.choice(data.n_users, size=n_rows, replace=False)
    cols = rng.choice(data.n_items, size=n_cols, replace=False)

    triples = zip(data.users, data.items, data.ratings, strict=True)
    lookup = {(int(u), int(i)): float(r) for u, i, r in triples}
    grid = [[int(lookup.get((int(u), int(i)), 0)) for i in cols] for u in rows]

    observed = sum(1 for row in grid for cell in row if cell)
    fraction = observed / (n_rows * n_cols)
    print(f"  [hero] {n_rows}x{n_cols} uniform sample, {100 * fraction:.1f}% observed")
    return {
        "rows": n_rows,
        "cols": n_cols,
        "grid": grid,
        "observedFraction": round(fraction, 4),
        "observedPct": round(100 * fraction, 1),
        "missingPct": round(100 * (1 - fraction), 1),
    }


def build_headline(results: dict[str, Any]) -> dict[str, Any]:
    """The three numbers the page leads with, derived rather than hand-typed."""
    headline: dict[str, Any] = {
        "best_rmse": None,
        "best_label": "not yet measured",
        "n_claims": 0,
    }

    comparison = results.get("model_comparison_ml-100k")
    if comparison and comparison.get("table"):
        best = min(comparison["table"], key=lambda r: r["test_rmse"])
        headline["best_rmse"] = round(best["test_rmse"], 4)
        headline["best_label"] = best["label"]

    claims = results.get("claim_verification_ml-100k")
    if claims:
        headline["n_claims"] = len(claims.get("claims", []))

    scalability = results.get("scalability")
    if scalability:
        per = scalability.get("per_dataset_fit", {})
        if per:
            lo = min(f["slope"] for f in per.values())
            hi = max(f["slope"] for f in per.values())
            headline["scaling_exponent"] = f"{lo:.3f}–{hi:.3f}"

    return headline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SITE_OUT)
    args = parser.parse_args()

    print("=== building site ===")
    results = collect_results()
    demo = build_demo_subset()
    demo["densityPct"] = round(100 * demo["density"], 1)

    site_data = {
        "results": results,
        "demo": demo,
        "matrix": build_matrix_preview(),
        "headline": build_headline(results),
    }

    template = (SITE_SRC / "index.template.html").read_text(encoding="utf-8")
    styles = (SITE_SRC / "styles.css").read_text(encoding="utf-8")
    script = (SITE_SRC / "app.js").read_text(encoding="utf-8")

    # `</script>` inside JSON would close the host tag early.
    blob = json.dumps(site_data, separators=(",", ":")).replace("</", "<\\/")

    html = (
        template.replace("/*__STYLES__*/", styles)
        .replace("/*__DATA__*/", f"window.SITE_DATA={blob};")
        .replace("/*__APP__*/", script)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"  [built] {args.out.relative_to(REPO_ROOT)} ({len(html) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
