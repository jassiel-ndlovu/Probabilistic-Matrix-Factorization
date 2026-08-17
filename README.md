<div align="center">

# Probabilistic Matrix Factorization on MovieLens

**A comparative study of probabilistic and Bayesian matrix factorization**

[![tests](https://img.shields.io/badge/tests-51%20passing-0a7a0a)](tests/)
[![gradients](https://img.shields.io/badge/gradients-finite--difference%20checked-005db7)](tests/test_models.py)
[![python](https://img.shields.io/badge/python-3.10%2B-005db7)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache%202.0-565a50)](LICENSE)

Five factorization models and four non-factorization baselines on MovieLens 100K and 1M,
with ablations on capacity, regularization, link function and optimizer, error stratified
by user activity, and a measurement of how training cost scales.

[Interactive summary](site/index.html) ·
[Paper (PDF)](report/main.pdf) ·
[Notebook](notebooks/pmf_walkthrough.ipynb) ·
[Results](results/)

</div>

---

## The model

Each user *i* carries a latent vector **u**ᵢ ∈ ℝᵈ and each item *j* a vector **v**ⱼ ∈ ℝᵈ,
and an observed rating is a noisy inner product of the two:

> *p*(R | U, V, σ²) = ∏ᵢ ∏ⱼ [ 𝒩(Rᵢⱼ | **u**ᵢᵀ**v**ⱼ, σ²) ]^Iᵢⱼ

With zero-mean Gaussian priors on both factor sets, maximising the log posterior is
equivalent to minimising regularized squared error over the observed entries alone:

> *E* = ½ Σ Iᵢⱼ (Rᵢⱼ − **u**ᵢᵀ**v**ⱼ)² + (λ_U/2)‖U‖²_F + (λ_V/2)‖V‖²_F,
> where λ_U = σ²/σ²_U and λ_V = σ²/σ²_V

The regularization strength is a ratio of variances rather than a free parameter — the
sense in which the factorization is *probabilistic*. The Bayesian extension (BPMF) replaces
the fixed prior variances with Gaussian–Wishart hyperpriors and integrates over both
factors and hyperparameters by Gibbs sampling.

---

## Results

<!-- BEGIN:RESULTS -->
**MovieLens 100K, 5 random splits, 95% intervals.**

| Model | Test RMSE | Test MAE | Beats bias baseline? |
|---|---|---|---|
| **BPMF** | 0.9087 ± 0.0050 | 0.7108 | yes |
| PMF + biases | 0.9144 ± 0.0053 | 0.7227 | yes |
| PMF + logistic link | 0.9272 ± 0.0039 | 0.7412 | yes |
| PMF | 0.9311 ± 0.0046 | 0.7375 | yes |
| PMF (Adam) | 0.9358 ± 0.0051 | 0.7423 | yes |
| Bias baseline | 0.9428 ± 0.0046 | 0.7482 | — |
| Item mean | 1.0247 ± 0.0036 | 0.8164 | **no** |
| User mean | 1.0402 ± 0.0065 | 0.8341 | **no** |
| Global mean | 1.1225 ± 0.0044 | 0.9426 | **no** |

Best: **BPMF** at **0.9087**, 0.0341 RMSE ahead of the strongest non-factorization baseline.

**MovieLens 1M, 3 splits** (reduced roster; hyperparameters transferred from the 100K search rather than re-tuned):

| Model | Test RMSE |
|---|---|
| PMF + biases | 0.8644 ± 0.0042 |
| PMF | 0.8722 ± 0.0045 |
| Bias baseline | 0.9084 ± 0.0053 |
| Item mean | 0.9790 ± 0.0064 |
| User mean | 1.0353 ± 0.0042 |

The ordering is preserved at ten times the data.

**Linear scaling holds at fixed model size.** Fitting log(time) against log(N) over N = 7,983 to 800,167, within each dataset: ml-100k **0.976** [0.951, 1.001]; ml-1m **0.994** [0.983, 1.005]. Pooled across datasets the exponent rises to **1.151** [1.068, 1.234] — but that is a constant per-entity offset from the dense momentum update, not superlinear growth in N.
<!-- END:RESULTS -->

Full tables, confidence intervals and every ablation are in [`report/`](report/) and
rendered interactively in [`site/index.html`](site/index.html).

---

## What the study finds

**The Bayesian treatment wins, and where it wins matters.** BPMF is the strongest model
tested, and its advantage over MAP estimation concentrates in the sparse-profile strata
rather than spreading evenly across users — the behaviour the hierarchical priors are meant
to produce.

**Most of the achievable accuracy is additive.** Damped user and item offsets, carrying no
interaction term at all, recover the bulk of it. Every factorization variant clears that
baseline, but by a smaller margin than separates the baseline from a per-item average. The
increment attributable to latent structure is real, and is the quantity a factorization
should be judged on.

**Regularization dominates capacity.** λ is the largest single hyperparameter effect
measured. Latent dimensionality matters far less once early stopping is in place — early
stopping is itself performing the capacity control, which the *d*-sweep with it disabled
shows directly.

**The logistic link is structurally right and practically marginal.** It bounds predictions
to the rating range by construction, but adds little over clipping an unbounded model. Its
useful λ sits an order of magnitude below the identity model's, because the gradient
carries a factor *g*(x)(1−*g*(x)) ≤ ¼ on the data term and not on the prior; searching both
over a single grid makes it look considerably worse than it is.

**Cost is linear in the observations at fixed model size.** Within each dataset the log–log
exponent brackets 1. Pooling across datasets lifts it, but that is a constant per-entity
surcharge from the dense momentum update rather than superlinear growth in *N*.

---

## Quickstart

```bash
git clone https://github.com/jassiel-ndlovu/Probabilistic-Matrix-Factorization
cd Probabilistic-Matrix-Factorization
python -m venv .venv && .venv/Scripts/activate    # Linux/macOS: source .venv/bin/activate
pip install -e ".[all]"
python scripts/download_data.py
```

Train a model in five lines:

```python
from pmf import load_movielens, train_val_test_split, rmse
from pmf.models import PMF

split = train_val_test_split(load_movielens("ml-100k"), seed=0)
model = PMF(n_factors=20, lr=0.001, reg=0.1, use_bias=True).fit(split.train, split.val)
print(rmse(split.test.ratings, model.predict_data(split.test)))
```

Reproduce the whole study:

```bash
python scripts/run_experiments.py
```

---

## Layout

```
src/pmf/
├── data.py           MovieLens acquisition, contiguous re-indexing, deterministic splits
├── metrics.py        RMSE, MAE, precision/recall/NDCG@k, error stratified by user activity
├── baselines.py      Global/user/item means and a damped bias model
├── models/
│   ├── pmf.py        MAP PMF: SGD+momentum, identity or logistic link, optional biases
│   ├── bpmf.py       Bayesian PMF via Gibbs sampling with Gaussian–Wishart hyperpriors
│   └── torch_pmf.py  The same objective under Adam
├── tuning.py         Parallel validation-set grid search
├── experiments/      Comparisons, ablations, diagnostics
└── viz.py            Figures, built from saved results rather than live fits

report/               The paper (LaTeX and compiled PDF) and its bibliography
site/                 Self-contained interactive summary with an in-browser trainer
notebooks/            Guided walkthrough, from the derivation to the recommendations
legacy/               An earlier third-party implementation, kept for comparison
tests/                51 tests, including finite-difference gradient checks
```

---

## Implementation notes

**Two regularization conventions, and they are not interchangeable.** The objective as
published applies one `λ/2‖U‖²_F` penalty regardless of how often a factor is observed.
Reference implementations instead shrink a factor once per observation — the count-weighted
penalty `λ/2 Σᵢ nᵢ‖uᵢ‖²` of Zhou et al. Under the uniform form the useful λ is roughly `nᵢ`
times larger and shifts with dataset density; under the count-weighted form it transfers.
Both are implemented and selected by `reg_mode`; the reported results use the latter.

**Gradients are verified, not assumed.** Every variant's hand-derived gradient is checked
against a central finite difference of the objective it claims to descend, across both link
functions, both regularization modes, and with and without bias terms.

**BPMF averages predictions, not factors.** The predictive estimate is
`(1/S) Σₛ uᵢ⁽ˢ⁾·vⱼ⁽ˢ⁾`. Averaging factors across Gibbs sweeps is meaningless under the
rotational non-identifiability of `UVᵀ`, and a test asserts the two differ.

**Everything carries an interval.** The headline comparison uses five random splits and the
ablations three, with Student-*t* 95% intervals. A single split on 100k ratings carries
enough sampling noise to reorder models differing by less than about 0.006 RMSE.

**Numbers are generated, never transcribed.** `scripts/make_report_tables.py` emits the
paper's macros and tables from `results/*.json`, `scripts/update_readme.py` fills the
results block above, and `scripts/build_site.py` embeds the same JSON in the site. A stale
number is not expressible.

---

## Provenance

This study began as a re-execution of an earlier report by the same author, which quoted a
single MovieLens 100K test RMSE of 1.0109 with no baseline, confidence interval or random
seed. That figure is reachable only by substantially under-training and sits behind the
additive baseline reported here, so the numbers in this repository supersede it. The
paper's appendix records the comparison together with corrections to several points of
attribution and notation in that text. The third-party implementation the earlier work
adapted is preserved unmodified in [`legacy/`](legacy/) with its defects catalogued.

---

## Data

[MovieLens](https://grouplens.org/datasets/movielens/) 100K and 1M, downloaded on first use.

> F. M. Harper and J. A. Konstan. The MovieLens Datasets: History and Context.
> *ACM Transactions on Interactive Intelligent Systems* 5(4), 2015.

## License

Apache 2.0 — see [LICENSE](LICENSE). The `legacy/` directory retains the original project's
licensing.
