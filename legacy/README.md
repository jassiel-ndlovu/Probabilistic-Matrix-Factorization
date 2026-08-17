# `legacy/` — the original forked implementation

This directory preserves, **verbatim and unmodified**, the implementation from
[`fuhailin/Probabilistic-Matrix-Factorization`](https://github.com/fuhailin/Probabilistic-Matrix-Factorization)
that the 2025 report was originally based on.

It is kept for one reason: **reproducibility of the claims being audited.** The
re-execution study in `report/` compares the rewritten implementation against
what this code actually produces, so the original has to remain runnable and
untouched.

Do not import from this package in new code. It is not part of the `pmf`
package and is excluded from the test suite and from linting.

## Known defects in this implementation

These are documented in the report (§ Re-execution) and are the reason for the
rewrite. They are **not** fixed here.

| # | Location | Defect |
|---|----------|--------|
| 1 | `ProbabilisticMatrixFactorization.py:95-98` | The value appended to `rmse_train` is `sqrt((‖e‖² + ½λ(‖U‖²+‖V‖²)) / N)` — the L2 penalty is folded into the reported "training RMSE". It is not an RMSE and is not comparable to `rmse_test`. |
| 2 | `ProbabilisticMatrixFactorization.py:74-80` | Gradients are accumulated by a Python `for` loop into freshly allocated dense `(num_items, d)` and `(num_users, d)` zero matrices on **every** minibatch, making a step cost O(mn) instead of O(\|batch\|·d). |
| 3 | `ProbabilisticMatrixFactorization.py` (global) | No RNG seeding anywhere, so no run is reproducible. |
| 4 | `ProbabilisticMatrixFactorization.py:113` | Predictions are never clipped to the valid rating range \[1, 5\]. |
| 5 | `ProbabilisticMatrixFactorization.py:55-56` | `num_batches × batch_size` may exceed the training-set size; indices wrap via `np.mod`, so some samples are visited several times per "epoch" and others not at all. |
| 6 | `ProbabilisticMatrixFactorization.py:31-32` | User/item counts are taken as `max(id) + 1` over train **and** test, allocating an unused row 0 and silently leaking the test set's index range into model construction. |
| 7 | `ProbabilisticMatrixFactorization.py:126-145` | `topK` evaluates precision/recall without excluding items the user rated in training, so already-seen items can count as hits. |
| 8 | `LoadData.py:1` | `from numpy import *` shadows builtins (notably `max`, `min`, `sum`, `round`). |
| 9 | `LoadData.py:22-33` | `spilt_rating_dat` draws one `random.random()` per row with no seed, giving a split of non-deterministic size. |

## Running it

From the repository root:

```bash
python legacy/RunExample.py
```

This requires `data/ml-100k/u.data` to be present and will open a matplotlib
window. The audited, non-interactive re-run used by the report is
`scripts/reproduce_legacy.py`, which drives the same class programmatically.
