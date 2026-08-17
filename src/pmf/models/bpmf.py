r"""Bayesian Probabilistic Matrix Factorization (Salakhutdinov & Mnih, ICML 2008).

Where :mod:`pmf.models.pmf` fixes the prior variances by hand and returns a
point estimate, BPMF places Gaussian–Wishart hyperpriors on the factor priors
and integrates over everything with Gibbs sampling:

.. math::
    u_i \sim \mathcal{N}(\mu_U, \Lambda_U^{-1}), \qquad
    v_j \sim \mathcal{N}(\mu_V, \Lambda_V^{-1}),

.. math::
    p(\mu_U, \Lambda_U) = \mathcal{N}\!\big(\mu_U \mid \mu_0, (\beta_0\Lambda_U)^{-1}\big)\,
                          \mathcal{W}(\Lambda_U \mid W_0, \nu_0).

Every conditional is conjugate, so a sweep is exact sampling rather than a
Metropolis step. This is what removes the manual :math:`\lambda` tuning that the
original report correctly identified as PMF's main practical weakness — the
regularisation strength is now inferred from the data, per sweep.

One implementation note that matters for correctness: the predictive
distribution is approximated by averaging *predictions* over retained samples,

.. math::
    \hat r_{ij} \approx \frac{1}{S}\sum_{s} \big(u_i^{(s)}\big)^\top v_j^{(s)},

not by averaging the factors and multiplying. The two differ because the inner
product is not linear in the pair, and averaging factors across sweeps is
additionally meaningless under the rotational non-identifiability of :math:`UV^\top`.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import linalg

from ..data import RatingData
from ..metrics import rmse
from .base import History, Recommender

__all__ = ["BPMF"]


class BPMF(Recommender):
    """Gibbs-sampled Bayesian PMF.

    Parameters
    ----------
    n_factors:
        Latent dimensionality :math:`d`.
    alpha:
        Observation noise precision :math:`1/\\sigma^2`, held fixed at 2 as in
        the paper.
    beta_0, nu_0, mu_0, W_0:
        Gaussian–Wishart hyperprior parameters. Defaults follow the paper:
        :math:`\\beta_0 = 2`, :math:`\\nu_0 = d`, :math:`\\mu_0 = 0`,
        :math:`W_0 = I`.
    n_iter:
        Total Gibbs sweeps.
    burn_in:
        Sweeps discarded before predictions start being accumulated.
    thin:
        Retain every ``thin``-th post-burn-in sample.
    max_samples:
        Hard cap on retained samples, to bound memory on the larger datasets.
    init_from:
        Optional fitted :class:`~pmf.models.pmf.PMF` whose MAP factors seed the
        chain. The paper initialises this way; it shortens burn-in considerably.
    """

    def __init__(
        self,
        n_factors: int = 20,
        alpha: float = 2.0,
        beta_0: float = 2.0,
        nu_0: float | None = None,
        n_iter: int = 100,
        burn_in: int = 20,
        thin: int = 1,
        max_samples: int = 60,
        init_scale: float = 0.05,
        init_from=None,
        clip: bool = True,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.n_factors = int(n_factors)
        self.alpha = float(alpha)
        self.beta_0 = float(beta_0)
        self.nu_0 = float(nu_0) if nu_0 is not None else float(n_factors)
        self.n_iter = int(n_iter)
        self.burn_in = int(burn_in)
        self.thin = max(1, int(thin))
        self.max_samples = int(max_samples)
        self.init_scale = float(init_scale)
        self.init_from = init_from
        self.clip = bool(clip)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.samples_U: list[np.ndarray] = []
        self.samples_V: list[np.ndarray] = []
        self.history = History()
        self.global_mean = 0.0

    @property
    def name(self) -> str:
        return f"BPMF (d={self.n_factors})"

    # ------------------------------------------------------------------ fit

    def fit(self, train: RatingData, val: RatingData | None = None) -> BPMF:
        rng = np.random.default_rng(self.seed)
        d = self.n_factors

        self.rating_min, self.rating_max = train.rating_min, train.rating_max
        self.global_mean = float(train.ratings.mean())
        centred = train.ratings - self.global_mean

        n_users, n_items = train.n_users, train.n_items
        by_user = _group(train.users, train.items, centred, n_users)
        by_item = _group(train.items, train.users, centred, n_items)

        if self.init_from is not None and getattr(self.init_from, "U", None) is not None:
            U = self.init_from.U.copy()
            V = self.init_from.V.copy()
        else:
            U = rng.normal(0.0, self.init_scale, size=(n_users, d))
            V = rng.normal(0.0, self.init_scale, size=(n_items, d))

        mu_0 = np.zeros(d)
        W_0_inv = np.eye(d)

        started = time.perf_counter()
        n_kept = 0

        for sweep in range(1, self.n_iter + 1):
            # 1. Hyperparameters, conjugate Gaussian-Wishart draws.
            mu_U, Lambda_U = _sample_hyperparams(U, mu_0, self.beta_0, self.nu_0, W_0_inv, rng)
            mu_V, Lambda_V = _sample_hyperparams(V, mu_0, self.beta_0, self.nu_0, W_0_inv, rng)

            # 2./3. Factors, conditionally independent given the other side.
            U = _sample_factors(by_user, V, mu_U, Lambda_U, self.alpha, rng)
            V = _sample_factors(by_item, U, mu_V, Lambda_V, self.alpha, rng)

            keep = sweep > self.burn_in and (sweep - self.burn_in) % self.thin == 0
            if keep and len(self.samples_U) < self.max_samples:
                self.samples_U.append(U.copy())
                self.samples_V.append(V.copy())
                n_kept = len(self.samples_U)

            train_rmse = rmse(train.ratings, self.predict_data(train)) if n_kept else float("nan")
            val_rmse = (
                rmse(val.ratings, self.predict_data(val)) if (val is not None and n_kept) else None
            )
            self.history.record(
                sweep, float("nan"), train_rmse, val_rmse, time.perf_counter() - started
            )

            if self.verbose and (sweep % 10 == 0 or sweep == self.n_iter):
                tail = f", val RMSE {val_rmse:.4f}" if val_rmse is not None else ""
                print(f"  sweep {sweep:3d}  samples {n_kept:3d}  train RMSE {train_rmse:.4f}{tail}")

        if not self.samples_U:
            raise RuntimeError("no samples retained; burn_in must be smaller than n_iter")
        return self

    # -------------------------------------------------------------- predict

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        if not self.samples_U:
            raise RuntimeError("call fit() before predict()")

        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        known = (users < self.samples_U[0].shape[0]) & (items < self.samples_V[0].shape[0])

        out = np.full(users.shape[0], self.global_mean, dtype=np.float64)
        if not known.any():
            return out

        u, i = users[known], items[known]
        total = np.zeros(u.shape[0], dtype=np.float64)
        for U, V in zip(self.samples_U, self.samples_V, strict=True):
            total += np.einsum("ij,ij->i", U[u], V[i])

        out[known] = self.global_mean + total / len(self.samples_U)
        return self._clip(out) if self.clip else out


# ---------------------------------------------------------------- internals


def _group(
    index: np.ndarray, partner: np.ndarray, values: np.ndarray, n: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bucket observations by entity into a CSR-like adjacency list.

    Entry ``k`` holds the partner-entity indices rated by entity ``k`` and the
    matching (centred) ratings. Sorting once and slicing beats ``n`` boolean
    masks over the full array, which is what keeps a sweep linear in the number
    of observations rather than quadratic in the number of entities.
    """
    order = np.argsort(index, kind="stable")
    sorted_partner = partner[order]
    sorted_values = values[order]
    boundaries = np.searchsorted(index[order], np.arange(n + 1))
    return [
        (
            sorted_partner[boundaries[k] : boundaries[k + 1]],
            sorted_values[boundaries[k] : boundaries[k + 1]],
        )
        for k in range(n)
    ]


def _sample_hyperparams(
    factors: np.ndarray,
    mu_0: np.ndarray,
    beta_0: float,
    nu_0: float,
    W_0_inv: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw :math:`(\\mu, \\Lambda)` from the Gaussian-Wishart conditional."""
    n, d = factors.shape
    mean = factors.mean(axis=0)
    deviation = factors - mean
    scatter = deviation.T @ deviation

    beta_star = beta_0 + n
    nu_star = nu_0 + n
    mu_star = (beta_0 * mu_0 + n * mean) / beta_star

    gap = (mu_0 - mean)[:, None]
    W_star_inv = W_0_inv + scatter + (beta_0 * n / beta_star) * (gap @ gap.T)
    W_star = linalg.inv(0.5 * (W_star_inv + W_star_inv.T))
    W_star = 0.5 * (W_star + W_star.T)

    Lambda = _sample_wishart(W_star, nu_star, rng)
    mu = _sample_gaussian_from_precision(mu_star, beta_star * Lambda, rng, d)
    return mu, Lambda


def _sample_wishart(scale: np.ndarray, df: float, rng: np.random.Generator) -> np.ndarray:
    """Wishart draw via the Bartlett decomposition.

    Avoids the per-call overhead and repeated factorisation of
    ``scipy.stats.wishart.rvs`` — this runs twice per sweep for hundreds of sweeps.
    """
    d = scale.shape[0]
    chol = linalg.cholesky(scale, lower=True)

    A = np.zeros((d, d))
    A[np.diag_indices(d)] = np.sqrt(rng.chisquare(df - np.arange(d)))
    rows, cols = np.tril_indices(d, -1)
    A[rows, cols] = rng.standard_normal(rows.size)

    factor = chol @ A
    return factor @ factor.T


def _sample_gaussian_from_precision(
    mean: np.ndarray, precision: np.ndarray, rng: np.random.Generator, d: int
) -> np.ndarray:
    """Draw :math:`x \\sim \\mathcal{N}(\\mu, \\Lambda^{-1})` without inverting :math:`\\Lambda`.

    With :math:`\\Lambda = LL^\\top`, solving :math:`L^\\top x = z` for
    :math:`z \\sim \\mathcal{N}(0, I)` gives covariance
    :math:`L^{-\\top}L^{-1} = \\Lambda^{-1}` as required.
    """
    chol = linalg.cholesky(precision, lower=True)
    z = rng.standard_normal(d)
    return mean + linalg.solve_triangular(chol.T, z, lower=False)


def _sample_factors(
    grouped: list[tuple[np.ndarray, np.ndarray]],
    other: np.ndarray,
    mu: np.ndarray,
    Lambda: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one full side of the factorisation given the other side.

    For entity :math:`i` with observed partners :math:`J_i`,
    :math:`\\Lambda_i^* = \\Lambda + \\alpha\\sum_{j\\in J_i} v_j v_j^\\top` and
    :math:`\\mu_i^* = (\\Lambda_i^*)^{-1}(\\alpha\\sum_j v_j r_{ij} + \\Lambda\\mu)`.
    Entities with no observations simply fall back to the prior.
    """
    n, d = len(grouped), other.shape[1]
    out = np.empty((n, d), dtype=np.float64)
    Lambda_mu = Lambda @ mu

    for k, (partners, ratings) in enumerate(grouped):
        if partners.size == 0:
            out[k] = _sample_gaussian_from_precision(mu, Lambda, rng, d)
            continue
        block = other[partners]
        precision = Lambda + alpha * (block.T @ block)
        rhs = alpha * (block.T @ ratings) + Lambda_mu
        mean = linalg.solve(precision, rhs, assume_a="pos")
        out[k] = _sample_gaussian_from_precision(mean, precision, rng, d)

    return out
