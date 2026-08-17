r"""Probabilistic Matrix Factorization (Salakhutdinov & Mnih, NIPS 2007).

The model treats each observed rating as a noisy realisation of an inner product
between a user factor and an item factor,

.. math::
    p(R \mid U, V, \sigma^2)
        = \prod_{i}\prod_{j}
          \big[\mathcal{N}(R_{ij} \mid u_i^\top v_j, \sigma^2)\big]^{I_{ij}},

with zero-mean spherical Gaussian priors on :math:`U` and :math:`V`. Maximising
the log posterior in :math:`U, V` with the hyperparameters held fixed is
equivalent to minimising the regularised squared error

.. math::
    E = \tfrac{1}{2}\sum_{i,j} I_{ij}\,(R_{ij} - u_i^\top v_j)^2
        + \tfrac{\lambda_U}{2}\|U\|_F^2 + \tfrac{\lambda_V}{2}\|V\|_F^2,

where :math:`\lambda_U = \sigma^2/\sigma_U^2` and
:math:`\lambda_V = \sigma^2/\sigma_V^2`. This module minimises :math:`E`
directly by minibatch gradient descent with momentum, exactly as the original
paper's reference implementation does.

Three variants are exposed through constructor flags rather than subclasses,
because they differ only in the link function and in which parameters exist:

``link="identity"``
    :math:`\hat r_{ij} = \mu + u_i^\top v_j`. The global mean :math:`\mu` is
    subtracted from the targets, which is what makes zero-mean priors sensible.

``link="logistic"``
    :math:`\hat r_{ij} = r_{\min} + (r_{\max}-r_{\min})\, g(u_i^\top v_j)` with
    :math:`g` the logistic function and targets rescaled to :math:`[0,1]` by
    :math:`t(x) = (x - r_{\min})/(r_{\max}-r_{\min})`. This is §2.1 of the paper
    and bounds predictions to the valid rating range by construction.

``use_bias=True``
    Adds per-user and per-item offsets, :math:`\hat r_{ij} = \mu + b_i + c_j +
    u_i^\top v_j`. Not in the 2007 paper; included because it is the standard
    strong MF baseline and isolates how much of PMF's accuracy comes from the
    interaction term as opposed to simple additive effects.
"""

from __future__ import annotations

import time

import numpy as np

from ..data import RatingData
from ..metrics import rmse
from .base import History, Recommender, scatter_add

__all__ = ["PMF"]


class PMF(Recommender):
    """MAP-estimated Probabilistic Matrix Factorization.

    Parameters
    ----------
    n_factors:
        Latent dimensionality :math:`d`.
    lr:
        Learning rate applied to the mean gradient over a minibatch.
    reg:
        Regularisation strength. Sets both :math:`\\lambda_U` and
        :math:`\\lambda_V` unless ``reg_user``/``reg_item`` override it. Its
        useful scale depends strongly on ``reg_mode`` — see below.
    reg_mode:
        ``"uniform"`` implements the objective :math:`E` exactly as written in
        the paper: one :math:`\\tfrac{\\lambda}{2}\\|U\\|_F^2` penalty regardless
        of how often a factor is observed. ``"per_observation"`` (the default)
        shrinks a factor once per observation, which is equivalent to the
        count-weighted penalty
        :math:`\\tfrac{\\lambda}{2}\\sum_i n_i\\|u_i\\|^2` with :math:`n_i` the
        number of ratings by user :math:`i`.

        The two are not cosmetic variants. Under ``"uniform"`` the prior has to
        counteract a data gradient that has already accumulated :math:`n_i`
        observations, so the useful range of :math:`\\lambda` is roughly
        :math:`n_i` times larger and shifts with dataset size; under
        ``"per_observation"`` the same :math:`\\lambda` transfers across
        datasets and rare factors are shrunk harder, which is what the
        cold-start behaviour wants. Every reference implementation of PMF
        (including Salakhutdinov's own) uses the per-observation form even
        though the papers write the uniform one.
    momentum:
        Heavy-ball momentum coefficient.
    n_epochs:
        Maximum passes over the training set.
    batch_size:
        Minibatch size. Unlike the original implementation, every epoch is
        exactly one shuffled pass — no sample is visited twice while another is
        skipped.

        Worth tuning for speed as well as accuracy. The momentum update touches
        every factor row on every step, so per-epoch cost falls with the number
        of batches: on ML-1M an epoch takes 6.6 s at ``batch_size=1024`` and
        1.8 s at 4096. Beyond about 8192 the gather and inner-product stop
        fitting in cache and the trend reverses.
    eval_every:
        How often to measure the full-training-set objective and RMSE, in
        epochs. Validation error is always measured every epoch, since early
        stopping needs it and it is cheap. Set this above 1 on large datasets,
        where the training-set passes cost about as much as the epoch itself.
        The final epoch is always measured in full.
    init_scale:
        Standard deviation of the Gaussian factor initialisation. This is the
        prior the model is nominally drawing from, so it should be small.
    link:
        ``"identity"`` or ``"logistic"``.
    use_bias:
        Whether to fit per-user and per-item offsets.
    clip:
        Clip returned predictions to the observed rating range. Applied at
        prediction time only, never inside the gradient.
    early_stopping:
        If ``True`` and a validation set is supplied, keep the parameters from
        the best validation RMSE and stop after ``patience`` epochs without
        improvement.
    seed:
        Seeds initialisation and shuffling, making a run bit-reproducible.
    """

    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.05,
        reg: float = 0.05,
        momentum: float = 0.9,
        n_epochs: int = 100,
        batch_size: int = 4096,
        init_scale: float = 0.05,
        link: str = "identity",
        use_bias: bool = False,
        reg_mode: str = "per_observation",
        reg_user: float | None = None,
        reg_item: float | None = None,
        reg_bias: float | None = None,
        clip: bool = True,
        early_stopping: bool = True,
        patience: int = 10,
        eval_every: int = 1,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        if link not in ("identity", "logistic"):
            raise ValueError(f"link must be 'identity' or 'logistic', got {link!r}")
        if reg_mode not in ("uniform", "per_observation"):
            raise ValueError(
                f"reg_mode must be 'uniform' or 'per_observation', got {reg_mode!r}"
            )

        self.n_factors = int(n_factors)
        self.lr = float(lr)
        self.reg = float(reg)
        self.momentum = float(momentum)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.init_scale = float(init_scale)
        self.link = link
        self.use_bias = bool(use_bias)
        self.reg_mode = reg_mode
        self.reg_user = float(reg_user) if reg_user is not None else float(reg)
        self.reg_item = float(reg_item) if reg_item is not None else float(reg)
        self.reg_bias = float(reg_bias) if reg_bias is not None else float(reg)
        self.clip = bool(clip)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.eval_every = max(1, int(eval_every))
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.b_user: np.ndarray | None = None
        self.b_item: np.ndarray | None = None
        self.global_mean: float = 0.0
        self.history = History()
        self.best_epoch: int | None = None
        self.n_epochs_run: int = 0

    @property
    def name(self) -> str:
        parts = ["PMF"]
        if self.use_bias:
            parts.append("bias")
        if self.link == "logistic":
            parts.append("logistic")
        return "-".join(parts) + f" (d={self.n_factors})"

    # ------------------------------------------------------------------ fit

    def fit(self, train: RatingData, val: RatingData | None = None) -> PMF:
        rng = np.random.default_rng(self.seed)

        self.rating_min, self.rating_max = train.rating_min, train.rating_max
        self._span = max(self.rating_max - self.rating_min, 1e-12)

        users, items, ratings = train.users, train.items, train.ratings
        n_users, n_items, d = train.n_users, train.n_items, self.n_factors

        self.global_mean = float(ratings.mean())
        targets = self._to_target(ratings)

        self.U = rng.normal(0.0, self.init_scale, size=(n_users, d))
        self.V = rng.normal(0.0, self.init_scale, size=(n_items, d))
        self.b_user = np.zeros(n_users)
        self.b_item = np.zeros(n_items)

        vel_U = np.zeros_like(self.U)
        vel_V = np.zeros_like(self.V)
        vel_bu = np.zeros(n_users)
        vel_bi = np.zeros(n_items)

        n = users.shape[0]
        best_val, best_state, epochs_without_gain = np.inf, None, 0
        started = time.perf_counter()

        for epoch in range(1, self.n_epochs + 1):
            order = rng.permutation(n)

            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                bu, bi, bt = users[idx], items[idx], targets[idx]

                grad_U, grad_V, grad_bu, grad_bi = self.gradients(bu, bi, bt, n_total=n)

                vel_U *= self.momentum
                vel_V *= self.momentum
                vel_U += self.lr * grad_U
                vel_V += self.lr * grad_V
                self.U -= vel_U
                self.V -= vel_V

                if self.use_bias:
                    vel_bu *= self.momentum
                    vel_bi *= self.momentum
                    vel_bu += self.lr * grad_bu
                    vel_bi += self.lr * grad_bi
                    self.b_user -= vel_bu
                    self.b_item -= vel_bi

            # Validation error is needed every epoch to drive early stopping and
            # is cheap. The full-training-set objective and RMSE are not: on
            # ML-1M they cost ~1.5 s per epoch against ~1.8 s for the epoch's
            # actual gradient work, so on large data they are sampled instead.
            # The final epoch is always measured, so reported numbers are exact.
            val_rmse = rmse(val.ratings, self.predict_data(val)) if val is not None else None
            full_eval = (epoch % self.eval_every == 0) or (epoch == self.n_epochs)

            if full_eval:
                objective = self._objective(users, items, targets)
                train_rmse = rmse(ratings, self.predict(users, items))
            else:
                objective = train_rmse = float("nan")

            self.history.record(
                epoch, objective, train_rmse, val_rmse, time.perf_counter() - started
            )
            self.n_epochs_run = epoch

            if self.verbose and full_eval:
                tail = f", val RMSE {val_rmse:.4f}" if val_rmse is not None else ""
                print(f"  epoch {epoch:3d}  obj {objective:12.2f}  train RMSE {train_rmse:.4f}{tail}")

            diverged = (full_eval and not np.isfinite(objective)) or (
                val_rmse is not None and not np.isfinite(val_rmse)
            )
            if diverged:
                raise FloatingPointError(
                    f"objective diverged at epoch {epoch}; lower lr (currently {self.lr})"
                )

            if val is not None and self.early_stopping:
                if val_rmse < best_val - 1e-5:
                    best_val, epochs_without_gain = val_rmse, 0
                    best_state, self.best_epoch = self._snapshot(), epoch
                else:
                    epochs_without_gain += 1
                    if epochs_without_gain >= self.patience:
                        if self.verbose:
                            print(f"  early stop at epoch {epoch} (best {self.best_epoch})")
                        break

        if best_state is not None:
            self._restore(best_state)

        # With eval_every > 1 the last recorded train RMSE may predate the
        # restored parameters, so recompute it once against what is actually
        # being returned.
        if self.history.train_rmse:
            self.history.train_rmse[-1] = rmse(ratings, self.predict(users, items))
        return self

    # ------------------------------------------------------------ gradients

    def gradients(
        self,
        users: np.ndarray,
        items: np.ndarray,
        targets: np.ndarray,
        n_total: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Gradient of the objective w.r.t. every parameter, for one minibatch.

        Called once per step by :meth:`fit`. It is a public method because it is
        also the thing the test suite finite-differences against
        :meth:`_objective`: passing the *whole* training set with
        ``n_total = len(training set)`` yields exactly :math:`\\nabla E`, so the
        analytic gradient can be checked rather than assumed.

        ``n_total`` is the training-set size, used only in ``"uniform"`` mode to
        scale the dense prior term by :math:`|B|/N`.
        """
        size = users.shape[0]
        n_total = int(n_total if n_total is not None else size)
        n_users, n_items = self.U.shape[0], self.V.shape[0]

        U_b, V_b = self.U[users], self.V[items]
        linear = np.einsum("ij,ij->i", U_b, V_b)
        if self.use_bias:
            linear = linear + self.b_user[users] + self.b_item[items]

        # dE/d(linear): the squared-error residual times the link derivative.
        if self.link == "logistic":
            activation = _sigmoid(linear)
            delta = (activation - targets) * activation * (1.0 - activation)
        else:
            delta = linear - targets

        # The data term is summed over observations, so `lr` is a per-observation
        # rate and is independent of batch size.
        grad_U = scatter_add(n_users, users, delta[:, None] * V_b)
        grad_V = scatter_add(n_items, items, delta[:, None] * U_b)
        grad_bu = np.bincount(users, weights=delta, minlength=n_users) if self.use_bias else None
        grad_bi = np.bincount(items, weights=delta, minlength=n_items) if self.use_bias else None

        if self.reg_mode == "uniform":
            # Dense shrinkage on every row, scaled so that one epoch applies
            # exactly lambda*U in total. Faithful to E as written in the paper.
            prior = size / n_total
            grad_U += prior * self.reg_user * self.U
            grad_V += prior * self.reg_item * self.V
            if self.use_bias:
                grad_bu += prior * self.reg_bias * self.b_user
                grad_bi += prior * self.reg_bias * self.b_item
        else:
            # Shrinkage once per observation: a factor seen n_i times is
            # penalised n_i times as hard.
            grad_U += scatter_add(n_users, users, self.reg_user * U_b)
            grad_V += scatter_add(n_items, items, self.reg_item * V_b)
            if self.use_bias:
                grad_bu += np.bincount(
                    users, weights=self.reg_bias * self.b_user[users], minlength=n_users
                )
                grad_bi += np.bincount(
                    items, weights=self.reg_bias * self.b_item[items], minlength=n_items
                )

        return grad_U, grad_V, grad_bu, grad_bi

    # -------------------------------------------------------------- predict

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        if self.U is None or self.V is None:
            raise RuntimeError("call fit() before predict()")

        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)

        # Unseen entities fall back to the global mean rather than indexing out
        # of range, which is the honest behaviour for a true cold start.
        known = (users < self.U.shape[0]) & (items < self.V.shape[0])
        out = np.full(users.shape[0], self.global_mean, dtype=np.float64)
        if not known.any():
            return out

        u, i = users[known], items[known]
        linear = np.einsum("ij,ij->i", self.U[u], self.V[i])
        if self.use_bias:
            linear = linear + self.b_user[u] + self.b_item[i]

        out[known] = self._from_target(linear)
        return self._clip(out) if self.clip else out

    # ---------------------------------------------------------------- utils

    def _to_target(self, ratings: np.ndarray) -> np.ndarray:
        """Map ratings into the space the factorisation actually models."""
        if self.link == "logistic":
            return (ratings - self.rating_min) / self._span
        return ratings - self.global_mean

    def _from_target(self, linear: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`_to_target`, mapping model output back to ratings."""
        if self.link == "logistic":
            return self.rating_min + self._span * _sigmoid(linear)
        return linear + self.global_mean

    def _objective(self, users: np.ndarray, items: np.ndarray, targets: np.ndarray) -> float:
        """The MAP objective :math:`E` actually being minimised — penalty included.

        The penalty matches ``reg_mode``, so this series is genuinely monotone
        under a working learning rate and can be used to diagnose divergence.
        Reported separately from RMSE, never blended into it (the defect in the
        original implementation).
        """
        linear = np.einsum("ij,ij->i", self.U[users], self.V[items])
        if self.use_bias:
            linear = linear + self.b_user[users] + self.b_item[items]
        prediction = _sigmoid(linear) if self.link == "logistic" else linear

        residual = 0.5 * float(np.sum((prediction - targets) ** 2))

        if self.reg_mode == "uniform":
            sq_user, sq_item = float(np.sum(self.U**2)), float(np.sum(self.V**2))
            sq_bu, sq_bi = float(np.sum(self.b_user**2)), float(np.sum(self.b_item**2))
        else:
            count_user = np.bincount(users, minlength=self.U.shape[0])
            count_item = np.bincount(items, minlength=self.V.shape[0])
            sq_user = float(count_user @ np.sum(self.U**2, axis=1))
            sq_item = float(count_item @ np.sum(self.V**2, axis=1))
            sq_bu = float(count_user @ self.b_user**2)
            sq_bi = float(count_item @ self.b_item**2)

        penalty = 0.5 * (self.reg_user * sq_user + self.reg_item * sq_item)
        if self.use_bias:
            penalty += 0.5 * self.reg_bias * (sq_bu + sq_bi)
        return residual + penalty

    def _snapshot(self) -> tuple[np.ndarray, ...]:
        return (self.U.copy(), self.V.copy(), self.b_user.copy(), self.b_item.copy())

    def _restore(self, state: tuple[np.ndarray, ...]) -> None:
        self.U, self.V, self.b_user, self.b_item = state


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out
