"""PMF trained with PyTorch autograd and Adam.

The original report states that its result was produced "with a latent
dimensionality of 20, optimised using the Adam optimiser". This module exists so
that configuration can be reproduced exactly rather than approximated by the
NumPy trainer, which uses SGD with momentum as in the 2007 paper.

The objective is identical to :class:`pmf.models.pmf.PMF`; only the optimiser
and the autodiff backend differ. Where the two disagree on a given dataset, the
difference is attributable to the optimiser, which is the comparison the
re-execution study needs.

PyTorch is an optional dependency — importing this module without it raises a
clear error rather than failing at first use.
"""

from __future__ import annotations

import time

import numpy as np

from ..data import RatingData
from ..metrics import rmse
from .base import History, Recommender

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "TorchPMF requires PyTorch. Install it with `pip install -e .[torch]`, "
        "or use pmf.models.PMF for the dependency-free NumPy trainer."
    ) from exc

__all__ = ["TorchPMF"]


class _Factorization(nn.Module):
    """Embedding tables plus optional additive offsets."""

    def __init__(self, n_users: int, n_items: int, n_factors: int, use_bias: bool, init_scale: float):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        nn.init.normal_(self.user_factors.weight, std=init_scale)
        nn.init.normal_(self.item_factors.weight, std=init_scale)

        self.use_bias = use_bias
        if use_bias:
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        out = (self.user_factors(users) * self.item_factors(items)).sum(dim=1)
        if self.use_bias:
            out = out + self.user_bias(users).squeeze(1) + self.item_bias(items).squeeze(1)
        return out


class TorchPMF(Recommender):
    """PMF with an Adam optimiser and explicit L2 penalties.

    Regularisation is applied as an explicit penalty on the *minibatch's* factors
    rather than through Adam's ``weight_decay``. The two are not equivalent here:
    ``weight_decay`` would decay every embedding row on every step, including the
    overwhelming majority not present in the batch, which for a sparse
    factorisation shrinks rarely-seen factors towards zero far more aggressively
    than the MAP objective prescribes.
    """

    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.01,
        reg: float = 0.05,
        n_epochs: int = 100,
        batch_size: int = 4096,
        init_scale: float = 0.05,
        use_bias: bool = False,
        clip: bool = True,
        early_stopping: bool = True,
        patience: int = 10,
        seed: int = 0,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        self.n_factors = int(n_factors)
        self.lr = float(lr)
        self.reg = float(reg)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.init_scale = float(init_scale)
        self.use_bias = bool(use_bias)
        self.clip = bool(clip)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.verbose = bool(verbose)

        self.model: _Factorization | None = None
        self.global_mean = 0.0
        self.history = History()
        self.best_epoch: int | None = None
        self.n_epochs_run = 0

    @property
    def name(self) -> str:
        suffix = "-bias" if self.use_bias else ""
        return f"PMF-Adam{suffix} (d={self.n_factors})"

    def fit(self, train: RatingData, val: RatingData | None = None) -> TorchPMF:
        torch.manual_seed(self.seed)
        self.rating_min, self.rating_max = train.rating_min, train.rating_max
        self.global_mean = float(train.ratings.mean())

        users = torch.as_tensor(train.users, dtype=torch.long, device=self.device)
        items = torch.as_tensor(train.items, dtype=torch.long, device=self.device)
        targets = torch.as_tensor(
            train.ratings - self.global_mean, dtype=torch.float32, device=self.device
        )

        self.model = _Factorization(
            train.n_users, train.n_items, self.n_factors, self.use_bias, self.init_scale
        ).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        n = users.shape[0]
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        best_val, best_state, epochs_without_gain = np.inf, None, 0
        started = time.perf_counter()

        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            order = torch.randperm(n, generator=generator).to(self.device)
            running = 0.0

            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                bu, bi, bt = users[idx], items[idx], targets[idx]

                prediction = self.model(bu, bi)
                loss = 0.5 * ((prediction - bt) ** 2).sum()
                penalty = 0.5 * self.reg * (
                    self.model.user_factors(bu).pow(2).sum()
                    + self.model.item_factors(bi).pow(2).sum()
                )
                if self.use_bias:
                    penalty = penalty + 0.5 * self.reg * (
                        self.model.user_bias(bu).pow(2).sum()
                        + self.model.item_bias(bi).pow(2).sum()
                    )
                objective = (loss + penalty) / idx.shape[0]

                optimiser.zero_grad(set_to_none=True)
                objective.backward()
                optimiser.step()
                running += float(objective) * idx.shape[0]

            train_rmse = rmse(train.ratings, self.predict_data(train))
            val_rmse = rmse(val.ratings, self.predict_data(val)) if val is not None else None
            self.history.record(
                epoch, running, train_rmse, val_rmse, time.perf_counter() - started
            )
            self.n_epochs_run = epoch

            if self.verbose:
                tail = f", val RMSE {val_rmse:.4f}" if val_rmse is not None else ""
                print(f"  epoch {epoch:3d}  train RMSE {train_rmse:.4f}{tail}")

            if val is not None and self.early_stopping:
                if val_rmse < best_val - 1e-5:
                    best_val, epochs_without_gain = val_rmse, 0
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    self.best_epoch = epoch
                else:
                    epochs_without_gain += 1
                    if epochs_without_gain >= self.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() before predict()")

        self.model.eval()
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        known = (users < self.model.user_factors.num_embeddings) & (
            items < self.model.item_factors.num_embeddings
        )

        out = np.full(users.shape[0], self.global_mean, dtype=np.float64)
        if not known.any():
            return out

        prediction = self.model(
            torch.as_tensor(users[known], dtype=torch.long, device=self.device),
            torch.as_tensor(items[known], dtype=torch.long, device=self.device),
        )
        out[known] = prediction.cpu().numpy().astype(np.float64) + self.global_mean
        return self._clip(out) if self.clip else out
