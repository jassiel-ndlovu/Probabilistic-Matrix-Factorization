"""Model correctness: gradients, recovery, reproducibility, bounds."""

from __future__ import annotations

import numpy as np
import pytest

from pmf.baselines import BiasBaseline, GlobalMean, ItemMean, UserMean
from pmf.data import RatingData, train_test_split
from pmf.metrics import rmse
from pmf.models import BPMF, PMF


def low_rank_data(
    n_users: int = 60, n_items: int = 40, rank: int = 3, seed: int = 0, noise: float = 0.0
) -> RatingData:
    """A dense matrix that genuinely *is* low rank, so a correct factoriser
    with enough capacity must be able to drive training error towards zero."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_users, rank))
    V = rng.normal(size=(n_items, rank))
    matrix = U @ V.T
    matrix = 1.0 + 4.0 * (matrix - matrix.min()) / (matrix.max() - matrix.min())
    if noise:
        matrix = matrix + rng.normal(0, noise, matrix.shape)

    users, items = np.meshgrid(np.arange(n_users), np.arange(n_items), indexing="ij")
    triples = np.column_stack(
        [users.ravel(), items.ravel(), np.clip(matrix.ravel(), 1.0, 5.0)]
    ).astype(np.float64)
    return RatingData("synthetic", triples, n_users, n_items, 1.0, 5.0)


# --------------------------------------------------------------- gradients


@pytest.mark.parametrize("reg_mode", ["uniform", "per_observation"])
@pytest.mark.parametrize("link", ["identity", "logistic"])
@pytest.mark.parametrize("use_bias", [False, True])
def test_analytic_gradient_matches_finite_differences(
    reg_mode: str, link: str, use_bias: bool
) -> None:
    """The load-bearing test. Every variant's hand-derived gradient is checked
    against a central finite difference of the objective it claims to descend.
    A sign error or a missing link-derivative factor fails here."""
    rng = np.random.default_rng(0)
    n_users, n_items, d = 7, 9, 4
    n_obs = 40

    model = PMF(
        n_factors=d, reg=0.3, reg_mode=reg_mode, link=link, use_bias=use_bias, seed=0
    )
    model.rating_min, model.rating_max = 1.0, 5.0
    model._span = 4.0
    model.global_mean = 3.0
    model.U = rng.normal(0, 0.5, size=(n_users, d))
    model.V = rng.normal(0, 0.5, size=(n_items, d))
    model.b_user = rng.normal(0, 0.2, size=n_users)
    model.b_item = rng.normal(0, 0.2, size=n_items)

    users = rng.integers(0, n_users, n_obs)
    items = rng.integers(0, n_items, n_obs)
    targets = rng.uniform(0, 1, n_obs) if link == "logistic" else rng.normal(0, 1, n_obs)

    grad_U, grad_V, grad_bu, grad_bi = model.gradients(users, items, targets, n_total=n_obs)

    eps = 1e-6
    for name, analytic in (("U", grad_U), ("V", grad_V), ("b_user", grad_bu), ("b_item", grad_bi)):
        if analytic is None:
            continue
        param = getattr(model, name)
        numeric = np.zeros_like(param)
        iterator = np.ndindex(*param.shape)
        for index in iterator:
            original = param[index]
            param[index] = original + eps
            plus = model._objective(users, items, targets)
            param[index] = original - eps
            minus = model._objective(users, items, targets)
            param[index] = original
            numeric[index] = (plus - minus) / (2 * eps)

        np.testing.assert_allclose(
            analytic, numeric, rtol=1e-4, atol=1e-6, err_msg=f"gradient mismatch for {name}"
        )


def test_objective_decreases_monotonically_on_a_clean_problem() -> None:
    """With a working learning rate the MAP objective must fall every epoch;
    if it does not, the step or the penalty term is inconsistent."""
    data = low_rank_data(noise=0.0)
    model = PMF(
        n_factors=5, lr=0.002, reg=0.01, momentum=0.8, n_epochs=40,
        batch_size=256, early_stopping=False, seed=0,
    ).fit(data)
    objective = np.array(model.history.objective)
    assert np.all(np.diff(objective) < 1e-6), "objective increased during training"
    assert objective[-1] < objective[0] * 0.5


# ---------------------------------------------------------------- recovery


def test_pmf_recovers_a_noiseless_low_rank_matrix() -> None:
    data = low_rank_data(rank=3, noise=0.0)
    model = PMF(
        n_factors=6, lr=0.005, reg=0.0, momentum=0.8, n_epochs=300,
        batch_size=256, early_stopping=False, seed=0,
    ).fit(data)
    assert rmse(data.ratings, model.predict_data(data)) < 0.05


def test_pmf_beats_the_global_mean_on_real_data(ml100k_split) -> None:
    train, test = ml100k_split.train, ml100k_split.test
    baseline = GlobalMean().fit(train)
    model = PMF(n_factors=10, lr=0.005, reg=0.1, n_epochs=40, seed=0).fit(train)
    assert rmse(test.ratings, model.predict_data(test)) < rmse(
        test.ratings, baseline.predict_data(test)
    )


# --------------------------------------------------------- reproducibility


def test_same_seed_gives_identical_parameters() -> None:
    data = low_rank_data()
    kwargs = dict(n_factors=4, lr=0.005, n_epochs=10, early_stopping=False)
    a = PMF(seed=42, **kwargs).fit(data)
    b = PMF(seed=42, **kwargs).fit(data)
    np.testing.assert_array_equal(a.U, b.U)
    np.testing.assert_array_equal(a.V, b.V)


def test_different_seeds_give_different_parameters() -> None:
    data = low_rank_data()
    kwargs = dict(n_factors=4, lr=0.005, n_epochs=10, early_stopping=False)
    a = PMF(seed=1, **kwargs).fit(data)
    b = PMF(seed=2, **kwargs).fit(data)
    assert not np.allclose(a.U, b.U)


# ----------------------------------------------------------------- bounds


def test_predictions_are_clipped_to_the_rating_range() -> None:
    data = low_rank_data()
    model = PMF(n_factors=4, n_epochs=5, clip=True, seed=0).fit(data)
    predictions = model.predict_data(data)
    assert predictions.min() >= data.rating_min
    assert predictions.max() <= data.rating_max


def test_logistic_link_is_bounded_without_clipping() -> None:
    """The point of the sigmoid variant: the range constraint is structural, so
    it holds even with clipping disabled."""
    data = low_rank_data()
    model = PMF(
        n_factors=4, link="logistic", lr=0.05, n_epochs=30, clip=False, seed=0
    ).fit(data)
    predictions = model.predict_data(data)
    assert predictions.min() >= data.rating_min - 1e-9
    assert predictions.max() <= data.rating_max + 1e-9


def test_unseen_entities_fall_back_to_the_global_mean() -> None:
    data = low_rank_data(n_users=20, n_items=15)
    model = PMF(n_factors=4, n_epochs=5, seed=0).fit(data)
    prediction = model.predict(np.array([999]), np.array([999]))
    assert prediction[0] == pytest.approx(model.global_mean)


def test_early_stopping_restores_the_best_epoch() -> None:
    data = low_rank_data(noise=0.8)
    split = train_test_split(data, test_size=0.3, seed=0)
    model = PMF(
        n_factors=20, lr=0.01, reg=0.0, n_epochs=200, patience=5,
        early_stopping=True, seed=0,
    ).fit(split.train, split.test)

    assert model.best_epoch is not None
    achieved = rmse(split.test.ratings, model.predict_data(split.test))
    assert achieved == pytest.approx(min(model.history.val_rmse), abs=1e-6)


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="link"):
        PMF(link="relu")
    with pytest.raises(ValueError, match="reg_mode"):
        PMF(reg_mode="l1")
    with pytest.raises(RuntimeError, match="fit"):
        PMF().predict(np.array([0]), np.array([0]))


# ------------------------------------------------------------------- BPMF


def test_bpmf_runs_and_beats_the_global_mean(ml100k_split) -> None:
    train, test = ml100k_split.train, ml100k_split.test
    model = BPMF(n_factors=10, n_iter=12, burn_in=4, seed=0).fit(train)
    assert len(model.samples_U) == 8
    assert rmse(test.ratings, model.predict_data(test)) < rmse(
        test.ratings, GlobalMean().fit(train).predict_data(test)
    )


def test_bpmf_averages_predictions_not_factors() -> None:
    """Averaging factors across sweeps is meaningless under the rotational
    non-identifiability of UV^T; the estimator must average predictions."""
    data = low_rank_data(n_users=25, n_items=20)
    model = BPMF(n_factors=4, n_iter=10, burn_in=2, seed=0).fit(data)

    users = np.array([0, 1, 2])
    items = np.array([0, 1, 2])
    samples = zip(model.samples_U, model.samples_V, strict=True)
    expected = np.mean(
        [np.einsum("ij,ij->i", U[users], V[items]) for U, V in samples], axis=0
    )
    naive = np.einsum(
        "ij,ij->i",
        np.mean(model.samples_U, axis=0)[users],
        np.mean(model.samples_V, axis=0)[items],
    )
    actual = model.predict(users, items) - model.global_mean
    expected_clipped = np.clip(expected + model.global_mean, 1, 5) - model.global_mean

    np.testing.assert_allclose(actual, expected_clipped, atol=1e-9)
    assert not np.allclose(expected, naive, atol=1e-6)


def test_bpmf_requires_samples() -> None:
    with pytest.raises(RuntimeError, match="burn_in"):
        BPMF(n_iter=5, burn_in=10, seed=0).fit(low_rank_data(n_users=10, n_items=10))


# -------------------------------------------------------------- baselines


@pytest.mark.parametrize("cls", [GlobalMean, UserMean, ItemMean, BiasBaseline])
def test_baselines_predict_within_range(cls, ml100k_split) -> None:
    model = cls().fit(ml100k_split.train)
    predictions = model.predict_data(ml100k_split.test)
    assert predictions.shape == (len(ml100k_split.test),)
    assert np.all(np.isfinite(predictions))


def test_baseline_ordering_is_as_expected(ml100k_split) -> None:
    """Each baseline should strictly improve on the previous one; if this
    ordering ever breaks, the split or the fit is wrong."""
    train, test = ml100k_split.train, ml100k_split.test
    scores = {
        cls.__name__: rmse(test.ratings, cls().fit(train).predict_data(test))
        for cls in (GlobalMean, UserMean, ItemMean, BiasBaseline)
    }
    assert scores["BiasBaseline"] < scores["ItemMean"] < scores["GlobalMean"]
    assert scores["BiasBaseline"] < scores["UserMean"] < scores["GlobalMean"]


def test_global_mean_equals_training_mean(ml100k_split) -> None:
    model = GlobalMean().fit(ml100k_split.train)
    assert model.mu == pytest.approx(ml100k_split.train.ratings.mean())
