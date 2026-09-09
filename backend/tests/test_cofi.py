"""Correctness of the collaborative filtering cost functions.

The headline test is the equivalence check: a vectorized masked sum is easy to
get subtly wrong in a way that still trains to *something*, and comparing it
against an independent loop transcription of the formula is the cheapest proof
that it implements the intended objective.
"""

import numpy as np
import pytest
import tensorflow as tf

from ml.cofi import (
    cofi_cost_func,
    cofi_cost_func_binary,
    cofi_cost_func_binary_loop,
    cofi_cost_func_loop,
    normalize_ratings,
)

#: Tolerance required by the brief. float64 actually agrees to ~1e-12; the loose
#: bound is what is contractually asserted.
TOLERANCE = 1e-4


def _random_problem(
    n_m: int = 9, n_u: int = 7, n: int = 4, density: float = 0.6, seed: int = 0
) -> tuple[np.ndarray, ...]:
    """A small random (X, W, b, Y, R) in float64."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_m, n))
    W = rng.normal(size=(n_u, n))
    b = rng.normal(size=(1, n_u))
    Y = rng.uniform(1.0, 5.0, size=(n_m, n_u))
    R = (rng.uniform(size=(n_m, n_u)) < density).astype(np.float64)
    return X, W, b, Y, R


def _as_tensors(*arrays: np.ndarray) -> list[tf.Tensor]:
    return [tf.constant(a, dtype=tf.float64) for a in arrays]


# ---------------------------------------------------------------------------
# Vectorized vs loop equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("lambda_", [0.0, 1.5])
def test_explicit_vectorized_matches_loop_reference(seed: int, lambda_: float) -> None:
    X, W, b, Y, R = _random_problem(seed=seed)

    vectorized = float(cofi_cost_func(*_as_tensors(X, W, b, Y, R), lambda_))
    reference = cofi_cost_func_loop(X, W, b, Y, R, lambda_)

    assert vectorized == pytest.approx(reference, abs=TOLERANCE)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("lambda_", [0.0, 1.5])
def test_implicit_vectorized_matches_loop_reference(seed: int, lambda_: float) -> None:
    X, W, b, _, R = _random_problem(seed=seed)
    rng = np.random.default_rng(seed + 100)
    # Binary labels, and keep the logits in a moderate range so the reference's
    # direct log(sigmoid(z)) form stays numerically well behaved.
    Y = (rng.uniform(size=R.shape) < 0.35).astype(np.float64)
    X, W, b = X * 0.5, W * 0.5, b * 0.5

    vectorized = float(cofi_cost_func_binary(*_as_tensors(X, W, b, Y, R), lambda_))
    reference = cofi_cost_func_binary_loop(X, W, b, Y, R, lambda_)

    assert vectorized == pytest.approx(reference, abs=TOLERANCE)


def test_equivalence_holds_when_nothing_is_rated() -> None:
    """An all-zero mask must leave only the regularisation term."""
    X, W, b, Y, _ = _random_problem(seed=7)
    R = np.zeros_like(Y)

    vectorized = float(cofi_cost_func(*_as_tensors(X, W, b, Y, R), 2.0))
    reference = cofi_cost_func_loop(X, W, b, Y, R, 2.0)
    expected_regularisation = 1.0 * (np.sum(X**2) + np.sum(W**2))

    assert vectorized == pytest.approx(reference, abs=TOLERANCE)
    assert vectorized == pytest.approx(expected_regularisation, abs=TOLERANCE)


def test_equivalence_holds_when_everything_is_rated() -> None:
    X, W, b, Y, _ = _random_problem(seed=8)
    R = np.ones_like(Y)

    vectorized = float(cofi_cost_func(*_as_tensors(X, W, b, Y, R), 0.7))
    reference = cofi_cost_func_loop(X, W, b, Y, R, 0.7)
    assert vectorized == pytest.approx(reference, abs=TOLERANCE)


def test_unrated_entries_cannot_influence_the_cost() -> None:
    """Changing a rating the mask excludes must not move the cost at all.

    This is the failure mode the mask exists to prevent: if R were applied to
    the prediction instead of the error, or forgotten entirely, the model would
    be fitting made-up zeros for every dish nobody ordered.
    """
    X, W, b, Y, R = _random_problem(seed=9)
    unrated = np.argwhere(R == 0)
    assert len(unrated) > 0, "test needs at least one unrated entry"

    before = float(cofi_cost_func(*_as_tensors(X, W, b, Y, R), 1.0))
    Y_tampered = Y.copy()
    for i, j in unrated:
        Y_tampered[i, j] = 999.0
    after = float(cofi_cost_func(*_as_tensors(X, W, b, Y_tampered, R), 1.0))

    assert before == pytest.approx(after, abs=TOLERANCE)


def test_cost_is_zero_for_a_perfect_unregularised_fit() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(6, 3))
    W = rng.normal(size=(5, 3))
    b = rng.normal(size=(1, 5))
    Y = X @ W.T + b  # exactly reproducible predictions
    R = np.ones_like(Y)

    assert float(cofi_cost_func(*_as_tensors(X, W, b, Y, R), 0.0)) == pytest.approx(0.0, abs=1e-8)


def test_bias_is_not_regularised() -> None:
    """b is excluded from the penalty on purpose: shrinking it toward zero would
    fight the per-user baseline it exists to represent."""
    X, W, b, Y, R = _random_problem(seed=11)
    R = np.zeros_like(R)  # isolate the regularisation term

    small_b = float(cofi_cost_func(*_as_tensors(X, W, b, Y, R), 3.0))
    large_b = float(cofi_cost_func(*_as_tensors(X, W, b * 50.0, Y, R), 3.0))
    assert small_b == pytest.approx(large_b, abs=TOLERANCE)


# ---------------------------------------------------------------------------
# Mean normalisation
# ---------------------------------------------------------------------------


def test_normalisation_removes_the_item_mean_from_rated_entries() -> None:
    Y = np.array([[5.0, 3.0, 0.0], [2.0, 0.0, 4.0]])
    R = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    Y_norm, mu = normalize_ratings(Y, R)

    assert mu == pytest.approx([4.0, 3.0])
    assert Y_norm[0, 0] == pytest.approx(1.0)
    assert Y_norm[0, 1] == pytest.approx(-1.0)
    assert Y_norm[1, 0] == pytest.approx(-1.0)
    assert Y_norm[1, 2] == pytest.approx(1.0)


def test_normalisation_leaves_unrated_entries_at_zero() -> None:
    Y = np.array([[5.0, 3.0, 0.0], [2.0, 0.0, 4.0]])
    R = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    Y_norm, _ = normalize_ratings(Y, R)
    assert Y_norm[0, 2] == 0.0
    assert Y_norm[1, 1] == 0.0


def test_a_user_who_rated_nothing_receives_the_item_mean() -> None:
    """The property mean normalisation exists for.

    An untrained user column ends at W[j] = 0, b[0,j] = 0 because nothing in the
    data pushes it anywhere and the regulariser pulls it to the origin. Without
    normalisation their prediction for every dish is 0 - below the bottom of a
    1-5 scale. With it, the same user is told each dish's average rating.
    """
    Y = np.array([[5.0, 3.0, 0.0], [2.0, 4.0, 0.0], [4.0, 4.0, 0.0]])
    R = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    cold_user = 2

    _, mu = normalize_ratings(Y, R)

    n_m, n_u, n = 3, 3, 4
    X = np.random.default_rng(0).normal(size=(n_m, n))
    W = np.zeros((n_u, n))
    b = np.zeros((1, n_u))

    predictions = (X @ W.T + b)[:, cold_user] + mu
    assert predictions == pytest.approx(mu)
    assert np.all(predictions >= 1.0), "a cold user must not be shown sub-scale predictions"


def test_an_item_nobody_rated_falls_back_to_the_global_mean() -> None:
    """A brand new dish has no mean of its own. Leaving it at zero would sink it
    to the bottom of every ranking for reasons unrelated to how good it is."""
    Y = np.array([[5.0, 3.0], [2.0, 4.0], [0.0, 0.0]])
    R = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])

    _, mu = normalize_ratings(Y, R)

    assert mu[2] == pytest.approx(3.5)  # mean of 5, 3, 2, 4
    assert mu[2] > 0.0


def test_normalisation_handles_a_completely_empty_matrix() -> None:
    Y = np.zeros((3, 2))
    R = np.zeros((3, 2))

    Y_norm, mu = normalize_ratings(Y, R)
    assert not np.isnan(mu).any()
    assert not np.isnan(Y_norm).any()
