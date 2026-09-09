"""Collaborative filtering cost functions.

Andrew Ng's formulation, implemented from scratch. The parameters are learned
simultaneously:

    prediction(i, j) = W[j] . X[i] + b[0, j]

    J(W, b, X) = 0.5 * sum_{(i,j) : R[i,j]=1} (W[j].X[i] + b[0,j] - Y[i,j])^2
                 + (lambda/2) * sum(W^2)
                 + (lambda/2) * sum(X^2)

Shapes used throughout this module:

    X : (n_m, n)   learned item feature matrix
    W : (n_u, n)   learned user parameter matrix
    b : (1, n_u)   per-user bias
    Y : (n_m, n_u) ratings
    R : (n_m, n_u) 1 where user j rated item i, else 0

``W`` and ``b`` are kept as separate tensors rather than folded into one
augmented parameter matrix. Merging them is a common trick, but it makes the
bias share ``W``'s regularisation, and here the bias is deliberately
unregularised: penalising it would drag every user's baseline toward zero,
which is exactly the per-user offset the model is supposed to keep.

Each vectorized cost has a slow loop-based twin. The twins exist to be tested
against, not to be run: a vectorized masked sum is easy to get subtly wrong, and
an independent transcription of the formula is the cheapest way to prove it is
not.
"""

from __future__ import annotations

import enum

import numpy as np
import tensorflow as tf

FloatArray = np.ndarray


class RatingMode(enum.StrEnum):
    """Which feedback signal the model is trained on.

    ``EXPLICIT`` uses 1-5 star ratings with squared error. ``IMPLICIT`` uses the
    impression log - shown-and-ordered (1) against shown-and-not-ordered (0) -
    with binary cross-entropy. Restaurants generally have far more of the second
    kind of data than the first.
    """

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


# ---------------------------------------------------------------------------
# Explicit feedback: squared error
# ---------------------------------------------------------------------------


def cofi_cost_func(
    X: tf.Tensor,
    W: tf.Tensor,
    b: tf.Tensor,
    Y: tf.Tensor,
    R: tf.Tensor,
    lambda_: float,
) -> tf.Tensor:
    """Vectorized collaborative filtering cost for explicit ratings.

    Args:
        X: (n_m, n) item feature matrix.
        W: (n_u, n) user parameter matrix.
        b: (1, n_u) per-user bias.
        Y: (n_m, n_u) ratings.
        R: (n_m, n_u) binary mask, 1 where a rating exists.
        lambda_: L2 regularisation strength, applied to X and W only.

    Returns:
        Scalar tensor holding J(W, b, X).

    The mask is applied to the *error* rather than used to select entries, so
    unrated pairs contribute exactly zero to the sum while the whole thing stays
    one dense matrix operation that ``GradientTape`` can differentiate.
    """
    errors = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    squared_error = 0.5 * tf.reduce_sum(errors**2)
    regularisation = (lambda_ / 2.0) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return squared_error + regularisation


def cofi_cost_func_loop(
    X: FloatArray,
    W: FloatArray,
    b: FloatArray,
    Y: FloatArray,
    R: FloatArray,
    lambda_: float,
) -> float:
    """Loop-based reference implementation of :func:`cofi_cost_func`.

    Args:
        X: (n_m, n) item feature matrix.
        W: (n_u, n) user parameter matrix.
        b: (1, n_u) per-user bias.
        Y: (n_m, n_u) ratings.
        R: (n_m, n_u) binary mask.
        lambda_: L2 regularisation strength.

    Returns:
        J(W, b, X) as a Python float.

    Deliberately slow and literal - one term of the summation per iteration.
    This is the transcription the vectorized version is checked against; it is
    never used for training.
    """
    n_m, n_u = Y.shape
    total = 0.0
    for j in range(n_u):
        w = W[j, :]
        b_j = b[0, j]
        for i in range(n_m):
            x = X[i, :]
            total += np.square(R[i, j] * (np.dot(w, x) + b_j - Y[i, j]))
    total = total / 2.0
    total += (lambda_ / 2.0) * (np.sum(np.square(W)) + np.sum(np.square(X)))
    return float(total)


# ---------------------------------------------------------------------------
# Implicit feedback: binary cross-entropy
# ---------------------------------------------------------------------------


def cofi_cost_func_binary(
    X: tf.Tensor,
    W: tf.Tensor,
    b: tf.Tensor,
    Y: tf.Tensor,
    R: tf.Tensor,
    lambda_: float,
) -> tf.Tensor:
    """Vectorized cost for implicit (binary) feedback.

    Args:
        X: (n_m, n) item feature matrix.
        W: (n_u, n) user parameter matrix.
        b: (1, n_u) per-user bias.
        Y: (n_m, n_u) binary labels - 1 ordered, 0 shown and not ordered.
        R: (n_m, n_u) binary mask, 1 where the item was actually shown.
        lambda_: L2 regularisation strength, applied to X and W only.

    Returns:
        Scalar tensor holding the regularised cross-entropy.

    The prediction is g(W[j].X[i] + b[0,j]) with g the logistic sigmoid, but the
    loss is computed from the logits directly via
    ``sigmoid_cross_entropy_with_logits``. Taking log of an explicit sigmoid
    overflows to infinity once a logit saturates, which happens routinely during
    training.

    R matters more here than in the explicit case: an unseen pair is genuinely
    unknown, not a zero. Treating every non-order as a rejection would teach the
    model that customers dislike every dish they were never shown.
    """
    logits = tf.linalg.matmul(X, tf.transpose(W)) + b
    per_entry = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)
    masked = tf.reduce_sum(per_entry * R)
    regularisation = (lambda_ / 2.0) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return masked + regularisation


def cofi_cost_func_binary_loop(
    X: FloatArray,
    W: FloatArray,
    b: FloatArray,
    Y: FloatArray,
    R: FloatArray,
    lambda_: float,
) -> float:
    """Loop-based reference implementation of :func:`cofi_cost_func_binary`.

    Args:
        X: (n_m, n) item feature matrix.
        W: (n_u, n) user parameter matrix.
        b: (1, n_u) per-user bias.
        Y: (n_m, n_u) binary labels.
        R: (n_m, n_u) binary mask.
        lambda_: L2 regularisation strength.

    Returns:
        The regularised cross-entropy as a Python float.

    Written from the textbook definition -log(g(z)) / -log(1-g(z)) rather than
    the stable logit form, so that it is a genuinely independent check on the
    vectorized version instead of a copy of its numerics.
    """
    n_m, n_u = Y.shape
    total = 0.0
    for j in range(n_u):
        w = W[j, :]
        b_j = b[0, j]
        for i in range(n_m):
            if R[i, j] == 0:
                continue
            z = float(np.dot(w, X[i, :]) + b_j)
            g = 1.0 / (1.0 + np.exp(-z))
            y = Y[i, j]
            total += -(y * np.log(g) + (1.0 - y) * np.log(1.0 - g))
    total += (lambda_ / 2.0) * (np.sum(np.square(W)) + np.sum(np.square(X)))
    return float(total)


# ---------------------------------------------------------------------------
# Mean normalisation
# ---------------------------------------------------------------------------


def normalize_ratings(Y: FloatArray, R: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Subtract each item's mean rating, over rated entries only.

    Args:
        Y: (n_m, n_u) ratings.
        R: (n_m, n_u) binary mask.

    Returns:
        ``(Y_norm, mu)`` where ``Y_norm`` is (n_m, n_u) with the item mean removed
        from every rated entry and zeros elsewhere, and ``mu`` is (n_m,) the mean
        rating of each item.

    Why this is not optional: a user who has rated nothing gets W[j] = 0 and
    b[0,j] = 0 from the regulariser, so their raw prediction for every item is
    zero. Zero is not a neutral rating on a 1-5 scale - it is worse than the
    worst possible review, and the entire menu would rank identically badly.
    Training on mean-centred data and adding mu[i] back at prediction time means
    that same user instead receives each item's average rating, which is the
    correct thing to believe about someone you know nothing about.

    An item nobody has rated has no mean of its own, so it falls back to the
    global mean over all observed ratings. Leaving it at zero would push brand
    new dishes to the bottom of every ranking for reasons unrelated to quality.
    """
    rated_per_item = R.sum(axis=1)
    observed = R.sum()
    global_mean = float((Y * R).sum() / observed) if observed else 0.0

    safe_counts = np.where(rated_per_item == 0, 1.0, rated_per_item)
    mu = (Y * R).sum(axis=1) / safe_counts
    mu = np.where(rated_per_item == 0, global_mean, mu)

    Y_norm = (Y - mu[:, None]) * R
    return Y_norm, mu
