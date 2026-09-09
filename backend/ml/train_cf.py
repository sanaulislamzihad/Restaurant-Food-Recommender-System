"""Offline training job for the collaborative filtering model.

    python -m ml.train_cf
    python -m ml.train_cf --mode implicit --iterations 600

Training never happens inside a request handler. This writes a new versioned
directory under ``models/`` which the API then loads read-only.

The optimisation is a hand-written loop over ``tf.GradientTape`` rather than
``model.fit``: X, W and b are free parameters optimised jointly against a custom
objective, not layers of a network, so there is no Keras model to fit in the
first place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from app.core.config import get_settings
from app.db.session import SessionLocal
from ml.artifacts import LOSS_CURVE_FILE, next_version_dir, save_cf_artifacts
from ml.cofi import (
    RatingMode,
    cofi_cost_func,
    cofi_cost_func_binary,
    normalize_ratings,
)
from ml.data import RatingMatrices, load_matrices

#: Parameters are initialised small. With n=10 latent features, standard normal
#: init would give predictions with a standard deviation around sqrt(10), which
#: swamps mean-centred targets that live in roughly [-2, 2] and makes the first
#: few hundred steps a recovery from a bad starting point.
INIT_SCALE = 0.1

LOG_EVERY = 20


@dataclass
class TrainConfig:
    mode: RatingMode = RatingMode.EXPLICIT
    n_features: int = 10
    iterations: int = 400
    learning_rate: float = 1e-1
    lambda_: float = 1.0
    seed: int = 42
    normalize: bool = True
    #: The evaluation harness fits dozens of models; their per-iteration logs
    #: would bury the report it is trying to print.
    verbose: bool = True


@dataclass
class TrainResult:
    X: np.ndarray
    W: np.ndarray
    b: np.ndarray
    mu: np.ndarray
    history: list[float]
    metrics: dict[str, float]
    seconds: float


def _data_fingerprint(matrices: RatingMatrices) -> str:
    """Cheap identifier for the training data.

    Lets a model card be checked against the current database: if the
    fingerprint has moved, the saved model was trained on different data.
    """
    payload = (
        f"{matrices.n_m}:{matrices.n_u}:{matrices.n_observed}:"
        f"{float((matrices.Y * matrices.R).sum()):.4f}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def train(matrices: RatingMatrices, cfg: TrainConfig) -> TrainResult:
    """Fit X, W and b by gradient descent on the collaborative filtering cost."""
    Y_raw, R_raw = matrices.Y, matrices.R

    if cfg.mode is RatingMode.EXPLICIT and cfg.normalize:
        Y_train, mu = normalize_ratings(Y_raw, R_raw)
    else:
        # Implicit mode never mean-normalises: subtracting an average from a 0/1
        # label and pushing the result through a sigmoid is not a defined
        # operation. Item popularity is absorbed by the bias term instead.
        Y_train = Y_raw
        mu = np.zeros(matrices.n_m, dtype=np.float64)

    Y = tf.constant(Y_train, dtype=tf.float64)
    R = tf.constant(R_raw, dtype=tf.float64)

    tf.random.set_seed(cfg.seed)
    initialiser = tf.random.Generator.from_seed(cfg.seed)

    # W and b stay separate tensors throughout - see the note in ml/cofi.py.
    X = tf.Variable(
        initialiser.normal((matrices.n_m, cfg.n_features), dtype=tf.float64) * INIT_SCALE,
        name="X",
    )
    W = tf.Variable(
        initialiser.normal((matrices.n_u, cfg.n_features), dtype=tf.float64) * INIT_SCALE,
        name="W",
    )
    b = tf.Variable(tf.zeros((1, matrices.n_u), dtype=tf.float64), name="b")

    cost_func = cofi_cost_func if cfg.mode is RatingMode.EXPLICIT else cofi_cost_func_binary
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

    history: list[float] = []
    started = time.perf_counter()

    for iteration in range(cfg.iterations):
        with tf.GradientTape() as tape:
            cost = cost_func(X, W, b, Y, R, cfg.lambda_)

        gradients = tape.gradient(cost, [X, W, b])
        optimizer.apply_gradients(zip(gradients, [X, W, b], strict=True))

        cost_value = float(cost)
        history.append(cost_value)
        if cfg.verbose and (iteration % LOG_EVERY == 0 or iteration == cfg.iterations - 1):
            print(f"  iteration {iteration:5d}  cost {cost_value:14,.4f}")

    seconds = time.perf_counter() - started

    X_out = X.numpy()
    W_out = W.numpy()
    b_out = b.numpy()
    metrics = _training_metrics(X_out, W_out, b_out, mu, Y_raw, R_raw, cfg)
    metrics["final_cost"] = history[-1]

    return TrainResult(
        X=X_out, W=W_out, b=b_out, mu=mu, history=history, metrics=metrics, seconds=seconds
    )


def _training_metrics(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    Y: np.ndarray,
    R: np.ndarray,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Fit quality on the data the model was trained on.

    These are *training* numbers and are named as such. They say the optimiser
    converged, not that the model generalises; held-out evaluation against
    baselines is M3's job.
    """
    mask = R == 1
    observed = int(mask.sum())
    if observed == 0:
        return {}

    if cfg.mode is RatingMode.EXPLICIT:
        predictions = X @ W.T + b + mu[:, None]
        errors = (predictions - Y)[mask]
        return {
            "train_rmse": float(np.sqrt(np.mean(errors**2))),
            "train_mae": float(np.mean(np.abs(errors))),
        }

    logits = X @ W.T + b
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    eps = 1e-12
    labels = Y[mask]
    probs = np.clip(probabilities[mask], eps, 1 - eps)
    return {
        "train_log_loss": float(
            -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        ),
        "train_accuracy": float(np.mean((probs >= 0.5) == (labels >= 0.5))),
        "positive_rate": float(np.mean(labels)),
    }


def save_loss_curve(history: list[float], path: Path, cfg: TrainConfig) -> None:
    """Write the loss curve as a PNG next to the model."""
    import matplotlib

    matplotlib.use("Agg")  # no display in a container or CI
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4.2), dpi=140)
    axis.plot(history, color="#c2410c", linewidth=1.8)
    axis.set_xlabel("iteration")
    axis.set_ylabel("cost J(W, b, X)")
    axis.set_title(f"Collaborative filtering training loss ({cfg.mode})")
    # The first few steps dwarf everything after them on a linear axis.
    axis.set_yscale("log")
    axis.grid(True, alpha=0.25, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the collaborative filtering model.")
    parser.add_argument(
        "--mode", choices=[m.value for m in RatingMode], default=RatingMode.EXPLICIT.value
    )
    parser.add_argument("--features", type=int, default=10, dest="n_features")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-1)
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="skip mean normalisation (explicit mode only; for the ablation in the docs)",
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        mode=RatingMode(args.mode),
        n_features=args.n_features,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        lambda_=args.lambda_,
        seed=args.seed,
        normalize=not args.no_normalize,
    )

    settings = get_settings()
    with SessionLocal() as session:
        matrices = load_matrices(session, cfg.mode)

    print(f"Loaded {cfg.mode} matrices: {matrices.n_m} items x {matrices.n_u} users")
    print(f"  observed entries : {matrices.n_observed:,}  ({matrices.density:.2%} dense)")
    if matrices.n_observed == 0:
        raise SystemExit("No observations found. Run `python -m scripts.seed --reset` first.")
    print(f"Training {cfg.iterations} iterations, n={cfg.n_features}, lambda={cfg.lambda_}")

    result = train(matrices, cfg)

    directory = next_version_dir(settings.model_root)
    metadata: dict[str, Any] = {
        "model": "collaborative_filtering",
        "trained_at": datetime.now(UTC).isoformat(),
        "hyperparameters": {
            "mode": str(cfg.mode),
            "n_features": cfg.n_features,
            "iterations": cfg.iterations,
            "learning_rate": cfg.learning_rate,
            "lambda": cfg.lambda_,
            "seed": cfg.seed,
            "mean_normalized": cfg.mode is RatingMode.EXPLICIT and cfg.normalize,
            "init_scale": INIT_SCALE,
        },
        "data": {
            "n_items": matrices.n_m,
            "n_users": matrices.n_u,
            "n_observed": matrices.n_observed,
            "density": round(matrices.density, 6),
            "fingerprint": _data_fingerprint(matrices),
        },
        # Training-set fit only. Held-out metrics against baselines are M3.
        "training_metrics": {k: round(v, 6) for k, v in result.metrics.items()},
        "runtime_seconds": round(result.seconds, 2),
        "tensorflow_version": tf.__version__,
    }

    save_cf_artifacts(
        directory,
        X=result.X,
        W=result.W,
        b=result.b,
        mu=result.mu,
        item_ids=matrices.item_ids,
        user_ids=matrices.user_ids,
        metadata=metadata,
    )
    save_loss_curve(result.history, directory / LOSS_CURVE_FILE, cfg)
    (directory / "loss_history.json").write_text(json.dumps(result.history), encoding="utf-8")

    print(f"\nSaved {directory.name} to {directory}")
    print(
        f"  cost {result.history[0]:,.2f} -> {result.history[-1]:,.2f} " f"in {result.seconds:.1f}s"
    )
    for name, value in result.metrics.items():
        print(f"  {name:18s} {value:.4f}")


if __name__ == "__main__":
    main()
