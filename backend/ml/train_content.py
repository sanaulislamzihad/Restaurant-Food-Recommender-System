"""Offline training job for the content-based two-tower model.

    python -m ml.train_content
    python -m ml.train_content --epochs 40 --mode implicit

Writes a new versioned directory containing the user tower as numpy weights,
the precomputed item embeddings, the feature vocabulary and the aggregate
statistics the feature builder needs. The API loads all of that without
TensorFlow.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import FoodItem, Order, Rating, User
from app.db.session import SessionLocal
from ml.artifacts import next_version_dir
from ml.cofi import RatingMode
from ml.content_inference import CONTENT_META_FILE, CONTENT_WEIGHTS_FILE
from ml.features import FeatureBuilder, FeatureVocabulary, ItemStats, UserStats
from ml.two_tower import (
    TowerWeights,
    build_two_tower,
    extract_tower,
    scale_ratings,
)


@dataclass
class ContentTrainConfig:
    mode: RatingMode = RatingMode.EXPLICIT
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    validation_split: float = 0.1
    seed: int = 42
    patience: int = 5


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_catalogue(session: Session) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Items and users as plain dicts, so the feature code never touches the ORM."""
    items = [
        {
            "id": row.id,
            "cuisine": row.cuisine,
            "spice_level": row.spice_level,
            "is_veg": row.is_veg,
            "is_rice_based": row.is_rice_based,
            "price": float(row.price),
            "prep_time_min": row.prep_time_min,
            "ingredient_tags": row.ingredient_tags or [],
        }
        for row in session.execute(select(FoodItem).order_by(FoodItem.id)).scalars()
    ]
    users = [
        {
            "id": row.id,
            "age": row.age,
            "gender": str(row.gender) if row.gender else None,
            "area": row.area,
            "spice_tolerance": row.spice_tolerance,
        }
        for row in session.execute(select(User).order_by(User.id)).scalars()
    ]
    return items, users


def compute_item_stats(session: Session) -> ItemStats:
    rows = session.execute(
        select(Rating.food_item_id, func.count(), func.avg(Rating.rating)).group_by(
            Rating.food_item_id
        )
    ).all()
    overall = session.scalar(select(func.avg(Rating.rating)))
    return ItemStats(
        average_rating={int(i): float(a) for i, _, a in rows},
        rating_count={int(i): int(c) for i, c, _ in rows},
        global_mean=float(overall) if overall is not None else 3.5,
    )


def compute_user_stats(session: Session) -> UserStats:
    order_rows = session.execute(
        select(Order.user_id, func.count(), func.avg(Order.total_amount)).group_by(Order.user_id)
    ).all()
    rating_rows = session.execute(
        select(Rating.user_id, func.avg(Rating.rating)).group_by(Rating.user_id)
    ).all()
    cuisine_rows = session.execute(
        select(Rating.user_id, FoodItem.cuisine, func.avg(Rating.rating))
        .join(FoodItem, FoodItem.id == Rating.food_item_id)
        .group_by(Rating.user_id, FoodItem.cuisine)
    ).all()
    overall = session.scalar(select(func.avg(Rating.rating)))

    return UserStats(
        total_orders={int(u): int(c) for u, c, _ in order_rows},
        average_order_value={int(u): float(v) for u, _, v in order_rows},
        average_rating={int(u): float(a) for u, a in rating_rows},
        cuisine_rating={(int(u), str(c)): float(a) for u, c, a in cuisine_rows},
        global_mean=float(overall) if overall is not None else 3.5,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_content_model(
    cfg: ContentTrainConfig,
    user_matrix: np.ndarray,
    item_matrix: np.ndarray,
    user_index: dict[int, int],
    item_index: dict[int, int],
    interactions: list[tuple[int, int, float]],
) -> tuple[tf.keras.Model, dict[str, float]]:
    """Fit the two towers on (user, item, target) triples."""
    tf.keras.utils.set_random_seed(cfg.seed)

    rows = np.fromiter((user_index[u] for u, _, _ in interactions), dtype=np.int64)
    cols = np.fromiter((item_index[i] for _, i, _ in interactions), dtype=np.int64)
    raw_targets = np.fromiter((v for _, _, v in interactions), dtype=np.float32)

    # Ratings are mapped onto the cosine's own range; see the note in two_tower.
    targets = (
        raw_targets
        if cfg.mode is RatingMode.IMPLICIT
        else scale_ratings(raw_targets).astype(np.float32)
    )

    x_users = user_matrix[rows]
    x_items = item_matrix[cols]

    model = build_two_tower(user_matrix.shape[1], item_matrix.shape[1], cfg.mode)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="binary_crossentropy" if cfg.mode is RatingMode.IMPLICIT else "mse",
    )

    history = model.fit(
        [x_users, x_items],
        targets,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_split,
        shuffle=True,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.patience,
                restore_best_weights=True,
            )
        ],
    )

    predictions = model.predict([x_users, x_items], batch_size=1024, verbose=0).ravel()
    metrics: dict[str, float] = {
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "epochs_run": float(len(history.history["loss"])),
    }
    if cfg.mode is RatingMode.EXPLICIT:
        stars = predictions * 2.0 + 3.0
        errors = stars - raw_targets
        metrics["train_rmse"] = float(np.sqrt(np.mean(errors**2)))
        metrics["train_mae"] = float(np.mean(np.abs(errors)))
    else:
        metrics["train_accuracy"] = float(np.mean((predictions >= 0.5) == (raw_targets >= 0.5)))

    return model, metrics


def save_content_artifacts(
    directory: Path,
    *,
    model: tf.keras.Model,
    builder: FeatureBuilder,
    item_matrix: np.ndarray,
    item_ids: list[int],
    item_stats: ItemStats,
    user_stats: UserStats,
    cfg: ContentTrainConfig,
    metrics: dict[str, float],
    n_interactions: int,
) -> Path:
    """Write the serving artifacts: numpy weights plus everything to rebuild features."""
    directory.mkdir(parents=True, exist_ok=True)

    user_weights = TowerWeights.from_keras(extract_tower(model, "user_NN"))
    item_tower = extract_tower(model, "item_NN")
    item_weights = TowerWeights.from_keras(item_tower)

    # Item embeddings are computed once, here, and never at request time.
    raw = item_tower.predict(item_matrix, batch_size=512, verbose=0)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    item_embeddings = (raw / np.where(norms == 0.0, 1.0, norms)).astype(np.float32)

    payload: dict[str, np.ndarray] = {
        "n_layers": np.array(len(user_weights.kernels)),
        "item_embeddings": item_embeddings,
    }
    for index, (kernel, bias) in enumerate(
        zip(user_weights.kernels, user_weights.biases, strict=True)
    ):
        payload[f"user_kernel_{index}"] = kernel
        payload[f"user_bias_{index}"] = bias
    # The item tower is exported as well, not only its precomputed outputs. A
    # dish added to the menu after training has no precomputed embedding, and
    # without the weights the model could not score the very case - a brand-new
    # item - that it exists to handle.
    for index, (kernel, bias) in enumerate(
        zip(item_weights.kernels, item_weights.biases, strict=True)
    ):
        payload[f"item_kernel_{index}"] = kernel
        payload[f"item_bias_{index}"] = bias
    np.savez_compressed(directory / CONTENT_WEIGHTS_FILE, **payload)  # type: ignore[arg-type]

    logit_scale = 1.0
    if cfg.mode is RatingMode.IMPLICIT:
        logit_scale = float(model.get_layer("conversion_probability").scale.numpy())

    meta: dict[str, Any] = {
        "model": "content_two_tower",
        "trained_at": datetime.now(UTC).isoformat(),
        "mode": str(cfg.mode),
        "logit_scale": logit_scale,
        "embedding_dim": int(item_embeddings.shape[1]),
        "item_ids": item_ids,
        "vocabulary": builder.vocabulary.to_dict(),
        "n_user_features": len(builder.user_columns),
        "n_item_features": len(builder.item_columns),
        "hyperparameters": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "seed": cfg.seed,
            "patience": cfg.patience,
        },
        "data": {"n_interactions": n_interactions, "n_items": len(item_ids)},
        "training_metrics": {k: round(v, 6) for k, v in metrics.items()},
        "tensorflow_version": tf.__version__,
        "item_stats": {
            "average_rating": {str(k): v for k, v in item_stats.average_rating.items()},
            "rating_count": {str(k): v for k, v in item_stats.rating_count.items()},
            "global_mean": item_stats.global_mean,
        },
        "user_stats": {
            "total_orders": {str(k): v for k, v in user_stats.total_orders.items()},
            "average_rating": {str(k): v for k, v in user_stats.average_rating.items()},
            "average_order_value": {str(k): v for k, v in user_stats.average_order_value.items()},
            # Tuple keys are not representable in JSON.
            "cuisine_rating": {
                f"{user_id}|{cuisine}": value
                for (user_id, cuisine), value in user_stats.cuisine_rating.items()
            },
            "global_mean": user_stats.global_mean,
        },
    }
    (directory / CONTENT_META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the content-based two-tower model.")
    parser.add_argument(
        "--mode", choices=[m.value for m in RatingMode], default=RatingMode.EXPLICIT.value
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ContentTrainConfig(
        mode=RatingMode(args.mode),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    with SessionLocal() as session:
        items, users = load_catalogue(session)
        item_stats = compute_item_stats(session)
        user_stats = compute_user_stats(session)
        interactions = [
            (int(u), int(i), float(v))
            for u, i, v in session.execute(
                select(Rating.user_id, Rating.food_item_id, Rating.rating)
            ).all()
        ]

    if not interactions:
        raise SystemExit("No ratings found. Run `python -m scripts.seed --reset` first.")

    vocabulary = FeatureVocabulary.build(items, users)
    builder = FeatureBuilder(vocabulary)

    item_matrix = builder.item_matrix(items, item_stats)
    user_matrix = builder.user_matrix(users, user_stats)
    item_ids = [int(item["id"]) for item in items]
    user_ids = [int(user["id"]) for user in users]
    item_index = {item_id: i for i, item_id in enumerate(item_ids)}
    user_index = {user_id: i for i, user_id in enumerate(user_ids)}

    print(f"Feature vectors: user {user_matrix.shape[1]}, item {item_matrix.shape[1]}")
    print(
        f"  vocabulary: {len(vocabulary.cuisines)} cuisines, "
        f"{len(vocabulary.areas)} areas, {len(vocabulary.tags)} ingredient tags"
    )
    print(f"  interactions: {len(interactions):,}")

    started = time.perf_counter()
    model, metrics = train_content_model(
        cfg, user_matrix, item_matrix, user_index, item_index, interactions
    )
    seconds = time.perf_counter() - started

    settings = get_settings()
    directory = next_version_dir(settings.model_root)
    metrics["runtime_seconds"] = round(seconds, 2)
    save_content_artifacts(
        directory,
        model=model,
        builder=builder,
        item_matrix=item_matrix,
        item_ids=item_ids,
        item_stats=item_stats,
        user_stats=user_stats,
        cfg=cfg,
        metrics=metrics,
        n_interactions=len(interactions),
    )

    print(f"\nSaved {directory.name} to {directory}")
    for name, value in metrics.items():
        print(f"  {name:18s} {value:.4f}")


if __name__ == "__main__":
    main()
