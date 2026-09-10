"""Serving-side inference for the two-tower model, without TensorFlow.

The user tower is three dense layers. Running it is a handful of matrix
multiplies, so the API loads the weights as numpy and does the forward pass
itself rather than pulling a ~600MB framework into the serving image for
arithmetic numpy already does well.

Item embeddings are not computed here at all. They are precomputed by the
training job and loaded straight from disk, because an item's features change
only when the menu does - recomputing 300 of them on every request would be
work done over and over for an answer that never differs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.features import FeatureBuilder, FeatureVocabulary, ItemStats, UserStats
from ml.two_tower import RATING_HALF_RANGE, RATING_MIDPOINT

CONTENT_WEIGHTS_FILE = "content_model.npz"
CONTENT_META_FILE = "content_meta.json"


def _relu(x: np.ndarray) -> np.ndarray:
    activated: np.ndarray = np.maximum(x, 0.0)
    return activated


def _l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    # A zero vector would divide by zero; leaving it at zero means "no signal",
    # which produces a similarity of 0 rather than a NaN that poisons the sort.
    return x / np.where(norms == 0.0, 1.0, norms)


@dataclass
class ContentModel:
    """A trained two-tower model, ready to score.

    item_embeddings: (n_items, 32) L2-normalised, precomputed
    """

    user_kernels: list[np.ndarray]
    user_biases: list[np.ndarray]
    item_embeddings: np.ndarray
    item_ids: list[int]
    vocabulary: FeatureVocabulary
    item_stats: ItemStats
    user_stats: UserStats
    mode: str
    logit_scale: float
    metadata: dict[str, Any]
    version: str = ""

    def __post_init__(self) -> None:
        self._builder = FeatureBuilder(self.vocabulary)
        self._item_index = {item_id: i for i, item_id in enumerate(self.item_ids)}

    @property
    def builder(self) -> FeatureBuilder:
        return self._builder

    def item_index(self) -> dict[int, int]:
        return self._item_index

    def user_embedding(self, user: dict[str, Any]) -> np.ndarray:
        """Embed one user, shape (32,).

        This is the only tower evaluated per request, which is why the item side
        is precomputed and this side is kept cheap.
        """
        activations = np.asarray(self._builder.user_vector(user, self.user_stats), dtype=np.float32)
        last = len(self.user_kernels) - 1
        for index, (kernel, bias) in enumerate(
            zip(self.user_kernels, self.user_biases, strict=True)
        ):
            activations = activations @ kernel + bias
            # Every layer but the output is ReLU; the output stays linear so
            # embeddings can occupy the whole space and similarities can be
            # negative.
            if index < last:
                activations = _relu(activations)
        return _l2_normalize(activations)

    def score_items(self, user: dict[str, Any], item_ids: list[int]) -> dict[int, float]:
        """Predicted rating for each item, in **star units**.

        The dot product is a cosine in [-1, 1]; it is mapped back onto the 1-5
        scale here so the caller can blend it with a collaborative-filtering
        prediction without comparing two different quantities.
        """
        if not item_ids:
            return {}

        vu = self.user_embedding(user)
        rows = [self._item_index.get(item_id) for item_id in item_ids]
        known = [
            (item_id, row) for item_id, row in zip(item_ids, rows, strict=True) if row is not None
        ]
        if not known:
            return {}

        indices = np.fromiter((row for _, row in known), dtype=np.int64)
        similarities = self.item_embeddings[indices] @ vu

        if self.mode == "implicit":
            probabilities = 1.0 / (1.0 + np.exp(-self.logit_scale * similarities))
            return {item_id: float(p) for (item_id, _), p in zip(known, probabilities, strict=True)}

        ratings = similarities * RATING_HALF_RANGE + RATING_MIDPOINT
        return {
            item_id: float(np.clip(rating, 1.0, 5.0))
            for (item_id, _), rating in zip(known, ratings, strict=True)
        }

    def similar_items(self, item_id: int, limit: int = 10) -> list[tuple[int, float]]:
        """Items whose content embedding is closest to this one.

        Unlike the collaborative neighbour table, this works for a dish nobody
        has rated, because it compares descriptions rather than rating patterns.
        """
        row = self._item_index.get(item_id)
        if row is None:
            return []
        similarities = self.item_embeddings @ self.item_embeddings[row]
        similarities[row] = -np.inf  # never itself
        order = np.argsort(-similarities)[:limit]
        return [(self.item_ids[int(i)], float(similarities[int(i)])) for i in order]


def load_content_model(root: Path, version: str) -> ContentModel | None:
    """Load a trained content model, or None if this version has no content half.

    Returns None rather than raising: the content model is an enhancement, and a
    deployment that has only trained collaborative filtering should still serve
    recommendations.
    """
    directory = root / version
    weights_path = directory / CONTENT_WEIGHTS_FILE
    meta_path = directory / CONTENT_META_FILE
    if not weights_path.exists() or not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(weights_path) as data:
        depth = int(data["n_layers"])
        user_kernels = [data[f"user_kernel_{i}"] for i in range(depth)]
        user_biases = [data[f"user_bias_{i}"] for i in range(depth)]
        item_embeddings = data["item_embeddings"]

    return ContentModel(
        user_kernels=user_kernels,
        user_biases=user_biases,
        item_embeddings=item_embeddings,
        item_ids=[int(i) for i in meta["item_ids"]],
        vocabulary=FeatureVocabulary.from_dict(meta["vocabulary"]),
        item_stats=ItemStats(
            average_rating={
                int(k): float(v) for k, v in meta["item_stats"]["average_rating"].items()
            },
            rating_count={int(k): int(v) for k, v in meta["item_stats"]["rating_count"].items()},
            global_mean=float(meta["item_stats"]["global_mean"]),
        ),
        user_stats=UserStats(
            total_orders={int(k): int(v) for k, v in meta["user_stats"]["total_orders"].items()},
            average_rating={
                int(k): float(v) for k, v in meta["user_stats"]["average_rating"].items()
            },
            average_order_value={
                int(k): float(v) for k, v in meta["user_stats"]["average_order_value"].items()
            },
            cuisine_rating={
                (int(key.split("|", 1)[0]), key.split("|", 1)[1]): float(value)
                for key, value in meta["user_stats"]["cuisine_rating"].items()
            },
            global_mean=float(meta["user_stats"]["global_mean"]),
        ),
        mode=str(meta["mode"]),
        logit_scale=float(meta.get("logit_scale", 1.0)),
        metadata=meta,
        version=version,
    )
