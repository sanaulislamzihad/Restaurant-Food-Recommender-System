"""Feature engineering for the content-based two-tower model.

This module is the single definition of what a user vector and an item vector
contain. Training and serving both import it, because a feature pipeline that
exists twice will eventually disagree with itself - and the failure is silent:
the model keeps returning numbers, they are just about the wrong dish.

The vocabularies (cuisines, areas, ingredient tags) are learned from the
catalogue at training time and saved alongside the weights. A vocabulary rebuilt
at serving time from a changed menu would shift every column, so the saved one
is authoritative and anything unseen falls into an explicit "unknown" slot.

User and item vectors are deliberately different lengths. The towers project
both to the same 32 dimensions, which is the point of the architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Ingredient tags rarer than this are dropped from the vocabulary. A tag seen
#: twice cannot support a learned weight; it only adds a column of noise.
MIN_TAG_FREQUENCY = 5

AGE_BUCKETS = ((0, 25), (25, 35), (35, 45), (45, 200))
PRICE_BUCKETS = ((0, 120), (120, 260), (260, 420), (420, 650), (650, 10_000))
PREP_BUCKETS = ((0, 15), (15, 25), (25, 40), (40, 200))
ORDER_VALUE_BUCKETS = ((0, 250), (250, 500), (500, 900), (900, 100_000))

GENDERS = ("male", "female", "other")

#: A dish is treated as vegan when it is vegetarian and carries none of these.
#: The schema has no vegan flag - the brief asks for one, so it is derived here
#: rather than invented as a column nobody fills in.
ANIMAL_DERIVED_TAGS = frozenset(
    {"milk", "yogurt", "egg", "cheese", "ghee", "butter", "cream", "honey"}
)


def _one_hot(index: int | None, size: int) -> list[float]:
    """One-hot with an extra trailing slot for "not one of the known values"."""
    vector = [0.0] * (size + 1)
    vector[index if index is not None else size] = 1.0
    return vector


def _bucket_index(value: float, buckets: tuple[tuple[int, int], ...]) -> int:
    for index, (low, high) in enumerate(buckets):
        if low <= value < high:
            return index
    return len(buckets) - 1


def _log_scale(value: float) -> float:
    """Compress a count into a small range.

    Order counts and rating counts are heavy-tailed; feeding a raw 400 next to
    features in [0, 1] lets one column dominate the first layer purely by
    magnitude.
    """
    return math.log1p(max(value, 0.0)) / 10.0


@dataclass
class FeatureVocabulary:
    """Everything needed to turn rows into vectors, learned once at training."""

    cuisines: list[str]
    areas: list[str]
    tags: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {"cuisines": self.cuisines, "areas": self.areas, "tags": self.tags}

    @classmethod
    def from_dict(cls, payload: dict[str, list[str]]) -> FeatureVocabulary:
        return cls(
            cuisines=list(payload["cuisines"]),
            areas=list(payload["areas"]),
            tags=list(payload["tags"]),
        )

    @classmethod
    def build(cls, items: list[dict[str, Any]], users: list[dict[str, Any]]) -> FeatureVocabulary:
        cuisines = sorted({str(item["cuisine"]) for item in items})
        areas = sorted({str(user["area"]) for user in users if user.get("area")})

        counts: dict[str, int] = {}
        for item in items:
            for tag in item.get("ingredient_tags") or []:
                counts[str(tag)] = counts.get(str(tag), 0) + 1
        tags = sorted(tag for tag, count in counts.items() if count >= MIN_TAG_FREQUENCY)

        return cls(cuisines=cuisines, areas=areas, tags=tags)


@dataclass
class ItemStats:
    """Rating aggregates per item, computed from the training split only."""

    average_rating: dict[int, float] = field(default_factory=dict)
    rating_count: dict[int, int] = field(default_factory=dict)
    global_mean: float = 3.5


@dataclass
class UserStats:
    """Behaviour aggregates per user, computed from the training split only."""

    total_orders: dict[int, int] = field(default_factory=dict)
    average_rating: dict[int, float] = field(default_factory=dict)
    average_order_value: dict[int, float] = field(default_factory=dict)
    #: (user_id, cuisine) -> mean rating that user gives that cuisine.
    cuisine_rating: dict[tuple[int, str], float] = field(default_factory=dict)
    global_mean: float = 3.5


class FeatureBuilder:
    """Turns database rows into fixed-width float vectors."""

    def __init__(self, vocabulary: FeatureVocabulary) -> None:
        self.vocabulary = vocabulary
        self._cuisine_index = {name: i for i, name in enumerate(vocabulary.cuisines)}
        self._area_index = {name: i for i, name in enumerate(vocabulary.areas)}
        self._tag_index = {name: i for i, name in enumerate(vocabulary.tags)}

    # ---- Items ----------------------------------------------------------

    @property
    def item_columns(self) -> list[str]:
        names = [f"cuisine={c}" for c in self.vocabulary.cuisines] + ["cuisine=unknown"]
        names += ["spice_level", "is_veg", "is_vegan", "is_non_veg", "is_rice_based"]
        names += [f"price_bucket={i}" for i in range(len(PRICE_BUCKETS))]
        names += ["average_rating", "rating_count", "has_no_ratings"]
        names += [f"prep_bucket={i}" for i in range(len(PREP_BUCKETS))]
        names += [f"tag={t}" for t in self.vocabulary.tags]
        return names

    def item_vector(self, item: dict[str, Any], stats: ItemStats) -> list[float]:
        """Features for one dish.

        Everything here is knowable the moment a dish is added to the menu
        except the two rating aggregates, which is precisely why this model can
        rank a brand-new item that collaborative filtering cannot place at all.
        """
        tags = {str(tag) for tag in (item.get("ingredient_tags") or [])}
        item_id = int(item["id"])
        count = stats.rating_count.get(item_id, 0)
        is_veg = bool(item["is_veg"])

        vector: list[float] = []
        vector += _one_hot(
            self._cuisine_index.get(str(item["cuisine"])), len(self.vocabulary.cuisines)
        )
        vector.append(float(item["spice_level"]) / 5.0)
        vector.append(1.0 if is_veg else 0.0)
        vector.append(1.0 if is_veg and not (tags & ANIMAL_DERIVED_TAGS) else 0.0)
        vector.append(0.0 if is_veg else 1.0)
        vector.append(1.0 if item["is_rice_based"] else 0.0)

        price_slot = [0.0] * len(PRICE_BUCKETS)
        price_slot[_bucket_index(float(item["price"]), PRICE_BUCKETS)] = 1.0
        vector += price_slot

        # An unrated dish gets the global mean plus an explicit flag, so the
        # network can tell "average" apart from "nobody has said yet".
        vector.append(stats.average_rating.get(item_id, stats.global_mean) / 5.0)
        vector.append(_log_scale(count))
        vector.append(1.0 if count == 0 else 0.0)

        prep_slot = [0.0] * len(PREP_BUCKETS)
        prep_slot[_bucket_index(float(item["prep_time_min"]), PREP_BUCKETS)] = 1.0
        vector += prep_slot

        tag_slot = [0.0] * len(self.vocabulary.tags)
        for tag in tags:
            position = self._tag_index.get(tag)
            if position is not None:
                tag_slot[position] = 1.0
        vector += tag_slot

        return vector

    # ---- Users ----------------------------------------------------------

    @property
    def user_columns(self) -> list[str]:
        names = [f"age_bucket={i}" for i in range(len(AGE_BUCKETS))] + ["age=unknown"]
        names += [f"gender={g}" for g in GENDERS] + ["gender=unknown"]
        names += [f"area={a}" for a in self.vocabulary.areas] + ["area=unknown"]
        names += ["total_orders", "average_rating_given", "has_no_history"]
        names += [f"order_value_bucket={i}" for i in range(len(ORDER_VALUE_BUCKETS))]
        names += ["spice_tolerance"]
        names += [f"cuisine_rating={c}" for c in self.vocabulary.cuisines]
        return names

    def user_vector(self, user: dict[str, Any], stats: UserStats) -> list[float]:
        """Features for one customer.

        A brand-new user still has age, gender, area and stated spice tolerance
        from registration, so this vector is never empty. That is what lets the
        content model say something sensible about somebody who has ordered
        nothing, which is the entire cold-start case.
        """
        user_id = int(user["id"])
        orders = stats.total_orders.get(user_id, 0)

        vector: list[float] = []
        age = user.get("age")
        vector += (
            _one_hot(_bucket_index(float(age), AGE_BUCKETS), len(AGE_BUCKETS))
            if age is not None
            else _one_hot(None, len(AGE_BUCKETS))
        )

        gender = str(user["gender"]) if user.get("gender") else None
        vector += _one_hot(GENDERS.index(gender) if gender in GENDERS else None, len(GENDERS))
        vector += _one_hot(
            self._area_index.get(str(user["area"])) if user.get("area") else None,
            len(self.vocabulary.areas),
        )

        vector.append(_log_scale(orders))
        vector.append(stats.average_rating.get(user_id, stats.global_mean) / 5.0)
        vector.append(1.0 if orders == 0 else 0.0)

        value_slot = [0.0] * len(ORDER_VALUE_BUCKETS)
        value_slot[
            _bucket_index(stats.average_order_value.get(user_id, 0.0), ORDER_VALUE_BUCKETS)
        ] = 1.0
        vector += value_slot

        vector.append(float(user.get("spice_tolerance", 2)) / 5.0)

        for cuisine in self.vocabulary.cuisines:
            # Falls back to the user's own average, then the global mean, so an
            # unseen cuisine reads as "no opinion" rather than as a bad review.
            fallback = stats.average_rating.get(user_id, stats.global_mean)
            vector.append(stats.cuisine_rating.get((user_id, cuisine), fallback) / 5.0)

        return vector

    # ---- Matrices -------------------------------------------------------

    def item_matrix(self, items: list[dict[str, Any]], stats: ItemStats) -> np.ndarray:
        """(n_items, n_item_features)."""
        return np.array([self.item_vector(item, stats) for item in items], dtype=np.float32)

    def user_matrix(self, users: list[dict[str, Any]], stats: UserStats) -> np.ndarray:
        """(n_users, n_user_features)."""
        return np.array([self.user_vector(user, stats) for user in users], dtype=np.float32)
