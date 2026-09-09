"""Build the (n_m, n_u) rating matrices the collaborative filter trains on.

Both feedback modes produce the same shape, so the training loop does not care
which one it was handed:

* explicit - 1-5 stars from the ``ratings`` table, ``R`` marks a rating exists
* implicit - 1/0 from the ``impressions`` table, ``R`` marks the item was shown

Every user and every item gets a row or column, including those with no
interactions at all. Dropping them would be convenient and wrong: a cold user's
all-zero column is exactly what the cold-start path has to handle, and dropping
cold items would quietly remove them from the catalogue the model can score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import Integer, func, select
from sqlalchemy.orm import Session

from app.db.models import FoodItem, Impression, Rating, User
from ml.cofi import RatingMode


@dataclass(frozen=True)
class RatingMatrices:
    """Training data plus the mapping back to database ids.

    Y: (n_m, n_u) ratings or binary labels, 0 where unobserved
    R: (n_m, n_u) 1 where observed
    """

    Y: np.ndarray
    R: np.ndarray
    item_ids: list[int]
    user_ids: list[int]
    mode: RatingMode

    @property
    def n_m(self) -> int:
        return len(self.item_ids)

    @property
    def n_u(self) -> int:
        return len(self.user_ids)

    @property
    def n_observed(self) -> int:
        return int(self.R.sum())

    @property
    def density(self) -> float:
        total = self.n_m * self.n_u
        return self.n_observed / total if total else 0.0

    def item_index(self) -> dict[int, int]:
        """food_item_id -> row index."""
        return {item_id: idx for idx, item_id in enumerate(self.item_ids)}

    def user_index(self) -> dict[int, int]:
        """user_id -> column index."""
        return {user_id: idx for idx, user_id in enumerate(self.user_ids)}


def _axis_ids(session: Session) -> tuple[list[int], list[int]]:
    item_ids = [row[0] for row in session.execute(select(FoodItem.id).order_by(FoodItem.id)).all()]
    user_ids = [row[0] for row in session.execute(select(User.id).order_by(User.id)).all()]
    return item_ids, user_ids


def load_explicit_matrices(session: Session) -> RatingMatrices:
    """Y holds 1-5 star ratings; R marks which pairs were rated."""
    item_ids, user_ids = _axis_ids(session)
    item_pos = {item_id: idx for idx, item_id in enumerate(item_ids)}
    user_pos = {user_id: idx for idx, user_id in enumerate(user_ids)}

    Y = np.zeros((len(item_ids), len(user_ids)), dtype=np.float64)
    R = np.zeros_like(Y)

    rows = session.execute(select(Rating.user_id, Rating.food_item_id, Rating.rating)).all()
    for user_id, item_id, value in rows:
        i, j = item_pos[item_id], user_pos[user_id]
        Y[i, j] = float(value)
        R[i, j] = 1.0

    return RatingMatrices(Y=Y, R=R, item_ids=item_ids, user_ids=user_ids, mode=RatingMode.EXPLICIT)


def load_implicit_matrices(session: Session) -> RatingMatrices:
    """Y holds 1 if the pair ever converted, 0 if shown and never ordered.

    A user may be shown the same dish repeatedly. The pair is collapsed to a
    single label - ordered at least once counts as a positive - rather than
    counting impressions, because repeat exposure says more about how the feed
    was ranked than about whether the customer likes the dish.
    """
    item_ids, user_ids = _axis_ids(session)
    item_pos = {item_id: idx for idx, item_id in enumerate(item_ids)}
    user_pos = {user_id: idx for idx, user_id in enumerate(user_ids)}

    Y = np.zeros((len(item_ids), len(user_ids)), dtype=np.float64)
    R = np.zeros_like(Y)

    rows = session.execute(
        select(
            Impression.user_id,
            Impression.food_item_id,
            # cast: SQLite has no boolean, so MAX() over it needs an integer.
            func.max(func.cast(Impression.was_ordered, Integer)),
        ).group_by(Impression.user_id, Impression.food_item_id)
    ).all()
    for user_id, item_id, ordered in rows:
        i, j = item_pos[item_id], user_pos[user_id]
        Y[i, j] = 1.0 if ordered else 0.0
        R[i, j] = 1.0

    return RatingMatrices(Y=Y, R=R, item_ids=item_ids, user_ids=user_ids, mode=RatingMode.IMPLICIT)


def load_matrices(session: Session, mode: RatingMode) -> RatingMatrices:
    """Dispatch on the configured feedback mode."""
    if mode is RatingMode.EXPLICIT:
        return load_explicit_matrices(session)
    return load_implicit_matrices(session)
