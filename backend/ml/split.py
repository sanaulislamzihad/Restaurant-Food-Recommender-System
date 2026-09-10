"""Chronological train/validation/test splitting.

Random splitting leaks the future. If a user's ratings are scattered at random
across the splits, the model can be trained on what they thought in March and
scored on what they thought in February - which no deployed system ever gets to
do, and which flatters every metric.

Each user's ratings are therefore ordered by time and cut by position: the
earliest go to train, the next block to validation, the most recent to test.
Hyperparameters are chosen on validation and reported on test, so no number in
the final table comes from data the choice was made on.

One honest caveat about per-user splitting. It is the standard protocol and it
is what the brief asks for, but it is not a strict simulation of deployment:
user A's training rating may be more recent than user B's test rating, so a
little cross-user future does bleed in. The alternative - one global timestamp
cutoff - avoids that but leaves every recently-joined user with no training
history at all, which throws away most of the cold-start population that the
system specifically has to handle. Per-user is the lesser distortion here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import FoodItem, Rating, User
from ml.cofi import RatingMode
from ml.data import RatingMatrices


@dataclass(frozen=True)
class RatingEvent:
    """One rating, with the timestamp the split depends on."""

    user_id: int
    item_id: int
    rating: float
    created_at: datetime


@dataclass(frozen=True)
class Split:
    """A chronological three-way split, plus the axes shared by all of them."""

    train: list[RatingEvent]
    validation: list[RatingEvent]
    test: list[RatingEvent]
    item_ids: list[int]
    user_ids: list[int]

    def summary(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
            "users_with_test_ratings": len({event.user_id for event in self.test}),
            "items_in_test": len({event.item_id for event in self.test}),
        }


def load_rating_events(session: Session) -> tuple[list[RatingEvent], list[int], list[int]]:
    """Load every rating as a timestamped event, plus the item and user axes."""
    item_ids = [row[0] for row in session.execute(select(FoodItem.id).order_by(FoodItem.id)).all()]
    # Staff accounts are not customers; see the note in ml/data.py.
    user_ids = [
        row[0]
        for row in session.execute(
            select(User.id).where(User.is_admin.is_(False)).order_by(User.id)
        ).all()
    ]

    rows = session.execute(
        select(Rating.user_id, Rating.food_item_id, Rating.rating, Rating.created_at)
    ).all()
    events = [
        RatingEvent(user_id=u, item_id=i, rating=float(r), created_at=t) for u, i, r, t in rows
    ]
    return events, item_ids, user_ids


def chronological_split(
    events: list[RatingEvent],
    item_ids: list[int],
    user_ids: list[int],
    *,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    min_ratings_to_split: int = 5,
) -> Split:
    """Split each user's history by time.

    Args:
        events: every rating, in any order.
        item_ids: the full item axis.
        user_ids: the full user axis.
        test_fraction: share of each user's most recent ratings held out for test.
        validation_fraction: share taken from just before the test block.
        min_ratings_to_split: users with fewer ratings than this contribute
            everything to train. Holding out one of a user's three ratings
            produces a metric dominated by noise, and those users are exactly the
            cold-start population that the content model - not this one - has to
            serve.

    Returns:
        A :class:`Split`. Every user and item stays on the axes even if they end
        up with no training data, because the cold cases must still be scoreable.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test_fraction + validation_fraction must leave room for training data")

    by_user: dict[int, list[RatingEvent]] = {}
    for event in events:
        by_user.setdefault(event.user_id, []).append(event)

    train: list[RatingEvent] = []
    validation: list[RatingEvent] = []
    test: list[RatingEvent] = []

    for user_events in by_user.values():
        # Ties on the timestamp are broken by item id so the split is stable
        # across runs; otherwise two rows sharing a timestamp could swap sides
        # and move the metrics slightly for no reason.
        ordered = sorted(user_events, key=lambda e: (e.created_at, e.item_id))

        if len(ordered) < min_ratings_to_split:
            train.extend(ordered)
            continue

        n = len(ordered)
        n_test = max(1, int(round(n * test_fraction)))
        n_validation = int(round(n * validation_fraction)) if validation_fraction else 0
        # Never let the held-out blocks consume the whole history.
        n_validation = min(n_validation, max(0, n - n_test - 1))

        cut_test = n - n_test
        cut_validation = cut_test - n_validation

        train.extend(ordered[:cut_validation])
        validation.extend(ordered[cut_validation:cut_test])
        test.extend(ordered[cut_test:])

    return Split(
        train=train,
        validation=validation,
        test=test,
        item_ids=item_ids,
        user_ids=user_ids,
    )


def matrices_from_events(
    events: list[RatingEvent], item_ids: list[int], user_ids: list[int]
) -> RatingMatrices:
    """Build (n_m, n_u) Y and R from a list of events.

    The axes are passed in rather than derived from the events, so that a split
    containing no ratings for some item still produces a matrix of the full
    catalogue shape. Otherwise train and test matrices would have different
    dimensions and could not be compared.
    """
    item_pos = {item_id: idx for idx, item_id in enumerate(item_ids)}
    user_pos = {user_id: idx for idx, user_id in enumerate(user_ids)}

    Y = np.zeros((len(item_ids), len(user_ids)), dtype=np.float64)
    R = np.zeros_like(Y)
    for event in events:
        i, j = item_pos[event.item_id], user_pos[event.user_id]
        Y[i, j] = event.rating
        R[i, j] = 1.0

    return RatingMatrices(Y=Y, R=R, item_ids=item_ids, user_ids=user_ids, mode=RatingMode.EXPLICIT)
