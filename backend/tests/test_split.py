"""Chronological splitting.

The tests that matter here are the leakage tests. A split that quietly puts a
user's later ratings into training and earlier ones into test would inflate
every number in the evaluation table, and would do it silently.
"""

from datetime import UTC, datetime, timedelta

import pytest

from ml.split import RatingEvent, chronological_split, matrices_from_events

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _events(user_id: int, count: int, start_day: int = 0) -> list[RatingEvent]:
    return [
        RatingEvent(
            user_id=user_id,
            item_id=(i % 20) + 1,
            rating=float((i % 5) + 1),
            created_at=BASE + timedelta(days=start_day + i),
        )
        for i in range(count)
    ]


def test_test_ratings_are_always_later_than_training_ratings() -> None:
    """The property the whole protocol rests on."""
    events = _events(1, 20) + _events(2, 15)
    split = chronological_split(events, list(range(1, 21)), [1, 2])

    for user_id in (1, 2):
        latest_train = max(e.created_at for e in split.train if e.user_id == user_id)
        earliest_validation = min(e.created_at for e in split.validation if e.user_id == user_id)
        latest_validation = max(e.created_at for e in split.validation if e.user_id == user_id)
        earliest_test = min(e.created_at for e in split.test if e.user_id == user_id)

        assert latest_train < earliest_validation
        assert latest_validation < earliest_test


def test_no_rating_appears_in_more_than_one_split() -> None:
    events = _events(1, 30)
    split = chronological_split(events, list(range(1, 21)), [1])

    def keys(block: list[RatingEvent]) -> set[tuple[int, int, str]]:
        return {(e.user_id, e.item_id, e.created_at.isoformat()) for e in block}

    train, validation, test = keys(split.train), keys(split.validation), keys(split.test)
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert len(train) + len(validation) + len(test) == len(events)


def test_users_below_the_minimum_contribute_everything_to_training() -> None:
    """Holding out one of a user's three ratings measures noise, and those users
    are the content model's problem rather than this one's."""
    events = _events(1, 3) + _events(2, 20)
    split = chronological_split(events, list(range(1, 21)), [1, 2], min_ratings_to_split=5)

    assert all(e.user_id == 2 for e in split.test)
    assert len([e for e in split.train if e.user_id == 1]) == 3


def test_every_user_with_enough_history_gets_at_least_one_test_rating() -> None:
    events = _events(1, 5) + _events(2, 6) + _events(3, 40)
    split = chronological_split(events, list(range(1, 21)), [1, 2, 3], min_ratings_to_split=5)

    users_in_test = {e.user_id for e in split.test}
    assert users_in_test == {1, 2, 3}


def test_split_sizes_roughly_track_the_requested_fractions() -> None:
    events = _events(1, 100)
    split = chronological_split(
        events, list(range(1, 21)), [1], test_fraction=0.2, validation_fraction=0.2
    )
    assert len(split.test) == 20
    assert len(split.validation) == 20
    assert len(split.train) == 60


def test_the_split_is_deterministic() -> None:
    events = _events(1, 25) + _events(2, 25)
    first = chronological_split(events, list(range(1, 21)), [1, 2])
    second = chronological_split(list(reversed(events)), list(range(1, 21)), [1, 2])
    assert [e.created_at for e in first.test] == [e.created_at for e in second.test]


def test_training_data_is_never_emptied_by_the_holdout_blocks() -> None:
    events = _events(1, 5)
    split = chronological_split(
        events, list(range(1, 21)), [1], test_fraction=0.4, validation_fraction=0.4
    )
    assert len([e for e in split.train if e.user_id == 1]) >= 1


@pytest.mark.parametrize(
    ("test_fraction", "validation_fraction"),
    [(0.0, 0.2), (1.0, 0.0), (0.6, 0.5), (0.2, -0.1)],
)
def test_invalid_fractions_are_rejected(test_fraction: float, validation_fraction: float) -> None:
    with pytest.raises(ValueError):
        chronological_split(
            _events(1, 10),
            list(range(1, 21)),
            [1],
            test_fraction=test_fraction,
            validation_fraction=validation_fraction,
        )


def test_matrices_keep_the_full_catalogue_shape() -> None:
    """A split missing some items must still produce a full-size matrix, or the
    train and test matrices could not be compared position by position."""
    events = [RatingEvent(1, 3, 5.0, BASE)]
    matrices = matrices_from_events(events, [1, 2, 3, 4, 5], [1, 2])

    assert matrices.Y.shape == (5, 2)
    assert matrices.R.sum() == 1
    assert matrices.Y[2, 0] == 5.0
