"""The schema's integrity rules must actually be enforced, not just declared.

These run against SQLite, which honours CHECK constraints and (because
``session.py`` turns the pragma on) foreign keys. Postgres enforces the same
declarations, so a rule that holds here holds there.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import FoodItem, Impression, ItemNeighbor, Rating, Restaurant, User


def _restaurant(session: Session) -> Restaurant:
    restaurant = Restaurant(name="Test Kitchen", area="Dhanmondi", cuisine_tags=["bengali"])
    session.add(restaurant)
    session.flush()
    return restaurant


def _item(session: Session, restaurant: Restaurant, **overrides: object) -> FoodItem:
    defaults: dict[str, object] = {
        "restaurant_id": restaurant.id,
        "name": "Test Dish",
        "cuisine": "bengali",
        "spice_level": 3,
        "price": Decimal("250.00"),
        "ingredient_tags": ["rice"],
    }
    defaults.update(overrides)
    item = FoodItem(**defaults)  # type: ignore[arg-type]
    session.add(item)
    session.flush()
    return item


def _user(session: Session, email: str = "a@example.com", **overrides: object) -> User:
    defaults: dict[str, object] = {
        "name": "Test User",
        "email": email,
        "password_hash": "x",
        "spice_tolerance": 3,
    }
    defaults.update(overrides)
    user = User(**defaults)  # type: ignore[arg-type]
    session.add(user)
    session.flush()
    return user


def test_rating_outside_one_to_five_is_rejected(db_session: Session) -> None:
    restaurant = _restaurant(db_session)
    item = _item(db_session, restaurant)
    user = _user(db_session)

    db_session.add(Rating(user_id=user.id, food_item_id=item.id, rating=Decimal("7")))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_a_user_cannot_rate_the_same_item_twice(db_session: Session) -> None:
    restaurant = _restaurant(db_session)
    item = _item(db_session, restaurant)
    user = _user(db_session)

    db_session.add(Rating(user_id=user.id, food_item_id=item.id, rating=Decimal("4")))
    db_session.flush()
    db_session.add(Rating(user_id=user.id, food_item_id=item.id, rating=Decimal("5")))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_spice_tolerance_is_bounded(db_session: Session) -> None:
    db_session.add(
        User(name="Too Hot", email="hot@example.com", password_hash="x", spice_tolerance=9)
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_spice_level_is_bounded(db_session: Session) -> None:
    restaurant = _restaurant(db_session)
    with pytest.raises(IntegrityError):
        _item(db_session, restaurant, spice_level=8)


def test_price_cannot_be_negative(db_session: Session) -> None:
    restaurant = _restaurant(db_session)
    with pytest.raises(IntegrityError):
        _item(db_session, restaurant, price=Decimal("-1.00"))


def test_email_is_unique(db_session: Session) -> None:
    _user(db_session, email="dup@example.com")
    with pytest.raises(IntegrityError):
        _user(db_session, email="dup@example.com")


def test_an_item_cannot_be_its_own_neighbour(db_session: Session) -> None:
    """Guards the 'similar items' row: recommending a dish as similar to itself
    would waste a slot in every carousel it appears in."""
    restaurant = _restaurant(db_session)
    item = _item(db_session, restaurant)

    db_session.add(
        ItemNeighbor(food_item_id=item.id, neighbor_id=item.id, distance=Decimal("0"), rank=1)
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_foreign_keys_are_enforced(db_session: Session) -> None:
    """SQLite ignores foreign keys unless the pragma is set; session.py sets it.
    If this test fails, referential bugs would pass locally and fail on Postgres."""
    db_session.add(Rating(user_id=999_999, food_item_id=999_999, rating=Decimal("3")))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_timestamps_are_timezone_aware_after_a_real_round_trip(db_session: Session) -> None:
    """Read the value back out of the database rather than off the in-memory
    object.

    The earlier version of this test asserted on the attribute straight after
    commit, which (with expire_on_commit=False) is still the Python value that
    went in - so it passed while SQLite was quietly returning naive datetimes
    and the API was serialising timestamps with no offset at all.
    """
    user = _user(db_session, email="tz@example.com")
    db_session.commit()
    db_session.expire_all()

    reloaded = db_session.get(User, user.id)
    assert reloaded is not None
    assert reloaded.created_at.tzinfo is not None
    assert reloaded.created_at.utcoffset() == timedelta(0)
    assert reloaded.created_at <= datetime.now(UTC)


def test_a_naive_timestamp_is_stored_as_utc(db_session: Session) -> None:
    """Client-supplied timestamps may arrive without an offset; they are assumed
    to be UTC rather than rejected, because the impressions endpoint is
    fire-and-forget telemetry."""
    restaurant = _restaurant(db_session)
    item = _item(db_session, restaurant)
    user = _user(db_session, email="naive@example.com")

    db_session.add(
        Impression(
            user_id=user.id,
            food_item_id=item.id,
            was_ordered=True,
            shown_at=datetime(2026, 3, 1, 12, 0, 0),  # noqa: DTZ001 - deliberately naive
        )
    )
    db_session.commit()
    db_session.expire_all()

    stored = db_session.query(Impression).one()
    assert stored.shown_at == datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
