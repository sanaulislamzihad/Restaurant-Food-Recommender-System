"""The seed generator has to produce data the later milestones can rely on.

These use a small seed (60 users) so the suite stays fast; the properties being
checked do not depend on scale.
"""

from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.db.models import FoodItem, Impression, Order, OrderItem, Rating, Restaurant, User
from app.db.session import SessionLocal
from scripts.menu_data import DISHES, RESTAURANTS, SIDE_CUISINES, TARGET_ITEM_COUNT
from scripts.seed import SeedConfig, seed


@pytest.fixture
def small_seed(empty_database: None, tmp_meta_path: Path) -> dict[str, object]:
    cfg = SeedConfig(
        users=60,
        ratings=1_200,
        negatives_per_positive=2,
        random_seed=7,
        reset=True,
        meta_path=tmp_meta_path,
    )
    return seed(cfg)


def test_menu_pools_can_fill_every_restaurant() -> None:
    """A restaurant asking for more dishes than its cuisines offer would make
    the seeder sample without replacement from too small a pool and crash."""
    by_cuisine: dict[str, int] = {}
    for dish in DISHES:
        by_cuisine[dish.cuisine] = by_cuisine.get(dish.cuisine, 0) + 1
    side_pool = sum(by_cuisine[c] for c in SIDE_CUISINES)

    for spec in RESTAURANTS:
        core_pool = sum(by_cuisine[c] for c in spec.cuisine_tags)
        assert core_pool >= spec.menu_size - spec.side_quota, spec.name
        assert side_pool >= spec.side_quota, spec.name


def test_seed_populates_every_table(small_seed: dict[str, object]) -> None:
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Restaurant)) == len(RESTAURANTS)
        assert session.scalar(select(func.count()).select_from(FoodItem)) == TARGET_ITEM_COUNT
        assert session.scalar(select(func.count()).select_from(User)) == 60
        for model in (Rating, Order, OrderItem, Impression):
            assert session.scalar(select(func.count()).select_from(model)) > 0


def test_seed_is_deterministic(empty_database: None, tmp_meta_path: Path) -> None:
    """Same seed, same data. Without this, no evaluation number is reproducible."""

    def run() -> list[tuple[int, int, float]]:
        seed(SeedConfig(users=40, ratings=600, random_seed=99, reset=True, meta_path=tmp_meta_path))
        with SessionLocal() as session:
            rows = session.execute(
                select(Rating.user_id, Rating.food_item_id, Rating.rating).order_by(Rating.id)
            ).all()
        return [(u, i, float(r)) for u, i, r in rows]

    assert run() == run()


def test_ratings_never_predate_the_item(small_seed: dict[str, object]) -> None:
    """Causality. A rating older than the dish it rates would leak future
    information into any chronological train/test split."""
    with SessionLocal() as session:
        violations = session.execute(
            select(func.count())
            .select_from(Rating)
            .join(FoodItem, FoodItem.id == Rating.food_item_id)
            .where(Rating.created_at < FoodItem.created_at)
        ).scalar()
    assert violations == 0


def test_cold_start_users_exist(small_seed: dict[str, object]) -> None:
    """The user-side cold-start path needs subjects to be testable at all."""
    with SessionLocal() as session:
        counts = dict(
            session.execute(select(Rating.user_id, func.count()).group_by(Rating.user_id)).all()
        )
        user_ids = [row[0] for row in session.execute(select(User.id)).all()]

    below_threshold = [uid for uid in user_ids if counts.get(uid, 0) < 5]
    assert below_threshold, "no user falls below the 5-rating cold-start threshold"


def test_cold_start_items_exist(small_seed: dict[str, object]) -> None:
    """Likewise for the item side: recent arrivals must stay genuinely cold."""
    with SessionLocal() as session:
        counts = dict(
            session.execute(
                select(Rating.food_item_id, func.count()).group_by(Rating.food_item_id)
            ).all()
        )
        item_ids = [row[0] for row in session.execute(select(FoodItem.id)).all()]

    below_threshold = [iid for iid in item_ids if counts.get(iid, 0) < 3]
    assert below_threshold, "no item falls below the 3-rating cold-start threshold"


def test_impressions_contain_both_outcomes(small_seed: dict[str, object]) -> None:
    """Implicit-feedback training is impossible with only y=1 rows: the model
    would have no examples of what people were offered and turned down."""
    with SessionLocal() as session:
        ordered = session.scalar(
            select(func.count()).select_from(Impression).where(Impression.was_ordered.is_(True))
        )
        not_ordered = session.scalar(
            select(func.count()).select_from(Impression).where(Impression.was_ordered.is_(False))
        )
    assert ordered and ordered > 0
    assert not_ordered and not_ordered > ordered


def test_order_totals_match_their_line_items(small_seed: dict[str, object]) -> None:
    with SessionLocal() as session:
        mismatches = session.execute(
            select(
                Order.id, Order.total_amount, func.sum(OrderItem.unit_price * OrderItem.quantity)
            )
            .join(OrderItem, OrderItem.order_id == Order.id)
            .group_by(Order.id, Order.total_amount)
            .having(Order.total_amount != func.sum(OrderItem.unit_price * OrderItem.quantity))
        ).all()
    assert mismatches == []


def test_refuses_to_overwrite_without_reset(
    small_seed: dict[str, object], tmp_meta_path: Path
) -> None:
    """Re-running the seeder by accident should not silently double the data."""
    with pytest.raises(SystemExit):
        seed(SeedConfig(users=10, ratings=50, reset=False, meta_path=tmp_meta_path))
