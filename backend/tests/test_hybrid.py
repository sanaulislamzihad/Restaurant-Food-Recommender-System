"""Retrieval, hybrid scoring and the business rules.

The alpha tests matter most. The switching rules are the difference between a
recommender that degrades gracefully for a new customer and one that confidently
recommends nonsense, and they are invisible in any end-to-end assertion.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import FoodItem, ItemNeighbor, Order, OrderItem, Restaurant, User
from app.schemas.recommendation import RecommendationSource
from app.services.ranking import effective_alpha, rank_candidates
from app.services.retrieval import Candidate, retrieve_candidates
from ml.features import (
    ANIMAL_DERIVED_TAGS,
    FeatureBuilder,
    FeatureVocabulary,
    ItemStats,
    UserStats,
)
from ml.two_tower import scale_ratings, unscale_ratings

SETTINGS = Settings(reco_alpha=0.6, reco_min_user_ratings=5, reco_min_item_ratings=3)


# ---------------------------------------------------------------------------
# The switching rules
# ---------------------------------------------------------------------------


def test_a_warm_user_and_warm_item_blend_both_models() -> None:
    alpha = effective_alpha(
        SETTINGS,
        user_rating_count=50,
        item_rating_count=40,
        has_collaborative=True,
        has_content=True,
    )
    assert alpha == pytest.approx(0.6)


def test_a_user_below_the_threshold_is_content_only() -> None:
    """Their learned parameters were shrunk to nothing by the regulariser, so a
    collaborative score for them carries no information."""
    alpha = effective_alpha(
        SETTINGS,
        user_rating_count=4,
        item_rating_count=40,
        has_collaborative=True,
        has_content=True,
    )
    assert alpha == 0.0


def test_an_item_below_the_threshold_is_content_only() -> None:
    """Same reasoning on the item side, and it applies per candidate - one cold
    dish in a list of warm ones is scored differently from its neighbours."""
    alpha = effective_alpha(
        SETTINGS,
        user_rating_count=50,
        item_rating_count=2,
        has_collaborative=True,
        has_content=True,
    )
    assert alpha == 0.0


def test_an_unseen_user_or_item_is_content_only() -> None:
    """Everyone who signed up since the last training run lands here. No config
    value can express this, which is why alpha is computed per candidate."""
    alpha = effective_alpha(
        SETTINGS,
        user_rating_count=50,
        item_rating_count=40,
        has_collaborative=False,
        has_content=True,
    )
    assert alpha == 0.0


def test_without_a_content_model_the_collaborative_score_is_used_alone() -> None:
    """Falling back to collaborative-only beats discarding a usable score."""
    alpha = effective_alpha(
        SETTINGS,
        user_rating_count=50,
        item_rating_count=40,
        has_collaborative=True,
        has_content=False,
    )
    assert alpha == 1.0


# ---------------------------------------------------------------------------
# Scale mapping
# ---------------------------------------------------------------------------


def test_rating_scaling_round_trips() -> None:
    """The content tower is trained on scaled targets and read back in stars;
    if these disagree every content score is silently shifted."""
    ratings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert unscale_ratings(scale_ratings(ratings)) == pytest.approx(ratings)


def test_scaling_maps_the_rating_range_onto_the_cosine_range() -> None:
    assert scale_ratings(np.array([1.0]))[0] == pytest.approx(-1.0)
    assert scale_ratings(np.array([5.0]))[0] == pytest.approx(1.0)
    assert scale_ratings(np.array([3.0]))[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _builder() -> FeatureBuilder:
    return FeatureBuilder(
        FeatureVocabulary(
            cuisines=["bengali", "dessert"], areas=["Dhanmondi"], tags=["rice", "milk"]
        )
    )


def test_a_brand_new_user_still_produces_a_meaningful_vector() -> None:
    """The cold-start case: registration details alone must yield a usable
    vector, which is what the content model needs and collaborative filtering
    cannot use."""
    builder = _builder()
    vector = builder.user_vector(
        {"id": 999, "age": 28, "gender": "female", "area": "Dhanmondi", "spice_tolerance": 4},
        UserStats(),
    )
    assert len(vector) == len(builder.user_columns)
    assert any(value != 0.0 for value in vector)
    # The "no history" flag must be set so the network can tell this apart from
    # a user who genuinely averages zero orders.
    assert vector[builder.user_columns.index("has_no_history")] == 1.0


def test_an_unrated_item_is_flagged_rather_than_scored_zero() -> None:
    builder = _builder()
    vector = builder.item_vector(
        {
            "id": 1,
            "cuisine": "bengali",
            "spice_level": 3,
            "is_veg": False,
            "is_rice_based": True,
            "price": 300.0,
            "prep_time_min": 20,
            "ingredient_tags": ["rice"],
        },
        ItemStats(global_mean=3.5),
    )
    assert vector[builder.item_columns.index("has_no_ratings")] == 1.0
    # Falls back to the global mean, not to zero, which would read as terrible.
    assert vector[builder.item_columns.index("average_rating")] == pytest.approx(3.5 / 5.0)


def test_vegan_is_derived_from_ingredients() -> None:
    """The schema has no vegan column; the brief wants the feature."""
    builder = _builder()
    base = {
        "id": 1,
        "cuisine": "dessert",
        "spice_level": 0,
        "is_veg": True,
        "is_rice_based": False,
        "price": 100.0,
        "prep_time_min": 10,
    }
    vegan_slot = builder.item_columns.index("is_vegan")

    dairy = builder.item_vector({**base, "ingredient_tags": ["milk", "sugar"]}, ItemStats())
    plant = builder.item_vector({**base, "ingredient_tags": ["mango", "sugar"]}, ItemStats())

    assert "milk" in ANIMAL_DERIVED_TAGS
    assert dairy[vegan_slot] == 0.0
    assert plant[vegan_slot] == 1.0


def test_unknown_categories_land_in_the_unknown_slot() -> None:
    """A menu that gains a cuisine after training must not shift every column."""
    builder = _builder()
    vector = builder.item_vector(
        {
            "id": 1,
            "cuisine": "martian",
            "spice_level": 0,
            "is_veg": False,
            "is_rice_based": False,
            "price": 100.0,
            "prep_time_min": 10,
            "ingredient_tags": [],
        },
        ItemStats(),
    )
    assert len(vector) == len(builder.item_columns)
    assert vector[builder.item_columns.index("cuisine=unknown")] == 1.0


# ---------------------------------------------------------------------------
# Retrieval and ranking against a database
# ---------------------------------------------------------------------------


@pytest.fixture
def catalogue(db_session: Session) -> dict[str, object]:
    restaurant = Restaurant(name="Test", area="Dhanmondi", cuisine_tags=["bengali"])
    db_session.add(restaurant)
    db_session.flush()

    items = []
    for index in range(8):
        item = FoodItem(
            restaurant_id=restaurant.id,
            name=f"Dish {index}",
            cuisine="bengali" if index < 5 else "dessert",
            spice_level=2,
            price=Decimal("200.00"),
            ingredient_tags=[],
            is_available=index != 7,  # dish 7 is sold out
            is_promoted=index == 3,
        )
        db_session.add(item)
        db_session.flush()
        items.append(item)

    user = User(name="U", email="u@example.com", password_hash="x", spice_tolerance=3)
    db_session.add(user)
    db_session.flush()

    order = Order(user_id=user.id, total_amount=Decimal("200.00"), created_at=datetime.now(UTC))
    db_session.add(order)
    db_session.flush()
    db_session.add(
        OrderItem(
            order_id=order.id,
            food_item_id=items[0].id,
            quantity=1,
            unit_price=Decimal("200.00"),
        )
    )
    db_session.add(
        ItemNeighbor(
            food_item_id=items[0].id, neighbor_id=items[1].id, distance=Decimal("0.4"), rank=1
        )
    )
    # A neighbour that is sold out - retrieval must not shortlist it.
    db_session.add(
        ItemNeighbor(
            food_item_id=items[0].id, neighbor_id=items[7].id, distance=Decimal("0.5"), rank=2
        )
    )
    db_session.commit()
    return {"items": items, "user": user, "order": order}


def test_retrieval_never_returns_nothing_for_a_brand_new_user(
    catalogue: dict[str, object], db_session: Session
) -> None:
    """The cold-start guarantee. The popularity floor exists for exactly this."""
    stranger = User(name="New", email="new@example.com", password_hash="x", spice_tolerance=2)
    db_session.add(stranger)
    db_session.commit()

    result = retrieve_candidates(db_session, stranger.id)
    assert len(result) > 0
    assert result.source_counts["similar_to_ordered"] == 0
    assert result.source_counts["popular"] > 0


def test_retrieval_excludes_unavailable_items(
    catalogue: dict[str, object], db_session: Session
) -> None:
    user = catalogue["user"]
    result = retrieve_candidates(db_session, user.id)  # type: ignore[union-attr]
    assert all(candidate.item.is_available for candidate in result.candidates)


def test_retrieval_explains_neighbours_by_the_dish_that_was_ordered(
    catalogue: dict[str, object], db_session: Session
) -> None:
    user = catalogue["user"]
    items = catalogue["items"]
    result = retrieve_candidates(db_session, user.id)  # type: ignore[union-attr]

    by_id = {c.item.id: c for c in result.candidates}
    neighbour = by_id[items[1].id]  # type: ignore[index]
    assert neighbour.source is RecommendationSource.SIMILAR_TO_ORDERED
    assert neighbour.reason == "Because you liked Dish 0"


def test_ranking_suppresses_something_ordered_in_the_last_day(
    catalogue: dict[str, object], db_session: Session
) -> None:
    """Nobody wants last night's dinner recommended to them this morning."""
    user = catalogue["user"]
    items = catalogue["items"]
    candidates = [
        Candidate(item=items[0], source=RecommendationSource.POPULAR, reason="x"),  # type: ignore[index]
        Candidate(item=items[1], source=RecommendationSource.POPULAR, reason="y"),  # type: ignore[index]
    ]

    top, diagnostics = rank_candidates(
        db_session,
        user,  # type: ignore[arg-type]
        candidates,
        settings=SETTINGS,
        user_rating_count=0,
        collaborative_scores={},
        content_scores={items[0].id: 5.0, items[1].id: 3.0},  # type: ignore[index]
        limit=10,
    )

    assert diagnostics.dropped_recent == 1
    assert [entry.candidate.item.id for entry in top] == [items[1].id]  # type: ignore[index]


def test_an_old_order_is_not_suppressed(catalogue: dict[str, object], db_session: Session) -> None:
    order = catalogue["order"]
    order.created_at = datetime.now(UTC) - timedelta(days=5)  # type: ignore[union-attr]
    db_session.commit()

    user = catalogue["user"]
    items = catalogue["items"]
    candidates = [Candidate(item=items[0], source=RecommendationSource.POPULAR, reason="x")]  # type: ignore[index]

    top, diagnostics = rank_candidates(
        db_session,
        user,  # type: ignore[arg-type]
        candidates,
        settings=SETTINGS,
        user_rating_count=0,
        collaborative_scores={},
        content_scores={items[0].id: 4.0},  # type: ignore[index]
        limit=10,
    )
    assert diagnostics.dropped_recent == 0
    assert len(top) == 1


def test_a_promotion_boosts_but_does_not_override(
    catalogue: dict[str, object], db_session: Session
) -> None:
    """A promotion should tip a close call, not float a bad dish to the top."""
    user = catalogue["user"]
    items = catalogue["items"]
    promoted, plain, much_better = items[3], items[4], items[5]  # type: ignore[index]

    candidates = [
        Candidate(item=promoted, source=RecommendationSource.POPULAR, reason="p"),
        Candidate(item=plain, source=RecommendationSource.POPULAR, reason="q"),
        Candidate(item=much_better, source=RecommendationSource.POPULAR, reason="r"),
    ]
    top, diagnostics = rank_candidates(
        db_session,
        user,  # type: ignore[arg-type]
        candidates,
        settings=SETTINGS,
        user_rating_count=0,
        collaborative_scores={},
        content_scores={promoted.id: 4.0, plain.id: 4.1, much_better.id: 4.9},
        limit=10,
    )

    order = [entry.candidate.item.id for entry in top]
    assert diagnostics.promoted_boosted == 1
    # +0.25 lifts it past the dish it was 0.1 behind ...
    assert order.index(promoted.id) < order.index(plain.id)
    # ... but not past one that is 0.9 better.
    assert order.index(much_better.id) < order.index(promoted.id)


def test_the_blend_is_the_documented_arithmetic(
    catalogue: dict[str, object], db_session: Session
) -> None:
    user = catalogue["user"]
    items = catalogue["items"]
    item = items[4]  # type: ignore[index]

    # Give the item enough ratings that neither switching rule fires.
    from app.db.models import Rating

    for index in range(4):
        other = User(
            name=f"R{index}",
            email=f"r{index}@example.com",
            password_hash="x",
            spice_tolerance=2,
        )
        db_session.add(other)
        db_session.flush()
        db_session.add(Rating(user_id=other.id, food_item_id=item.id, rating=Decimal("4")))
    db_session.commit()

    top, _ = rank_candidates(
        db_session,
        user,  # type: ignore[arg-type]
        [Candidate(item=item, source=RecommendationSource.POPULAR, reason="z")],
        settings=SETTINGS,
        user_rating_count=50,
        collaborative_scores={item.id: 5.0},
        content_scores={item.id: 2.5},
        limit=10,
    )
    # 0.6 * 5.0 + 0.4 * 2.5 = 4.0
    assert top[0].alpha == pytest.approx(0.6)
    assert top[0].score == pytest.approx(4.0)


def test_ranking_falls_back_to_the_item_mean_when_no_model_can_score(
    catalogue: dict[str, object], db_session: Session
) -> None:
    """Better to say what everyone else thought than to invent an ordering."""
    from app.db.models import Rating

    user = catalogue["user"]
    items = catalogue["items"]
    item = items[5]  # type: ignore[index]

    rater = User(name="X", email="x@example.com", password_hash="x", spice_tolerance=2)
    db_session.add(rater)
    db_session.flush()
    db_session.add(Rating(user_id=rater.id, food_item_id=item.id, rating=Decimal("5")))
    db_session.commit()

    top, diagnostics = rank_candidates(
        db_session,
        user,  # type: ignore[arg-type]
        [Candidate(item=item, source=RecommendationSource.POPULAR, reason="z")],
        settings=SETTINGS,
        user_rating_count=0,
        collaborative_scores={},
        content_scores={},
        limit=10,
    )
    assert diagnostics.scored_by_fallback == 1
    assert top[0].score == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Item-side cold start
# ---------------------------------------------------------------------------


def _fake_content_model():
    """A ContentModel with random weights, for exercising the plumbing."""
    from ml.content_inference import ContentModel

    builder = _builder()
    rng = np.random.default_rng(0)
    n_user = len(builder.user_columns)
    n_item = len(builder.item_columns)

    def tower(n_in: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        widths = [n_in, 16, 8, 4]
        kernels = [rng.normal(size=(widths[i], widths[i + 1])).astype(np.float32) for i in range(3)]
        biases = [rng.normal(size=(widths[i + 1],)).astype(np.float32) for i in range(3)]
        return kernels, biases

    user_kernels, user_biases = tower(n_user)
    item_kernels, item_biases = tower(n_item)

    return ContentModel(
        user_kernels=user_kernels,
        user_biases=user_biases,
        item_kernels=item_kernels,
        item_biases=item_biases,
        item_embeddings=np.zeros((0, 4), dtype=np.float32),
        item_ids=[],
        vocabulary=builder.vocabulary,
        item_stats=ItemStats(),
        user_stats=UserStats(),
        mode="explicit",
        logit_scale=1.0,
        metadata={},
    )


NEW_DISH = {
    "id": 9_999,
    "cuisine": "bengali",
    "spice_level": 5,
    "is_veg": False,
    "is_rice_based": False,
    "price": 520.0,
    "prep_time_min": 45,
    "ingredient_tags": ["chilli"],
}


def test_a_dish_added_after_training_can_still_be_embedded() -> None:
    """The item-side cold start. Without the item tower exported alongside the
    precomputed embeddings, a new dish could not be scored at all - which is the
    one case a feature-based tower exists to handle."""
    model = _fake_content_model()
    assert NEW_DISH["id"] not in model.item_index()

    embedding = model.embed_item(NEW_DISH)
    assert embedding.shape == (4,)
    assert float(embedding @ embedding) == pytest.approx(1.0, abs=1e-5)


def test_a_new_dish_is_scored_differently_for_different_users() -> None:
    """A score that ignored the user would make the whole exercise pointless."""
    model = _fake_content_model()
    hot = {"id": 1, "age": 30, "gender": "male", "area": "Dhanmondi", "spice_tolerance": 5}
    mild = {"id": 2, "age": 30, "gender": "male", "area": "Dhanmondi", "spice_tolerance": 0}

    assert model.score_new_item(hot, NEW_DISH) != model.score_new_item(mild, NEW_DISH)


def test_an_on_demand_score_stays_on_the_rating_scale() -> None:
    model = _fake_content_model()
    user = {"id": 1, "age": 30, "gender": "male", "area": "Dhanmondi", "spice_tolerance": 3}
    assert 1.0 <= model.score_new_item(user, NEW_DISH) <= 5.0
