"""Recommendation serving.

Loads the trained collaborative filtering artifacts once per process and scores
against them. No training, no TensorFlow, no distance computation happens here:
prediction is a dot product over saved numpy arrays, and item similarity is a
lookup in a table the offline job filled in.

Three entry points, matching the three endpoints:

* :func:`recommend_for_user` - the personalised feed
* :func:`similar_items` - "you liked this, try these"
* :func:`popular_items` - the cold-start fallback, and the anonymous home page

Every returned item carries a ``reason``. The reason is derived from what
actually put the item in the list, so it can never claim a connection the data
does not support.

M4 scope note: this ranks the whole catalogue directly. The two-stage retrieval
pipeline and the hybrid content blend belong to M6, and the seams they need -
candidate counting, source attribution - are already here.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, selectinload

from app.core.cache import (
    Cache,
    get_cache,
    popular_key,
    recommendations_key,
    similar_key,
)
from app.core.config import get_settings
from app.db.models import FoodItem, ItemNeighbor, Rating, User
from app.schemas.menu import FoodItemSummary
from app.schemas.recommendation import (
    RecommendationResponse,
    RecommendationSource,
    RecommendedItem,
)
from ml.artifacts import CFArtifacts, load_cf_artifacts

logger = logging.getLogger(__name__)

#: A rating at or above this counts as "you liked it" when explaining a
#: recommendation. Saying "because you liked X" about a dish somebody gave two
#: stars would be worse than giving no reason at all.
LIKED_THRESHOLD = 4.0

#: How many of the user's liked dishes are used to attribute reasons.
REASON_SOURCE_LIMIT = 10


@dataclass
class LoadedModel:
    artifacts: CFArtifacts
    user_index: dict[int, int]
    item_index: dict[int, int]


class _ModelHolder:
    """Process-wide holder for the trained model.

    Loaded lazily on first use rather than at import, so the API still starts
    and serves the menu when no model has been trained yet. A missing model
    degrades recommendations to the popularity fallback; it does not take the
    service down.
    """

    def __init__(self) -> None:
        self._model: LoadedModel | None = None
        self._attempted = False
        self._lock = threading.Lock()

    def get(self) -> LoadedModel | None:
        if self._model is None and not self._attempted:
            with self._lock:
                if self._model is None and not self._attempted:
                    self._attempted = True
                    self._model = self._load()
        return self._model

    def reload(self) -> LoadedModel | None:
        with self._lock:
            self._attempted = True
            self._model = self._load()
        return self._model

    @staticmethod
    def _load() -> LoadedModel | None:
        settings = get_settings()
        try:
            artifacts = load_cf_artifacts(
                settings.model_root, settings.model_version or None, mode="explicit"
            )
        except FileNotFoundError:
            logger.warning(
                "No trained model available; recommendations will fall back to popularity. "
                "Run `python -m ml.train_cf`."
            )
            return None
        logger.info("loaded collaborative filtering model %s", artifacts.version)
        return LoadedModel(
            artifacts=artifacts,
            user_index=artifacts.user_index(),
            item_index=artifacts.item_index(),
        )


_holder = _ModelHolder()


def get_model() -> LoadedModel | None:
    return _holder.get()


def reload_model() -> LoadedModel | None:
    """Drop and re-read the artifacts. Called after a retrain."""
    return _holder.reload()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def _available_items_query() -> Select[tuple[FoodItem]]:
    return select(FoodItem).where(FoodItem.is_available.is_(True))


def _liked_items(db: Session, user_id: int, limit: int = REASON_SOURCE_LIMIT) -> list[FoodItem]:
    """The user's most recent well-rated dishes, newest first."""
    rows = db.execute(
        select(FoodItem)
        .join(Rating, Rating.food_item_id == FoodItem.id)
        .where(Rating.user_id == user_id, Rating.rating >= LIKED_THRESHOLD)
        .order_by(Rating.created_at.desc())
        .limit(limit)
    ).scalars()
    return list(rows)


def _neighbour_attribution(db: Session, liked: list[FoodItem]) -> dict[int, str]:
    """Map candidate item id -> the liked dish that explains it.

    Built from the precomputed neighbour table, so "because you liked X" only
    ever appears when X really is among the item's nearest neighbours in the
    learned feature space.
    """
    if not liked:
        return {}

    liked_ids = [item.id for item in liked]
    names = {item.id: item.name for item in liked}

    rows = db.execute(
        select(ItemNeighbor.food_item_id, ItemNeighbor.neighbor_id)
        .where(ItemNeighbor.food_item_id.in_(liked_ids))
        .order_by(ItemNeighbor.food_item_id, ItemNeighbor.rank)
    ).all()

    attribution: dict[int, str] = {}
    for source_id, neighbour_id in rows:
        # First writer wins, and liked items arrive newest first, so the
        # explanation cites the most recent thing the user enjoyed.
        attribution.setdefault(neighbour_id, names[source_id])
    return attribution


def _rating_stats(db: Session) -> dict[int, tuple[int, float]]:
    """item id -> (rating count, average rating)."""
    rows = db.execute(
        select(Rating.food_item_id, func.count(), func.avg(Rating.rating)).group_by(
            Rating.food_item_id
        )
    ).all()
    return {item_id: (int(count), float(average)) for item_id, count, average in rows}


# ---------------------------------------------------------------------------
# Popular
# ---------------------------------------------------------------------------


def popular_items(db: Session, limit: int = 20, *, cache: Cache | None = None) -> list[FoodItem]:
    """Most-rated available dishes, best average breaking ties.

    Ordering by count first and average second on purpose: a dish with one
    five-star rating is not more popular than one with four hundred at 4.6, and
    ranking by average alone would fill the page with obscure items that
    happened to please the two people who tried them.
    """
    cache = cache or get_cache()
    key = popular_key(limit)
    cached_ids = cache.get(key)

    if isinstance(cached_ids, list) and cached_ids:
        items = (
            db.execute(_available_items_query().where(FoodItem.id.in_(cached_ids))).scalars().all()
        )
        by_id = {item.id: item for item in items}
        ordered = [by_id[i] for i in cached_ids if i in by_id]
        if ordered:
            return ordered

    rows = (
        db.execute(
            select(FoodItem)
            .outerjoin(Rating, Rating.food_item_id == FoodItem.id)
            .where(FoodItem.is_available.is_(True))
            .group_by(FoodItem.id)
            .order_by(
                func.count(Rating.id).desc(), func.coalesce(func.avg(Rating.rating), 0).desc()
            )
            .limit(limit)
        )
        .scalars()
        .all()
    )

    items = list(rows)
    cache.set(key, [item.id for item in items], get_settings().cache_ttl_seconds)
    return items


def popular_response(db: Session, limit: int = 20) -> RecommendationResponse:
    started = time.perf_counter()
    items = popular_items(db, limit)
    stats = _rating_stats(db)

    recommended = []
    for item in items:
        count, average = stats.get(item.id, (0, 0.0))
        reason = (
            f"Popular right now - {count} people rated it {average:.1f} stars"
            if count
            else "New on the menu"
        )
        recommended.append(
            RecommendedItem(
                item=FoodItemSummary.model_validate(item),
                score=float(count),
                reason=reason,
                source=RecommendationSource.POPULAR,
            )
        )

    return RecommendationResponse(
        items=recommended,
        model_version=None,
        is_cold_start=True,
        candidates_considered=len(items),
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )


# ---------------------------------------------------------------------------
# Personalised feed
# ---------------------------------------------------------------------------


def recommend_for_user(
    db: Session, user: User, limit: int = 20, *, cache: Cache | None = None
) -> RecommendationResponse:
    """Personalised recommendations, with a popularity fallback for cold users."""
    settings = get_settings()
    cache = cache or get_cache()
    started = time.perf_counter()

    rating_count = (
        db.scalar(select(func.count()).select_from(Rating).where(Rating.user_id == user.id)) or 0
    )
    model = get_model()

    # Three separate reasons to fall back, all landing in the same place: no
    # model on disk, a user the model has never seen, or a user with too little
    # history for their learned parameters to mean anything.
    if model is None or user.id not in model.user_index:
        return popular_response(db, limit)
    if rating_count < settings.reco_min_user_ratings:
        response = popular_response(db, limit)
        response.is_cold_start = True
        return response

    key = recommendations_key(user.id, model.artifacts.version, limit)
    cached = cache.get(key)
    if cached is not None:
        try:
            response = RecommendationResponse.model_validate(cached)
            response.cached = True
            # Report this request's latency, not the stored one. Echoing the
            # original computation's timing would make a cache hit look like it
            # took 30ms and quietly hide the thing the field exists to measure.
            response.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return response
        except ValueError:
            logger.warning("stale cache shape for %s; recomputing", key)

    column = model.user_index[user.id]
    predictions = model.artifacts.predict_for_user_index(column)

    already_rated = {
        row[0]
        for row in db.execute(select(Rating.food_item_id).where(Rating.user_id == user.id)).all()
    }
    available = db.execute(_available_items_query()).scalars().all()

    candidates: list[tuple[float, FoodItem]] = []
    for item in available:
        if item.id in already_rated:
            continue
        row = model.item_index.get(item.id)
        if row is None:
            # Added to the menu after the model was trained. It has no learned
            # features, so collaborative filtering has nothing to say about it;
            # the content model in M6 is what will rescue these.
            continue
        candidates.append((float(predictions[row]), item))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    top = candidates[:limit]

    liked = _liked_items(db, user.id)
    attribution = _neighbour_attribution(db, liked)

    recommended = []
    for score, item in top:
        source_name = attribution.get(item.id)
        if source_name:
            reason = f"Because you liked {source_name}"
            source = RecommendationSource.SIMILAR_TO_ORDERED
        else:
            reason = "Popular with customers who share your taste"
            source = RecommendationSource.COLLABORATIVE
        recommended.append(
            RecommendedItem(
                item=FoodItemSummary.model_validate(item),
                score=round(score, 4),
                reason=reason,
                source=source,
            )
        )

    response = RecommendationResponse(
        items=recommended,
        model_version=model.artifacts.version,
        is_cold_start=False,
        candidates_considered=len(candidates),
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    cache.set(key, response.model_dump(mode="json"), settings.cache_ttl_seconds)
    return response


# ---------------------------------------------------------------------------
# Similar items
# ---------------------------------------------------------------------------


def similar_items(
    db: Session, item_id: int, limit: int = 10, *, cache: Cache | None = None
) -> RecommendationResponse:
    """Nearest neighbours of one dish, straight from the precomputed table."""
    cache = cache or get_cache()
    started = time.perf_counter()

    model = get_model()
    version = model.artifacts.version if model else "none"
    key = similar_key(item_id, version, limit)

    cached = cache.get(key)
    if cached is not None:
        try:
            response = RecommendationResponse.model_validate(cached)
            response.cached = True
            response.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return response
        except ValueError:
            logger.warning("stale cache shape for %s; recomputing", key)

    source = db.get(FoodItem, item_id)
    if source is None:
        return RecommendationResponse(
            items=[], model_version=model.artifacts.version if model else None
        )

    rows = db.execute(
        select(FoodItem, ItemNeighbor.distance, ItemNeighbor.rank)
        .join(ItemNeighbor, ItemNeighbor.neighbor_id == FoodItem.id)
        .where(ItemNeighbor.food_item_id == item_id, FoodItem.is_available.is_(True))
        .order_by(ItemNeighbor.rank)
    ).all()

    recommended = []
    # The same dish sold at two restaurants is two rows with the same name, and
    # a carousel showing "New York Cheesecake" as similar to New York Cheesecake
    # reads as a bug even though the model is right. Dedupe on name at serving
    # time rather than corrupting the neighbour table, which is a pure artifact.
    seen_names = {source.name.casefold()}
    for item, distance, _rank in rows:
        name = item.name.casefold()
        if name in seen_names:
            continue
        seen_names.add(name)
        recommended.append(
            RecommendedItem(
                item=FoodItemSummary.model_validate(item),
                # Distance is better when smaller; invert it so the score
                # sorts the same direction as every other score in the API.
                score=round(1.0 / (1.0 + float(distance)), 4),
                reason=f"Similar to {source.name}",
                source=RecommendationSource.SIMILAR_TO_ORDERED,
            )
        )
        if len(recommended) == limit:
            break

    response = RecommendationResponse(
        items=recommended,
        model_version=model.artifacts.version if model else None,
        candidates_considered=len(rows),
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    cache.set(key, response.model_dump(mode="json"), get_settings().cache_ttl_seconds)
    return response


def invalidate_user_recommendations(user_id: int, cache: Cache | None = None) -> None:
    """Drop a user's cached feed after they rate or order something.

    Without this a customer rates a dish, reloads, and sees the identical list -
    which reads as the rating having been ignored.
    """
    cache = cache or get_cache()
    model = get_model()
    version = model.artifacts.version if model else "none"
    for limit in (10, 20, 50):
        cache.delete(recommendations_key(user_id, version, limit))


def load_item_with_restaurant(db: Session, item_id: int) -> FoodItem | None:
    return db.execute(
        select(FoodItem).options(selectinload(FoodItem.restaurant)).where(FoodItem.id == item_id)
    ).scalar_one_or_none()
