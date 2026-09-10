"""Recommendation serving.

The request path is retrieval then ranking:

    retrieve ~150 candidates (indexed queries)  ->  score them (hybrid model)

Neither stage needs TensorFlow. Collaborative predictions are a dot product over
saved numpy arrays, the content user tower is three numpy matrix multiplies, and
item similarity is a lookup in a table an offline job filled in.

Both models are optional and independently versioned. With neither, the API
still serves popularity; with only one, it serves that one. A missing model
degrades the feed, it does not take the service down.
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
from app.services.ranking import describe, rank_candidates, user_feature_row
from app.services.retrieval import retrieve_candidates
from ml.artifacts import CFArtifacts, latest_version_with, load_cf_artifacts
from ml.content_inference import (
    CONTENT_WEIGHTS_FILE,
    ContentModel,
    load_content_model,
)

logger = logging.getLogger(__name__)

LIKED_THRESHOLD = 4.0


@dataclass
class LoadedModel:
    artifacts: CFArtifacts
    user_index: dict[int, int]
    item_index: dict[int, int]


class _ModelHolder:
    """Process-wide holder for both trained models.

    Loaded lazily on first use rather than at import, so the API still starts
    and serves the menu before anything has been trained.
    """

    def __init__(self) -> None:
        self._collaborative: LoadedModel | None = None
        self._content: ContentModel | None = None
        self._attempted = False
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        if self._attempted:
            return
        with self._lock:
            if self._attempted:
                return
            self._attempted = True
            self._collaborative = self._load_collaborative()
            self._content = self._load_content()

    def collaborative(self) -> LoadedModel | None:
        self._ensure()
        return self._collaborative

    def content(self) -> ContentModel | None:
        self._ensure()
        return self._content

    def reload(self) -> None:
        with self._lock:
            self._attempted = True
            self._collaborative = self._load_collaborative()
            self._content = self._load_content()

    @staticmethod
    def _load_collaborative() -> LoadedModel | None:
        settings = get_settings()
        try:
            artifacts = load_cf_artifacts(
                settings.model_root, settings.model_version or None, mode="explicit"
            )
        except FileNotFoundError:
            logger.warning(
                "No collaborative model; the feed falls back to content and popularity. "
                "Run `python -m ml.train_cf`."
            )
            return None
        logger.info("loaded collaborative model %s", artifacts.version)
        return LoadedModel(
            artifacts=artifacts,
            user_index=artifacts.user_index(),
            item_index=artifacts.item_index(),
        )

    @staticmethod
    def _load_content() -> ContentModel | None:
        settings = get_settings()
        root = settings.model_root
        version = latest_version_with(root, CONTENT_WEIGHTS_FILE)
        if version is None:
            logger.warning(
                "No content model; cold-start users get popularity only. "
                "Run `python -m ml.train_content`."
            )
            return None
        model = load_content_model(root, version)
        if model is not None:
            logger.info("loaded content model %s", version)
        return model


_holder = _ModelHolder()


def get_model() -> LoadedModel | None:
    return _holder.collaborative()


def get_content_model() -> ContentModel | None:
    return _holder.content()


def reload_model() -> LoadedModel | None:
    """Drop and re-read both models. Called after a retrain."""
    _holder.reload()
    return _holder.collaborative()


def model_version_label() -> str:
    """Identifies the serving configuration, for cache keys and responses."""
    collaborative = _holder.collaborative()
    content = _holder.content()
    parts = [
        collaborative.artifacts.version if collaborative else "none",
        content.version if content else "none",
    ]
    return "+".join(parts)


# ---------------------------------------------------------------------------
# Shared queries
# ---------------------------------------------------------------------------


def _available_items_query() -> Select[tuple[FoodItem]]:
    return select(FoodItem).where(FoodItem.is_available.is_(True))


def _rating_stats(db: Session) -> dict[int, tuple[int, float]]:
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

    Count first, average second: a dish with one five-star rating is not more
    popular than one with four hundred at 4.6, and ranking by average alone
    fills the page with obscure items that pleased the two people who tried them.
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
                func.count(Rating.id).desc(),
                func.coalesce(func.avg(Rating.rating), 0).desc(),
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
    """Retrieve, then rank. The full pipeline."""
    settings = get_settings()
    cache = cache or get_cache()
    started = time.perf_counter()

    version = model_version_label()
    key = recommendations_key(user.id, version, limit)
    cached = cache.get(key)
    if cached is not None:
        try:
            response = RecommendationResponse.model_validate(cached)
            response.cached = True
            response.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return response
        except ValueError:
            logger.warning("stale cache shape for %s; recomputing", key)

    collaborative = _holder.collaborative()
    content = _holder.content()

    rating_count = (
        db.scalar(select(func.count()).select_from(Rating).where(Rating.user_id == user.id)) or 0
    )
    is_cold_start = rating_count < settings.reco_min_user_ratings

    # With no model at all there is nothing to rank with; popularity is the
    # honest answer rather than an arbitrary ordering dressed up as one.
    if collaborative is None and content is None:
        return popular_response(db, limit)

    candidates = retrieve_candidates(db, user.id)
    if not candidates.candidates:
        return popular_response(db, limit)

    item_ids = candidates.item_ids

    collaborative_scores: dict[int, float] = {}
    if collaborative is not None and user.id in collaborative.user_index:
        column = collaborative.user_index[user.id]
        predictions = collaborative.artifacts.predict_for_user_index(column)
        for item_id in item_ids:
            row = collaborative.item_index.get(item_id)
            if row is not None:
                collaborative_scores[item_id] = float(predictions[row])

    content_scores: dict[int, float] = {}
    if content is not None:
        user_row = user_feature_row(user)
        content_scores = content.score_items(user_row, item_ids)

        # Dishes added to the menu since the last training run have no
        # precomputed embedding. Embedding them on demand is the whole point of
        # a feature-based tower: without this they would silently fall through
        # to the item-mean fallback, which is the one case the content model
        # exists to rescue.
        missing = [
            candidate.item
            for candidate in candidates.candidates
            if candidate.item.id not in content_scores
        ]
        for item in missing:
            content_scores[item.id] = content.score_new_item(
                user_row,
                {
                    "id": item.id,
                    "cuisine": item.cuisine,
                    "spice_level": item.spice_level,
                    "is_veg": item.is_veg,
                    "is_rice_based": item.is_rice_based,
                    "price": float(item.price),
                    "prep_time_min": item.prep_time_min,
                    "ingredient_tags": item.ingredient_tags or [],
                },
            )
        if missing:
            logger.info(
                "embedded %d item(s) on demand: added to the menu after training",
                len(missing),
            )

    top, diagnostics = rank_candidates(
        db,
        user,
        candidates.candidates,
        settings=settings,
        user_rating_count=rating_count,
        collaborative_scores=collaborative_scores,
        content_scores=content_scores,
        limit=limit,
    )

    logger.info(
        "recommendations user=%s retrieved=%d ranked=%d retrieval_ms=%.1f ranking_ms=%.1f "
        "hybrid=%d content_only=%d cf_only=%d fallback=%d",
        user.id,
        len(candidates),
        len(top),
        candidates.retrieval_ms,
        diagnostics.ranking_ms,
        diagnostics.scored_by_hybrid,
        diagnostics.scored_by_content_only,
        diagnostics.scored_by_collaborative_only,
        diagnostics.scored_by_fallback,
    )

    response = RecommendationResponse(
        items=[
            RecommendedItem(
                item=FoodItemSummary.model_validate(entry.candidate.item),
                score=round(entry.score, 4),
                reason=describe(entry),
                source=entry.candidate.source,
            )
            for entry in top
        ],
        model_version=version,
        is_cold_start=is_cold_start,
        candidates_considered=len(candidates),
        retrieval_ms=candidates.retrieval_ms,
        ranking_ms=diagnostics.ranking_ms,
        retrieval_sources=candidates.source_counts,
        scoring_breakdown={
            "hybrid": diagnostics.scored_by_hybrid,
            "content_only": diagnostics.scored_by_content_only,
            "collaborative_only": diagnostics.scored_by_collaborative_only,
            "item_mean_fallback": diagnostics.scored_by_fallback,
        },
        dropped_recently_ordered=diagnostics.dropped_recent,
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
    """Dishes similar to this one.

    Prefers the collaborative neighbour table, which reflects how people
    actually rate. Falls back to content-embedding similarity for a dish too new
    to have neighbours - the case the collaborative model cannot serve at all.
    """
    cache = cache or get_cache()
    started = time.perf_counter()

    version = model_version_label()
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
        return RecommendationResponse(items=[], model_version=version)

    rows = db.execute(
        select(FoodItem, ItemNeighbor.distance)
        .join(ItemNeighbor, ItemNeighbor.neighbor_id == FoodItem.id)
        .where(ItemNeighbor.food_item_id == item_id, FoodItem.is_available.is_(True))
        .order_by(ItemNeighbor.rank)
    ).all()

    scored: list[tuple[FoodItem, float, str]] = [
        # Distance is better when smaller; invert so the score sorts the same
        # direction as every other score in the API.
        (item, 1.0 / (1.0 + float(distance)), "collaborative")
        for item, distance in rows
    ]
    used_content = False

    if not scored:
        content = _holder.content()
        if content is not None:
            pairs = content.similar_items(item_id, limit * 2)
            ids = [pair[0] for pair in pairs]
            by_id = {
                item.id: item
                for item in db.execute(
                    _available_items_query().where(FoodItem.id.in_(ids))
                ).scalars()
            }
            scored = [
                (by_id[i], float(similarity), "content") for i, similarity in pairs if i in by_id
            ]
            used_content = bool(scored)

    recommended = []
    # The same dish sold at two restaurants is two rows, and a carousel showing
    # "New York Cheesecake" as similar to New York Cheesecake reads as a bug
    # even though the model is right. Deduped at serving time rather than in the
    # neighbour table, which is a pure model artifact.
    seen_names = {source.name.casefold()}
    for item, score, _origin in scored:
        name = item.name.casefold()
        if name in seen_names:
            continue
        seen_names.add(name)
        recommended.append(
            RecommendedItem(
                item=FoodItemSummary.model_validate(item),
                score=round(score, 4),
                reason=(
                    f"Also enjoyed by people who like {source.name}"
                    if not used_content
                    else f"Similar in style to {source.name}"
                ),
                source=RecommendationSource.SIMILAR_TO_ORDERED,
            )
        )
        if len(recommended) == limit:
            break

    response = RecommendationResponse(
        items=recommended,
        model_version=version,
        candidates_considered=len(scored),
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
    version = model_version_label()
    for limit in (10, 12, 20, 50):
        cache.delete(recommendations_key(user_id, version, limit))


def load_item_with_restaurant(db: Session, item_id: int) -> FoodItem | None:
    return db.execute(
        select(FoodItem).options(selectinload(FoodItem.restaurant)).where(FoodItem.id == item_id)
    ).scalar_one_or_none()
