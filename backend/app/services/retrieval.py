"""Retrieval: turn a 300-dish catalogue into ~150 candidates worth ranking.

Two stages exist for one reason. Scoring every dish for every request works at
300 items and stops working long before a real chain's catalogue; retrieval is
cheap and index-backed, ranking is expensive and model-backed, so the expensive
stage should only ever see a shortlist.

Three sources, in descending order of how personal they are:

1. **Neighbours of what the user recently ordered.** The strongest signal, and
   the only one that yields a specific reason - "because you liked Kacchi
   Biryani" names a real dish the customer really ordered.
2. **Popular dishes in the cuisines they order most.** Broader, still personal.
3. **Popular overall.** The floor. This is what guarantees a brand-new customer
   never receives an empty feed, which is the whole cold-start requirement.

A dish surfaced by more than one source keeps the first, most personal one, so
the explanation shown is the strongest true thing that can be said about it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.cache import Cache, get_cache
from app.core.config import get_settings
from app.db.models import FoodItem, ItemNeighbor, Order, OrderItem, Rating
from app.schemas.recommendation import RecommendationSource

#: How many recent orders to draw neighbours from.
RECENT_ITEM_LIMIT = 10
#: Neighbours pulled per recent item.
NEIGHBOURS_PER_ITEM = 10
#: Cuisines considered "the user's", and how many dishes to take from each.
TOP_CUISINE_COUNT = 3
ITEMS_PER_CUISINE = 10
#: The popularity floor.
POPULAR_FLOOR = 20


@dataclass
class Candidate:
    item: FoodItem
    source: RecommendationSource
    reason: str


@dataclass
class CandidateSet:
    candidates: list[Candidate] = field(default_factory=list)
    #: How many each source contributed before de-duplication.
    source_counts: dict[str, int] = field(default_factory=dict)
    retrieval_ms: float = 0.0

    @property
    def item_ids(self) -> list[int]:
        return [candidate.item.id for candidate in self.candidates]

    def __len__(self) -> int:
        return len(self.candidates)


def _recent_ordered_items(db: Session, user_id: int, limit: int) -> list[FoodItem]:
    """The user's most recently ordered dishes, newest first, de-duplicated."""
    rows = db.execute(
        select(FoodItem, Order.created_at)
        .join(OrderItem, OrderItem.food_item_id == FoodItem.id)
        .join(Order, Order.id == OrderItem.order_id)
        .where(Order.user_id == user_id)
        .order_by(Order.created_at.desc())
        .limit(limit * 4)  # over-fetch: the same dish is often ordered repeatedly
    ).all()

    seen: set[int] = set()
    items: list[FoodItem] = []
    for item, _ in rows:
        if item.id in seen:
            continue
        seen.add(item.id)
        items.append(item)
        if len(items) == limit:
            break
    return items


def _top_cuisines(db: Session, user_id: int, limit: int) -> list[str]:
    rows = db.execute(
        select(FoodItem.cuisine, func.count())
        .join(OrderItem, OrderItem.food_item_id == FoodItem.id)
        .join(Order, Order.id == OrderItem.order_id)
        .where(Order.user_id == user_id)
        .group_by(FoodItem.cuisine)
        .order_by(func.count().desc())
        .limit(limit)
    ).all()
    return [str(row[0]) for row in rows]


def _by_ids_preserving_order(db: Session, ids: list[int]) -> list[FoodItem]:
    rows = (
        db.execute(select(FoodItem).where(FoodItem.id.in_(ids), FoodItem.is_available.is_(True)))
        .scalars()
        .all()
    )
    by_id = {item.id: item for item in rows}
    return [by_id[i] for i in ids if i in by_id]


def _popular_in_cuisine(
    db: Session, cuisine: str, limit: int, cache: Cache | None = None
) -> list[FoodItem]:
    """Popular dishes in one cuisine.

    Cached: this does not depend on who is asking, so recomputing it per request
    is the same aggregate over the whole ratings table for every user on the
    site.
    """
    cache = cache or get_cache()
    key = f"retrieval:cuisine:{cuisine}:{limit}"
    cached = cache.get(key)
    if isinstance(cached, list) and cached:
        items = _by_ids_preserving_order(db, [int(i) for i in cached])
        if items:
            return items

    result = list(
        db.execute(
            select(FoodItem)
            .outerjoin(Rating, Rating.food_item_id == FoodItem.id)
            .where(FoodItem.is_available.is_(True), FoodItem.cuisine == cuisine)
            .group_by(FoodItem.id)
            .order_by(func.count(Rating.id).desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    cache.set(key, [item.id for item in result], get_settings().cache_ttl_seconds)
    return result


def _popular_overall(db: Session, limit: int, cache: Cache | None = None) -> list[FoodItem]:
    """The popularity floor. Identical for every user, so it is cached."""
    cache = cache or get_cache()
    key = f"retrieval:popular:{limit}"
    cached = cache.get(key)
    if isinstance(cached, list) and cached:
        items = _by_ids_preserving_order(db, [int(i) for i in cached])
        if items:
            return items

    result = list(
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
    cache.set(key, [item.id for item in result], get_settings().cache_ttl_seconds)
    return result


def retrieve_candidates(db: Session, user_id: int) -> CandidateSet:
    """Build the candidate shortlist for one user.

    Unavailable dishes are filtered here rather than after ranking, so a
    sold-out item never consumes a slot in the shortlist that a orderable dish
    could have used.
    """
    started = time.perf_counter()
    result = CandidateSet()
    chosen: dict[int, Candidate] = {}

    # Someone who has already reviewed a dish does not need it recommended;
    # showing it wastes a slot in a short list on something they have decided
    # about. Excluded here rather than after ranking so a rated dish never
    # displaces a candidate that could have been shown.
    already_rated = {
        int(row[0])
        for row in db.execute(select(Rating.food_item_id).where(Rating.user_id == user_id)).all()
    }

    def add(item: FoodItem, source: RecommendationSource, reason: str) -> bool:
        if not item.is_available or item.id in chosen or item.id in already_rated:
            return False
        chosen[item.id] = Candidate(item=item, source=source, reason=reason)
        return True

    # ---- 1. Neighbours of recent orders ---------------------------------
    recent = _recent_ordered_items(db, user_id, RECENT_ITEM_LIMIT)
    added = 0
    if recent:
        recent_ids = [item.id for item in recent]
        names = {item.id: item.name for item in recent}
        rows = db.execute(
            select(ItemNeighbor.food_item_id, FoodItem)
            .join(FoodItem, FoodItem.id == ItemNeighbor.neighbor_id)
            .where(
                ItemNeighbor.food_item_id.in_(recent_ids),
                ItemNeighbor.rank <= NEIGHBOURS_PER_ITEM,
            )
            .order_by(ItemNeighbor.food_item_id, ItemNeighbor.rank)
        ).all()
        # Recent items are ordered newest first, so walking them in that order
        # means the reason cites the most recent dish that explains a candidate.
        by_source: dict[int, list[FoodItem]] = {}
        for source_id, neighbour in rows:
            by_source.setdefault(int(source_id), []).append(neighbour)
        for source_id in recent_ids:
            for neighbour in by_source.get(source_id, []):
                if add(
                    neighbour,
                    RecommendationSource.SIMILAR_TO_ORDERED,
                    f"Because you liked {names[source_id]}",
                ):
                    added += 1
    result.source_counts["similar_to_ordered"] = added

    # ---- 2. Their most-ordered cuisines ---------------------------------
    added = 0
    for cuisine in _top_cuisines(db, user_id, TOP_CUISINE_COUNT):
        label = cuisine.replace("_", " ")
        for item in _popular_in_cuisine(db, cuisine, ITEMS_PER_CUISINE):
            if add(
                item,
                RecommendationSource.FAVOURITE_CUISINE,
                f"You order a lot of {label}",
            ):
                added += 1
    result.source_counts["favourite_cuisine"] = added

    # ---- 3. The popularity floor ----------------------------------------
    added = 0
    for item in _popular_overall(db, POPULAR_FLOOR):
        if add(item, RecommendationSource.POPULAR, "Popular right now"):
            added += 1
    result.source_counts["popular"] = added

    result.candidates = list(chosen.values())
    result.retrieval_ms = round((time.perf_counter() - started) * 1000, 2)
    return result
