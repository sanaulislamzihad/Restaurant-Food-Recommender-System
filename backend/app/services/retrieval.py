"""Retrieval: turn a 300-dish catalogue into ~150 candidates worth ranking.

Two stages exist for one reason. Scoring every dish for every request works at
300 items and stops working long before a real chain's catalogue; retrieval is
cheap and index-backed, ranking is expensive and model-backed, so the expensive
stage should only ever see a shortlist.

Three sources, in descending order of how personal they are:

1. **Neighbours of dishes the user has shown they like.** The strongest signal,
   and the only one that yields a specific reason - "because you liked Kacchi
   Biryani" names a real dish.

   Both ratings and orders count, and a rating comes first: rating a dish four
   or five stars is an explicit statement of preference, whereas an order only
   says somebody tried it once and may well have been disappointed. Drawing
   from orders alone - which is what a literal reading of the brief gives -
   means a customer who rates a dish but has never ordered anything gets no
   personalised candidates at all.
2. **Popular dishes in the cuisines they favour**, judged by ratings and orders
   together for the same reason. Broader, still personal.
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

#: How many of the user's liked dishes to draw neighbours from.
RECENT_ITEM_LIMIT = 10
#: A rating at or above this counts as "they liked it".
LIKED_THRESHOLD = 4.0
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


def _seed_items(db: Session, user_id: int, limit: int) -> list[tuple[FoodItem, str]]:
    """Dishes to draw neighbours from, each with the verb that explains it.

    Highly-rated dishes are taken first, then recently ordered ones. A four- or
    five-star rating is the customer saying outright that they liked something;
    an order only says they tried it.

    Returns ``(item, verb)`` pairs where the verb is "liked" or "ordered", so
    the reason shown to the customer describes what actually happened rather
    than assuming.
    """
    rated = (
        db.execute(
            select(FoodItem)
            .join(Rating, Rating.food_item_id == FoodItem.id)
            .where(Rating.user_id == user_id, Rating.rating >= LIKED_THRESHOLD)
            .order_by(Rating.created_at.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )

    ordered = (
        db.execute(
            select(FoodItem)
            .join(OrderItem, OrderItem.food_item_id == FoodItem.id)
            .join(Order, Order.id == OrderItem.order_id)
            .where(Order.user_id == user_id)
            .order_by(Order.created_at.desc())
            .limit(limit * 4)  # over-fetch: the same dish is often ordered repeatedly
        )
        .scalars()
        .all()
    )

    seen: set[int] = set()
    seeds: list[tuple[FoodItem, str]] = []
    for item, verb in [(i, "liked") for i in rated] + [(i, "ordered") for i in ordered]:
        if item.id in seen:
            continue
        seen.add(item.id)
        seeds.append((item, verb))
        if len(seeds) == limit:
            break
    return seeds


def _top_cuisines(db: Session, user_id: int, limit: int) -> list[str]:
    """The user's favourite cuisines, from ratings and orders combined.

    Counting orders alone leaves a customer who only rates with no favourite
    cuisine at all, and therefore no candidates from this source.
    """
    tally: dict[str, int] = {}

    for cuisine, count in db.execute(
        select(FoodItem.cuisine, func.count())
        .join(Rating, Rating.food_item_id == FoodItem.id)
        .where(Rating.user_id == user_id, Rating.rating >= LIKED_THRESHOLD)
        .group_by(FoodItem.cuisine)
    ).all():
        tally[str(cuisine)] = tally.get(str(cuisine), 0) + int(count)

    for cuisine, count in db.execute(
        select(FoodItem.cuisine, func.count())
        .join(OrderItem, OrderItem.food_item_id == FoodItem.id)
        .join(Order, Order.id == OrderItem.order_id)
        .where(Order.user_id == user_id)
        .group_by(FoodItem.cuisine)
    ).all():
        tally[str(cuisine)] = tally.get(str(cuisine), 0) + int(count)

    return [name for name, _ in sorted(tally.items(), key=lambda kv: -kv[1])[:limit]]


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
    #
    # Matched on *name*, not id. The same dish is sold by several restaurants as
    # separate rows, so an id-only check happily recommends "Tandoori Roti" to
    # somebody who has just rated Tandoori Roti - correct by the data model and
    # plainly wrong to the customer reading it.
    already_rated = {
        str(row[0]).casefold()
        for row in db.execute(
            select(FoodItem.name)
            .join(Rating, Rating.food_item_id == FoodItem.id)
            .where(Rating.user_id == user_id)
        ).all()
    }
    #: Names already shortlisted, so the same dish from two restaurants does not
    #: take two slots in one feed.
    chosen_names: set[str] = set()

    def add(item: FoodItem, source: RecommendationSource, reason: str) -> bool:
        name = item.name.casefold()
        if not item.is_available or item.id in chosen:
            return False
        if name in already_rated or name in chosen_names:
            return False
        chosen_names.add(name)
        chosen[item.id] = Candidate(item=item, source=source, reason=reason)
        return True

    # ---- 1. Neighbours of dishes they liked or ordered -------------------
    recent = _seed_items(db, user_id, RECENT_ITEM_LIMIT)
    added = 0
    if recent:
        recent_ids = [item.id for item, _ in recent]
        names = {item.id: (item.name, verb) for item, verb in recent}
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
            name, verb = names[source_id]
            for neighbour in by_source.get(source_id, []):
                if add(
                    neighbour,
                    RecommendationSource.SIMILAR_TO_ORDERED,
                    f"Because you {verb} {name}",
                ):
                    added += 1
    result.source_counts["similar_to_liked"] = added

    # ---- 2. Their most-ordered cuisines ---------------------------------
    added = 0
    for cuisine in _top_cuisines(db, user_id, TOP_CUISINE_COUNT):
        label = cuisine.replace("_", " ")
        for item in _popular_in_cuisine(db, cuisine, ITEMS_PER_CUISINE):
            if add(
                item,
                RecommendationSource.FAVOURITE_CUISINE,
                # Neutral wording on purpose: the cuisine may come from ratings,
                # from orders, or both, and "you order a lot of this" is simply
                # false for a customer who has only ever rated.
                f"More {label} - one of your favourites",
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
