"""Ranking: score the shortlist, apply business rules, return the feed.

The hybrid score is

    score = alpha * collaborative + (1 - alpha) * content

which is only meaningful because both terms are in **star units**. The
collaborative model predicts a rating directly; the content model's cosine is
mapped back onto the 1-5 scale before it leaves `ContentModel.score_items`. A
literal reading of the brief would add a 1-5 rating to a -1..1 cosine, where the
larger-magnitude term wins every time for no reason connected to quality.

`alpha` is not one global number. It is forced to 0 - content only - in three
situations, and the reasons differ:

* the user has too few ratings for their learned parameters to mean anything
* the *item* has too few ratings for its learned features to mean anything
* the collaborative model has never seen this user or item at all, which happens
  to everyone who signed up since the last training run

The third case is the one a static config value cannot express, and it is why
alpha is computed per candidate rather than read from settings once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import FoodItem, Order, OrderItem, Rating, User
from app.schemas.recommendation import RecommendationSource
from app.services.retrieval import Candidate

#: Score used when neither model can say anything about a dish. The item's own
#: mean rating is the honest fallback: it is what everyone else thought.
FALLBACK_TO_ITEM_MEAN = True


@dataclass
class ScoredCandidate:
    candidate: Candidate
    score: float
    alpha: float
    collaborative: float | None
    content: float | None
    promoted: bool = False


@dataclass
class RankingDiagnostics:
    """What the ranking stage did, for the response and the logs."""

    candidates_in: int = 0
    dropped_recent: int = 0
    promoted_boosted: int = 0
    scored_by_hybrid: int = 0
    scored_by_content_only: int = 0
    scored_by_collaborative_only: int = 0
    scored_by_fallback: int = 0
    ranking_ms: float = 0.0
    source_counts: dict[str, int] = field(default_factory=dict)


def user_feature_row(user: User) -> dict[str, object]:
    """The plain-dict view of a user the feature builder expects."""
    return {
        "id": user.id,
        "age": user.age,
        "gender": str(user.gender) if user.gender else None,
        "area": user.area,
        "spice_tolerance": user.spice_tolerance,
    }


def effective_alpha(
    settings: Settings,
    *,
    user_rating_count: int,
    item_rating_count: int,
    has_collaborative: bool,
    has_content: bool,
) -> float:
    """How much of the blended score comes from collaborative filtering."""
    if not has_collaborative:
        return 0.0
    if not has_content:
        # Nothing to blend with. Using the collaborative score alone beats
        # discarding it.
        return 1.0
    if user_rating_count < settings.reco_min_user_ratings:
        return 0.0
    if item_rating_count < settings.reco_min_item_ratings:
        return 0.0
    return settings.reco_alpha


def _recently_ordered_item_ids(db: Session, user_id: int, hours: int) -> set[int]:
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    rows = db.execute(
        select(OrderItem.food_item_id)
        .join(Order, Order.id == OrderItem.order_id)
        .where(Order.user_id == user_id, Order.created_at >= cutoff)
    ).all()
    return {int(row[0]) for row in rows}


def _item_rating_counts(db: Session, item_ids: list[int]) -> dict[int, int]:
    if not item_ids:
        return {}
    rows = db.execute(
        select(Rating.food_item_id, func.count())
        .where(Rating.food_item_id.in_(item_ids))
        .group_by(Rating.food_item_id)
    ).all()
    return {int(item_id): int(count) for item_id, count in rows}


def _item_mean_ratings(db: Session, item_ids: list[int]) -> dict[int, float]:
    if not item_ids:
        return {}
    rows = db.execute(
        select(Rating.food_item_id, func.avg(Rating.rating))
        .where(Rating.food_item_id.in_(item_ids))
        .group_by(Rating.food_item_id)
    ).all()
    return {int(item_id): float(average) for item_id, average in rows}


def rank_candidates(
    db: Session,
    user: User,
    candidates: list[Candidate],
    *,
    settings: Settings,
    user_rating_count: int,
    collaborative_scores: dict[int, float],
    content_scores: dict[int, float],
    limit: int,
) -> tuple[list[ScoredCandidate], RankingDiagnostics]:
    """Score, apply business rules, and cut to the top N.

    Args:
        collaborative_scores: item id -> predicted rating, for items the CF
            model can place. Missing means "this model has nothing to say".
        content_scores: item id -> predicted rating from the two-tower model.
    """
    started = time.perf_counter()
    diagnostics = RankingDiagnostics(candidates_in=len(candidates))

    item_ids = [candidate.item.id for candidate in candidates]
    rating_counts = _item_rating_counts(db, item_ids)
    recent = _recently_ordered_item_ids(db, user.id, settings.reco_repeat_suppression_hours)
    means = _item_mean_ratings(db, item_ids) if FALLBACK_TO_ITEM_MEAN else {}

    scored: list[ScoredCandidate] = []
    for candidate in candidates:
        item = candidate.item

        # Business rule: do not recommend tonight what they ate this morning.
        if item.id in recent:
            diagnostics.dropped_recent += 1
            continue

        collaborative = collaborative_scores.get(item.id)
        content = content_scores.get(item.id)
        alpha = effective_alpha(
            settings,
            user_rating_count=user_rating_count,
            item_rating_count=rating_counts.get(item.id, 0),
            has_collaborative=collaborative is not None,
            has_content=content is not None,
        )

        if collaborative is not None and content is not None and 0.0 < alpha < 1.0:
            score = alpha * collaborative + (1.0 - alpha) * content
            diagnostics.scored_by_hybrid += 1
        elif content is not None and alpha == 0.0:
            score = content
            diagnostics.scored_by_content_only += 1
        elif collaborative is not None:
            score = collaborative
            diagnostics.scored_by_collaborative_only += 1
        else:
            score = means.get(item.id, 3.0)
            diagnostics.scored_by_fallback += 1

        promoted = bool(item.is_promoted)
        if promoted:
            # A promotion tips a close call; it does not override the model.
            score += settings.reco_promotion_boost
            diagnostics.promoted_boosted += 1

        scored.append(
            ScoredCandidate(
                candidate=candidate,
                score=score,
                alpha=alpha,
                collaborative=collaborative,
                content=content,
                promoted=promoted,
            )
        )

    # Ties broken by item id so the feed is stable between identical requests.
    scored.sort(key=lambda entry: (-entry.score, entry.candidate.item.id))
    top = scored[:limit]

    for entry in top:
        key = str(entry.candidate.source)
        diagnostics.source_counts[key] = diagnostics.source_counts.get(key, 0) + 1

    diagnostics.ranking_ms = round((time.perf_counter() - started) * 1000, 2)
    return top, diagnostics


def describe(entry: ScoredCandidate) -> str:
    """The reason string shown to the customer.

    Retrieval already produced a specific reason; ranking only adds the
    promotion note, because a boosted position is something the customer is
    entitled to know about.
    """
    reason = entry.candidate.reason
    if entry.promoted:
        return f"{reason} · on offer"
    return reason


def source_of(entry: ScoredCandidate) -> RecommendationSource:
    return entry.candidate.source


def food_item_of(entry: ScoredCandidate) -> FoodItem:
    return entry.candidate.item
