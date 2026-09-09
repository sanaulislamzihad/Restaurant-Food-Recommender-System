"""Recommendation endpoints."""

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import CurrentUser, DbSession
from app.schemas.recommendation import RecommendationResponse
from app.services.recommender import popular_response, recommend_for_user, similar_items

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/for-me", response_model=RecommendationResponse)
def recommendations_for_me(
    user: CurrentUser,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> RecommendationResponse:
    """The personalised feed.

    Falls back to popularity when the user is too new for collaborative
    filtering to say anything; ``is_cold_start`` in the response says which of
    the two happened, so the frontend can label the row honestly instead of
    calling a popularity list "picked for you".
    """
    return recommend_for_user(db, user, limit)


@router.get("/similar/{item_id}", response_model=RecommendationResponse)
def similar_to_item(
    item_id: int,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> RecommendationResponse:
    """Dishes near this one in the learned feature space.

    Open to anonymous callers: it is a property of the item, not of the viewer,
    and it is the row shown under every menu item.
    """
    return similar_items(db, item_id, limit)


@router.get("/popular", response_model=RecommendationResponse)
def popular(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> RecommendationResponse:
    """Most-rated dishes. The cold-start fallback, and the anonymous home page."""
    return popular_response(db, limit)
