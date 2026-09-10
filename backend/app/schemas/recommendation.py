"""Recommendation payloads.

Every recommended item carries a ``reason``. Explainability is a product
feature here, not decoration: "because you liked Kacchi Biryani" is the
difference between a list a customer trusts and a list that feels arbitrary.
The reason is generated from whichever retrieval source surfaced the item, so
it always describes something that actually happened.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.menu import FoodItemSummary


class RecommendationSource(StrEnum):
    """Why an item entered the candidate list.

    Kept as a closed set so the frontend can style the reason chips and the
    admin dashboard can report which source is carrying the feed.
    """

    SIMILAR_TO_ORDERED = "similar_to_ordered"
    FAVOURITE_CUISINE = "favourite_cuisine"
    POPULAR = "popular"
    TRENDING = "trending"
    COLLABORATIVE = "collaborative"


class RecommendedItem(BaseModel):
    item: FoodItemSummary
    score: float = Field(description="ranking score; comparable only within one response")
    reason: str = Field(description="human-readable explanation shown to the customer")
    source: RecommendationSource


class RecommendationResponse(BaseModel):
    items: list[RecommendedItem]
    model_version: str | None = Field(
        default=None, description="null when the response came from a non-model fallback"
    )
    is_cold_start: bool = Field(
        default=False,
        description="true when the user had too little history for collaborative filtering",
    )
    candidates_considered: int = Field(
        default=0, description="size of the retrieval stage's candidate list"
    )
    # The brief asks for retrieval size and ranking latency to be logged so the
    # two-stage trade-off can be measured; they are returned as well, because a
    # number in a log file is far harder to act on than one in the response.
    retrieval_ms: float = Field(default=0.0, description="time spent building candidates")
    ranking_ms: float = Field(default=0.0, description="time spent scoring candidates")
    retrieval_sources: dict[str, int] = Field(
        default_factory=dict,
        description="how many candidates each retrieval source contributed",
    )
    scoring_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="how the returned items were scored: hybrid, content-only, etc.",
    )
    dropped_recently_ordered: int = Field(
        default=0, description="candidates suppressed for having been ordered very recently"
    )
    latency_ms: float = Field(default=0.0, description="total server-side time")
    cached: bool = False
