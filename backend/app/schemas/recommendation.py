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
    latency_ms: float = Field(default=0.0, description="server-side ranking time")
    cached: bool = False
