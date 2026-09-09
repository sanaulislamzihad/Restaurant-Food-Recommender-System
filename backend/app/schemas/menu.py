"""Menu browsing payloads."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import Field

from app.schemas.common import ORMModel


class MenuSort(StrEnum):
    """Sort orders the menu endpoint accepts.

    A closed set rather than a free-text column name: taking an arbitrary
    string and interpolating it into ORDER BY is how injection gets in.
    """

    NAME = "name"
    PRICE_ASC = "price_asc"
    PRICE_DESC = "price_desc"
    RATING = "rating"
    POPULARITY = "popularity"
    NEWEST = "newest"


class RestaurantSummary(ORMModel):
    id: int
    name: str
    area: str
    cuisine_tags: list[str]


class FoodItemSummary(ORMModel):
    """The card view: everything a grid tile needs and nothing more."""

    id: int
    name: str
    cuisine: str
    price: Decimal
    spice_level: int
    is_veg: bool
    is_rice_based: bool
    prep_time_min: int
    image_url: str | None
    is_available: bool
    restaurant_id: int


class FoodItemDetail(FoodItemSummary):
    """The detail view, with the aggregates a card does not need."""

    description: str
    ingredient_tags: list[str]
    created_at: datetime
    restaurant: RestaurantSummary | None = None
    average_rating: float | None = Field(
        default=None, description="mean of all ratings, null when nobody has rated it"
    )
    rating_count: int = 0
