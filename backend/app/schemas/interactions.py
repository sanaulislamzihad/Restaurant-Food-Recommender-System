"""Ratings, orders and impressions."""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from app.db.models import OrderStatus
from app.schemas.common import ORMModel
from app.schemas.menu import FoodItemSummary


class RatingCreate(BaseModel):
    food_item_id: int
    rating: int = Field(ge=1, le=5)


class RatingResponse(ORMModel):
    id: int
    user_id: int
    food_item_id: int
    rating: Decimal
    created_at: datetime


class RatingWithItem(RatingResponse):
    food_item: FoodItemSummary


class OrderItemCreate(BaseModel):
    food_item_id: int
    quantity: int = Field(default=1, ge=1, le=50)


class OrderCreate(BaseModel):
    items: list[OrderItemCreate] = Field(min_length=1, max_length=50)


class OrderItemResponse(ORMModel):
    id: int
    food_item_id: int
    quantity: int
    unit_price: Decimal
    food_item: FoodItemSummary | None = None


class OrderResponse(ORMModel):
    id: int
    user_id: int
    total_amount: Decimal
    status: OrderStatus
    created_at: datetime
    items: list[OrderItemResponse]


class ImpressionCreate(BaseModel):
    food_item_id: int
    was_ordered: bool = False
    shown_at: datetime | None = None


class ImpressionBatch(BaseModel):
    """A batch of what the user was shown.

    Batched because impressions are logged for every card that scrolls past;
    one request per card would put more load on the API than the ordering
    traffic it exists to support.
    """

    impressions: list[ImpressionCreate] = Field(min_length=1, max_length=500)


class ImpressionBatchResponse(BaseModel):
    recorded: int
    skipped: int = Field(description="rows dropped because the item id does not exist")
