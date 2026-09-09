"""Ratings, orders and impression logging."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, insert, select
from sqlalchemy.orm import selectinload

from app.api.deps import CurrentUser, DbSession
from app.db.models import FoodItem, Impression, Order, OrderItem, OrderStatus, Rating
from app.schemas.common import Page
from app.schemas.interactions import (
    ImpressionBatch,
    ImpressionBatchResponse,
    OrderCreate,
    OrderResponse,
    RatingCreate,
    RatingResponse,
    RatingWithItem,
)
from app.services.recommender import invalidate_user_recommendations

router = APIRouter(tags=["interactions"])


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------


@router.post("/ratings", response_model=RatingResponse, status_code=status.HTTP_201_CREATED)
def submit_rating(payload: RatingCreate, user: CurrentUser, db: DbSession) -> Rating:
    """Create or update this user's rating for a dish.

    Upsert rather than insert: the schema allows one rating per user per item,
    and a customer changing their mind should not get a 409 telling them they
    already have an opinion.
    """
    item = db.get(FoodItem, payload.food_item_id)
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Menu item not found.")

    rating = db.scalar(
        select(Rating).where(Rating.user_id == user.id, Rating.food_item_id == payload.food_item_id)
    )
    if rating is None:
        rating = Rating(
            user_id=user.id, food_item_id=payload.food_item_id, rating=Decimal(payload.rating)
        )
        db.add(rating)
    else:
        rating.rating = Decimal(payload.rating)
        rating.created_at = datetime.now(UTC)

    db.commit()
    db.refresh(rating)

    # The feed must visibly react; otherwise rating something appears to do
    # nothing until the cache happens to expire.
    invalidate_user_recommendations(user.id)
    return rating


@router.get("/ratings/me", response_model=Page[RatingWithItem])
def my_ratings(
    user: CurrentUser,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[RatingWithItem]:
    total = (
        db.scalar(select(func.count()).select_from(Rating).where(Rating.user_id == user.id)) or 0
    )
    rows = (
        db.execute(
            select(Rating)
            .options(selectinload(Rating.food_item))
            .where(Rating.user_id == user.id)
            .order_by(Rating.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )

    return Page[RatingWithItem](
        items=[RatingWithItem.model_validate(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


@router.post("/orders", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
def create_order(payload: OrderCreate, user: CurrentUser, db: DbSession) -> Order:
    """Place an order.

    Prices are read from the database, never from the request. A client that
    could name its own ``unit_price`` would be able to buy a 900 taka steak for
    one taka.
    """
    requested_ids = [line.food_item_id for line in payload.items]
    items = {
        item.id: item
        for item in db.execute(select(FoodItem).where(FoodItem.id.in_(requested_ids)))
        .scalars()
        .all()
    }

    missing = sorted(set(requested_ids) - items.keys())
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown menu items: {missing}"
        )

    unavailable = sorted(i for i in requested_ids if not items[i].is_available)
    if unavailable:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"These items are currently unavailable: {unavailable}",
        )

    order = Order(user_id=user.id, total_amount=Decimal("0.00"), status=OrderStatus.PENDING)
    db.add(order)
    db.flush()

    total = Decimal("0.00")
    for line in payload.items:
        item = items[line.food_item_id]
        unit_price = Decimal(item.price)
        total += unit_price * line.quantity
        db.add(
            OrderItem(
                order_id=order.id,
                food_item_id=item.id,
                quantity=line.quantity,
                unit_price=unit_price,
            )
        )

    order.total_amount = total
    db.commit()

    order = db.execute(
        select(Order)
        .options(selectinload(Order.items).selectinload(OrderItem.order))
        .where(Order.id == order.id)
    ).scalar_one()

    invalidate_user_recommendations(user.id)
    return order


@router.get("/orders/history", response_model=Page[OrderResponse])
def order_history(
    user: CurrentUser,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[OrderResponse]:
    total = db.scalar(select(func.count()).select_from(Order).where(Order.user_id == user.id)) or 0
    rows = (
        db.execute(
            select(Order)
            .options(selectinload(Order.items))
            .where(Order.user_id == user.id)
            .order_by(Order.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .unique()
        .all()
    )

    return Page[OrderResponse](
        items=[OrderResponse.model_validate(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/orders/{order_id}", response_model=OrderResponse)
def get_order(order_id: int, user: CurrentUser, db: DbSession) -> Order:
    order = db.execute(
        select(Order).options(selectinload(Order.items)).where(Order.id == order_id)
    ).scalar_one_or_none()

    # A 404 rather than a 403 for someone else's order: telling the caller the
    # order exists but is not theirs leaks that it exists at all.
    if order is None or order.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found.")
    return order


# ---------------------------------------------------------------------------
# Impressions
# ---------------------------------------------------------------------------


@router.post(
    "/impressions/batch",
    response_model=ImpressionBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def log_impressions(
    payload: ImpressionBatch, user: CurrentUser, db: DbSession
) -> ImpressionBatchResponse:
    """Record what was shown to the user, and whether it converted.

    This is the only source of negative training data. Without it the implicit
    model would see nothing but items people ordered, and could never learn what
    they were offered and passed over.

    Unknown item ids are counted and dropped rather than failing the batch: this
    is fire-and-forget telemetry from a client that may be running a stale menu,
    and rejecting 200 good rows because one dish was delisted would lose real
    signal to protect nothing.
    """
    requested_ids = {line.food_item_id for line in payload.impressions}
    known = {
        row[0]
        for row in db.execute(select(FoodItem.id).where(FoodItem.id.in_(requested_ids))).all()
    }

    now = datetime.now(UTC)
    rows = [
        {
            "user_id": user.id,
            "food_item_id": line.food_item_id,
            "was_ordered": line.was_ordered,
            "shown_at": line.shown_at or now,
        }
        for line in payload.impressions
        if line.food_item_id in known
    ]

    if rows:
        db.execute(insert(Impression), rows)
        db.commit()

    return ImpressionBatchResponse(recorded=len(rows), skipped=len(payload.impressions) - len(rows))
