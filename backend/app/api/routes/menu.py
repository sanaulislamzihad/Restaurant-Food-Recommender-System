"""Menu browsing: filter, sort, paginate."""

from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import ColumnElement, Select, func, select
from sqlalchemy.orm import selectinload

from app.api.deps import DbSession
from app.db.models import FoodItem, Rating
from app.schemas.common import Page
from app.schemas.menu import FoodItemDetail, FoodItemSummary, MenuSort, RestaurantSummary

router = APIRouter(prefix="/menu", tags=["menu"])

MAX_PAGE_SIZE = 100


def _apply_sort(statement: Select[tuple[FoodItem]], sort: MenuSort) -> Select[tuple[FoodItem]]:
    """Translate the sort enum into an ORDER BY.

    Rating and popularity need an aggregate, so they join and group. Items with
    no ratings sort last rather than first: coalescing a missing average to 0
    would bury new dishes, but leaving it NULL puts them at the top on some
    backends and the bottom on others, and "it depends on the database" is not
    an ordering.
    """
    if sort is MenuSort.NAME:
        return statement.order_by(FoodItem.name.asc())
    if sort is MenuSort.PRICE_ASC:
        return statement.order_by(FoodItem.price.asc(), FoodItem.id.asc())
    if sort is MenuSort.PRICE_DESC:
        return statement.order_by(FoodItem.price.desc(), FoodItem.id.asc())
    if sort is MenuSort.NEWEST:
        return statement.order_by(FoodItem.created_at.desc(), FoodItem.id.desc())

    statement = statement.outerjoin(Rating, Rating.food_item_id == FoodItem.id).group_by(
        FoodItem.id
    )
    if sort is MenuSort.RATING:
        return statement.order_by(
            func.coalesce(func.avg(Rating.rating), 0).desc(), func.count(Rating.id).desc()
        )
    return statement.order_by(func.count(Rating.id).desc(), FoodItem.id.asc())


@router.get("", response_model=Page[FoodItemSummary])
def list_menu(
    db: DbSession,
    cuisine: Annotated[list[str] | None, Query(description="repeatable")] = None,
    min_price: Annotated[Decimal | None, Query(ge=0)] = None,
    max_price: Annotated[Decimal | None, Query(ge=0)] = None,
    is_veg: bool | None = None,
    max_spice: Annotated[int | None, Query(ge=0, le=5)] = None,
    restaurant_id: int | None = None,
    search: Annotated[str | None, Query(max_length=120)] = None,
    available_only: bool = True,
    sort: MenuSort = MenuSort.POPULARITY,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[FoodItemSummary]:
    filters: list[ColumnElement[bool]] = []
    if available_only:
        filters.append(FoodItem.is_available.is_(True))
    if cuisine:
        filters.append(FoodItem.cuisine.in_(cuisine))
    if min_price is not None:
        filters.append(FoodItem.price >= min_price)
    if max_price is not None:
        filters.append(FoodItem.price <= max_price)
    if is_veg is not None:
        filters.append(FoodItem.is_veg.is_(is_veg))
    if max_spice is not None:
        filters.append(FoodItem.spice_level <= max_spice)
    if restaurant_id is not None:
        filters.append(FoodItem.restaurant_id == restaurant_id)
    if search:
        pattern = f"%{search.strip()}%"
        filters.append(FoodItem.name.ilike(pattern) | FoodItem.description.ilike(pattern))

    total = db.scalar(select(func.count()).select_from(FoodItem).where(*filters)) or 0

    statement = _apply_sort(select(FoodItem).where(*filters), sort).limit(limit).offset(offset)
    items = db.execute(statement).scalars().unique().all()

    return Page[FoodItemSummary](
        items=[FoodItemSummary.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/cuisines", response_model=list[str])
def list_cuisines(db: DbSession) -> list[str]:
    """Distinct cuisines on the live menu, for building filter controls."""
    rows = db.execute(
        select(FoodItem.cuisine)
        .where(FoodItem.is_available.is_(True))
        .distinct()
        .order_by(FoodItem.cuisine)
    ).all()
    return [row[0] for row in rows]


@router.get("/{item_id}", response_model=FoodItemDetail)
def get_menu_item(item_id: int, db: DbSession) -> FoodItemDetail:
    item = db.execute(
        select(FoodItem).options(selectinload(FoodItem.restaurant)).where(FoodItem.id == item_id)
    ).scalar_one_or_none()

    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Menu item not found.")

    aggregate = db.execute(
        select(func.count(), func.avg(Rating.rating)).where(Rating.food_item_id == item_id)
    ).one()
    count, average = int(aggregate[0]), aggregate[1]

    detail = FoodItemDetail.model_validate(item)
    detail.rating_count = count
    detail.average_rating = float(average) if average is not None else None
    detail.restaurant = (
        RestaurantSummary.model_validate(item.restaurant) if item.restaurant else None
    )
    return detail
