"""Admin endpoints: the model card, retraining, and menu management.

Gated on ``users.is_admin``. Until M7 these sat behind the ordinary user token
because the schema had no role to check, which meant any registered customer
could read the model card and trigger a retrain.
"""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import DbSession, get_current_admin
from app.core.cache import get_cache
from app.core.config import BACKEND_DIR, get_settings
from app.db.models import FoodItem, ItemNeighbor, Rating, Restaurant
from app.schemas.admin import (
    CoverageResponse,
    CuisineCoverage,
    MenuItemCreate,
    MenuItemUpdate,
    ModelCard,
    ModelMetricsResponse,
    RetrainRequest,
    RetrainResponse,
)
from app.schemas.common import Message
from app.schemas.menu import FoodItemDetail, RestaurantSummary
from app.services.recommender import get_model, reload_model
from ml.artifacts import list_versions

# Every route here requires an administrator. Declared once on the router rather
# than repeated in each handler, none of which reads the user object.
router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(get_current_admin)])


def _evaluation_report() -> dict[str, Any] | None:
    """Held-out results from the last `ml.evaluate` run, if there was one."""
    path = BACKEND_DIR / "data" / "evaluation.json"
    if not path.exists():
        return None
    try:
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return loaded


@router.get("/model/metrics", response_model=ModelMetricsResponse)
def model_metrics(db: DbSession) -> ModelMetricsResponse:
    """What is loaded, what else is on disk, and how it scored."""
    settings = get_settings()
    model = get_model()

    card: ModelCard | None = None
    if model is not None:
        metadata = model.artifacts.metadata
        card = ModelCard(
            version=model.artifacts.version,
            model=metadata.get("model"),
            trained_at=metadata.get("trained_at"),
            hyperparameters=metadata.get("hyperparameters", {}),
            data=metadata.get("data", {}),
            training_metrics=metadata.get("training_metrics", {}),
            runtime_seconds=metadata.get("runtime_seconds"),
            tensorflow_version=metadata.get("tensorflow_version"),
        )

    neighbours = db.scalar(select(func.count()).select_from(ItemNeighbor)) or 0

    return ModelMetricsResponse(
        loaded=card,
        available_versions=list_versions(Path(settings.model_root)),
        neighbours_indexed=int(neighbours),
        evaluation=_evaluation_report(),
    )


@router.post("/model/retrain", response_model=RetrainResponse, status_code=status.HTTP_202_ACCEPTED)
def trigger_retrain(payload: RetrainRequest) -> RetrainResponse:
    """Report how to retrain, and reload whatever is currently on disk.

    This deliberately does **not** spawn training inside the API process.
    Training is a minutes-long, memory-hungry job that would block a worker,
    and the brief is explicit that it stays an offline job. Running it from a
    request handler is exactly the thing that turns one slow endpoint into an
    outage.

    A production version would enqueue onto a worker (Celery, RQ, an ECS task)
    and return a job id to poll. That queue does not exist yet, so rather than
    pretend, this returns the exact command to run and refreshes the loaded
    artifacts so a completed offline run can be picked up without a restart.
    """
    command = (
        f"python -m ml.train_cf --mode {payload.mode} "
        f"--iterations {payload.iterations} --lambda {payload.lambda_}"
    )
    if payload.rebuild_neighbours:
        command += " && python -m ml.build_neighbors"

    model = reload_model()
    loaded = model.artifacts.version if model else "none"

    return RetrainResponse(
        status="accepted",
        detail=(
            "Training is an offline job and is not run inside the API. "
            f"Reloaded model artifacts from disk (now serving: {loaded}). "
            "Run the command below, then call this endpoint again to pick it up."
        ),
        command=command,
    )


# ---------------------------------------------------------------------------
# Menu management
# ---------------------------------------------------------------------------


@router.post("/menu", response_model=FoodItemDetail, status_code=status.HTTP_201_CREATED)
def create_menu_item(payload: MenuItemCreate, db: DbSession) -> FoodItemDetail:
    """Add a dish.

    A dish created here is invisible to the collaborative model until the next
    training run, and has no precomputed content embedding either. The serving
    layer handles that by embedding it on demand, so it is rankable from the
    moment it is saved rather than after the next retrain.
    """
    if db.get(Restaurant, payload.restaurant_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown restaurant.")

    item = FoodItem(**payload.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)

    # A new dish changes what "popular in this cuisine" means.
    get_cache().clear_prefix("retrieval:")
    return _detail(db, item)


@router.patch("/menu/{item_id}", response_model=FoodItemDetail)
def update_menu_item(item_id: int, payload: MenuItemUpdate, db: DbSession) -> FoodItemDetail:
    item = db.get(FoodItem, item_id)
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Menu item not found.")

    # exclude_unset so a PATCH that omits a field leaves it alone rather than
    # overwriting it with the schema default.
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(item, field, value)

    db.commit()
    db.refresh(item)
    get_cache().clear_prefix("retrieval:")
    get_cache().clear_prefix("reco:")
    return _detail(db, item)


@router.delete("/menu/{item_id}", response_model=Message)
def retire_menu_item(item_id: int, db: DbSession) -> Message:
    """Take a dish off the menu.

    This marks it unavailable rather than deleting the row. Past orders
    reference it, and destroying a customer's order history to remove a dish
    from today's menu would be a poor trade - the foreign key is RESTRICT for
    that reason, so a hard delete would fail anyway once anyone had ordered it.
    """
    item = db.get(FoodItem, item_id)
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Menu item not found.")

    item.is_available = False
    db.commit()
    get_cache().clear_prefix("retrieval:")
    get_cache().clear_prefix("reco:")
    return Message(detail=f"{item.name} is no longer available.")


@router.get("/coverage", response_model=CoverageResponse)
def coverage(db: DbSession) -> CoverageResponse:
    """How much of the catalogue the recommender can actually reach."""
    settings = get_settings()
    threshold = settings.reco_min_item_ratings

    counts: dict[int, int] = {
        int(item_id): int(count)
        for item_id, count in db.execute(
            select(Rating.food_item_id, func.count()).group_by(Rating.food_item_id)
        ).all()
    }
    items = db.execute(select(FoodItem.id, FoodItem.cuisine, FoodItem.is_available)).all()
    with_neighbours = db.scalar(select(func.count(func.distinct(ItemNeighbor.food_item_id)))) or 0
    promoted = (
        db.scalar(select(func.count()).select_from(FoodItem).where(FoodItem.is_promoted.is_(True)))
        or 0
    )

    by_cuisine: dict[str, dict[str, int]] = {}
    for item_id, cuisine, _available in items:
        bucket = by_cuisine.setdefault(str(cuisine), {"total": 0, "recommendable": 0, "cold": 0})
        rated = int(counts.get(item_id, 0))
        bucket["total"] += 1
        if rated >= threshold:
            bucket["recommendable"] += 1
        if rated == 0:
            bucket["cold"] += 1

    return CoverageResponse(
        total_items=len(items),
        available_items=sum(1 for _, _, available in items if available),
        promoted_items=int(promoted),
        items_with_neighbours=int(with_neighbours),
        min_item_ratings=threshold,
        by_cuisine=[
            CuisineCoverage(
                cuisine=name,
                total_items=values["total"],
                recommendable=values["recommendable"],
                cold=values["cold"],
            )
            for name, values in sorted(by_cuisine.items())
        ],
    )


def _detail(db: Session, item: FoodItem) -> FoodItemDetail:
    aggregate = db.execute(
        select(func.count(), func.avg(Rating.rating)).where(Rating.food_item_id == item.id)
    ).one()
    detail = FoodItemDetail.model_validate(item)
    detail.rating_count = int(aggregate[0])
    detail.average_rating = float(aggregate[1]) if aggregate[1] is not None else None
    detail.restaurant = (
        RestaurantSummary.model_validate(item.restaurant) if item.restaurant else None
    )
    return detail
