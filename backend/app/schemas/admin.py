"""Admin payloads: the model card, and retraining."""

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


class ModelCard(BaseModel):
    """What is currently loaded and how it scored.

    This is the metadata written at training time, surfaced rather than
    recomputed, so the dashboard cannot disagree with the artifact on disk.
    """

    version: str
    model: str | None = None
    trained_at: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    training_metrics: dict[str, Any] = Field(default_factory=dict)
    runtime_seconds: float | None = None
    tensorflow_version: str | None = None


class ModelMetricsResponse(BaseModel):
    loaded: ModelCard | None
    available_versions: list[str]
    neighbours_indexed: int
    # Held-out results from ml.evaluate, if it has been run. Kept separate from
    # training_metrics so nobody mistakes a training fit for a held-out score.
    evaluation: dict[str, Any] | None = None


class RetrainRequest(BaseModel):
    mode: str = Field(default="explicit", pattern="^(explicit|implicit)$")
    iterations: int = Field(default=400, ge=10, le=5000)
    lambda_: float = Field(default=1.0, gt=0, alias="lambda")
    rebuild_neighbours: bool = True


class RetrainResponse(BaseModel):
    status: str
    detail: str
    command: str = Field(description="the CLI equivalent of this request")


# ---------------------------------------------------------------------------
# Menu management
# ---------------------------------------------------------------------------


class MenuItemCreate(BaseModel):
    restaurant_id: int
    name: str = Field(min_length=1, max_length=160)
    description: str = ""
    cuisine: str = Field(min_length=1, max_length=60)
    spice_level: int = Field(default=0, ge=0, le=5)
    is_veg: bool = False
    is_rice_based: bool = False
    price: Decimal = Field(gt=0)
    prep_time_min: int = Field(default=20, ge=1, le=240)
    image_url: str | None = None
    ingredient_tags: list[str] = Field(default_factory=list)
    is_available: bool = True
    is_promoted: bool = False


class MenuItemUpdate(BaseModel):
    """Every field optional: a PATCH should be able to change one thing."""

    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = None
    cuisine: str | None = Field(default=None, min_length=1, max_length=60)
    spice_level: int | None = Field(default=None, ge=0, le=5)
    is_veg: bool | None = None
    is_rice_based: bool | None = None
    price: Decimal | None = Field(default=None, gt=0)
    prep_time_min: int | None = Field(default=None, ge=1, le=240)
    image_url: str | None = None
    ingredient_tags: list[str] | None = None
    is_available: bool | None = None
    is_promoted: bool | None = None


class CuisineCoverage(BaseModel):
    cuisine: str
    total_items: int
    #: Items with enough ratings for collaborative filtering to place them.
    recommendable: int
    #: Items with no ratings at all - only the content model can rank these.
    cold: int


class CoverageResponse(BaseModel):
    """How much of the catalogue the recommender can actually reach.

    A model that scores well while only ever surfacing the same few dishes is a
    bad recommender with a good metric, so this is worth watching alongside the
    accuracy numbers.
    """

    total_items: int
    available_items: int
    promoted_items: int
    items_with_neighbours: int
    min_item_ratings: int
    by_cuisine: list[CuisineCoverage]
