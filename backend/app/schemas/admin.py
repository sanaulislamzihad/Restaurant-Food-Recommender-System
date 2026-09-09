"""Admin payloads: the model card, and retraining."""

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
