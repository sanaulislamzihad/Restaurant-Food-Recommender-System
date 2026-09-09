"""Admin endpoints: the model card, and triggering a retrain.

Authentication note, stated plainly rather than left implicit: these are behind
the ordinary user token, which means any registered user can read the model card
and request a retrain. There is no role column in the schema, so there is
nothing to check against yet. Adding ``users.is_admin`` and gating on it is the
obvious next step and is deliberately deferred rather than faked.
"""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, status
from sqlalchemy import func, select

from app.api.deps import DbSession, get_current_user
from app.core.config import BACKEND_DIR, get_settings
from app.db.models import ItemNeighbor
from app.schemas.admin import ModelCard, ModelMetricsResponse, RetrainRequest, RetrainResponse
from app.services.recommender import get_model, reload_model
from ml.artifacts import list_versions

# Auth is required but the user object is never read, so it is declared once
# here rather than threaded through every handler as an unused argument.
router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(get_current_user)])


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
