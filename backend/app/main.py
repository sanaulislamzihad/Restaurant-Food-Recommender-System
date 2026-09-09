"""FastAPI application.

M1 exposes health endpoints only, so the container has a real entrypoint and the
compose healthcheck has something to call. The menu, auth, ratings, orders and
recommendation routers arrive in M4.
"""

from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import engine

settings = get_settings()

app = FastAPI(
    title="Restaurant Food Recommender",
    version="0.1.0",
    description="Collaborative filtering + two-tower content model behind a food ordering API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    environment: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    database: bool
    dialect: str


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Liveness. Deliberately touches nothing external: a failing database should
    not cause the orchestrator to restart a process that is running fine."""
    return HealthResponse(status="ok", environment=settings.environment)


@app.get("/health/ready", response_model=ReadinessResponse, tags=["health"])
def readiness() -> ReadinessResponse:
    """Readiness. Reports whether the process can actually serve traffic."""
    database_ok = True
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - any connection failure means not ready
        database_ok = False

    return ReadinessResponse(
        status="ready" if database_ok else "degraded",
        database=database_ok,
        dialect=engine.dialect.name,
    )
