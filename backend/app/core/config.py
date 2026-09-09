"""Application configuration, loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Typed settings. See `.env.example` at the repo root for documentation."""

    model_config = SettingsConfigDict(
        # The repo-root .env is checked first so a single file configures both the
        # backend (run from backend/) and docker-compose (run from the root).
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Database -------------------------------------------------------
    database_url: str = "sqlite+pysqlite:///./foodrec.db"

    # ---- Cache ----------------------------------------------------------
    # Empty means "no Redis available" -> the cache layer falls back to an
    # in-process TTL dict so the API still runs with zero external services.
    redis_url: str = ""
    cache_ttl_seconds: int = 300

    # ---- Auth -----------------------------------------------------------
    jwt_secret_key: str = "change-me-this-is-not-a-real-secret"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # ---- Recommender ----------------------------------------------------
    reco_alpha: float = 0.6
    reco_min_user_ratings: int = 5
    reco_min_item_ratings: int = 3
    reco_candidate_target: int = 150
    reco_top_n: int = 20

    # ---- Model artifacts ------------------------------------------------
    model_dir: str = "models"
    # Empty means "use the newest version directory found on disk".
    model_version: str = ""

    # ---- App ------------------------------------------------------------
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"

    @property
    def model_root(self) -> Path:
        """Absolute path to the directory holding versioned model artifacts."""
        path = Path(self.model_dir)
        return path if path.is_absolute() else BACKEND_DIR / path

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor; safe to call from anywhere including FastAPI deps."""
    return Settings()
