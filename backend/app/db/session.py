"""Engine and session factory.

The dialect is driven entirely by ``DATABASE_URL``: SQLite for zero-service local
development, Postgres in docker-compose and CI. Nothing else in the codebase
should branch on the database backend.
"""

from collections.abc import Generator, Iterator
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def _build_engine() -> Engine:
    settings = get_settings()
    kwargs: dict[str, Any] = {"echo": False, "future": True}

    if settings.is_sqlite:
        # check_same_thread=False lets FastAPI's threadpool share the connection;
        # SQLAlchemy's own pooling still serialises access.
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_pre_ping"] = True
        kwargs["pool_size"] = 10
        kwargs["max_overflow"] = 20

    engine = create_engine(settings.database_url, **kwargs)

    if settings.is_sqlite:

        @event.listens_for(engine, "connect")
        def _enable_sqlite_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
            """SQLite ignores foreign keys unless asked, which would silently hide
            referential bugs that Postgres would catch in production."""
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


engine: Engine = _build_engine()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def session_scope() -> Iterator[Session]:
    """Context-manager style session for scripts and offline jobs.

    Usage::

        with contextlib.closing(next(session_scope())) as db: ...

    Prefer ``SessionLocal()`` directly in scripts; this exists for symmetry.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
