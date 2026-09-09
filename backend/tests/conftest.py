"""Test fixtures.

The database URL is redirected to a throwaway SQLite file *before* any app
module is imported, because ``app.db.session`` builds its engine at import time
and ``get_settings`` is cached. Environment variables take precedence over the
repo ``.env``, so this reliably isolates the test database from the development
one.
"""

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

_TEST_DIR = Path(tempfile.mkdtemp(prefix="foodrec-tests-"))
os.environ["DATABASE_URL"] = f"sqlite+pysqlite:///{(_TEST_DIR / 'test.db').as_posix()}"
os.environ["REDIS_URL"] = ""

import app.db.models  # noqa: E402,F401  - registers the mappers
from app.db.base import Base  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402


@pytest.fixture
def db_session() -> Iterator["object"]:
    """A session against a freshly created, empty schema."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def empty_database() -> Iterator[None]:
    """Fresh schema with no session held open, for code that opens its own."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def tmp_meta_path() -> Path:
    """Sidecar path for seed runs inside tests."""
    return _TEST_DIR / "seed_meta.json"
