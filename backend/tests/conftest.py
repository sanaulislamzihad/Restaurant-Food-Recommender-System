"""Test fixtures.

The database URL and model directory are redirected to a throwaway location
*before* any app module is imported, because ``app.db.session`` builds its
engine at import time and ``get_settings`` is cached. Environment variables take
precedence over the repo ``.env``, so this reliably isolates tests from the
development database and from the real trained artifacts - a model trained on
the 302-item seed would have no item ids in common with a test fixture, and
would fail in confusing ways rather than obvious ones.
"""

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

_TEST_DIR = Path(tempfile.mkdtemp(prefix="foodrec-tests-"))
os.environ["DATABASE_URL"] = f"sqlite+pysqlite:///{(_TEST_DIR / 'test.db').as_posix()}"
os.environ["REDIS_URL"] = ""
os.environ["MODEL_DIR"] = str(_TEST_DIR / "models")
os.environ["JWT_SECRET_KEY"] = "test-secret-not-used-anywhere-real"

import app.db.models  # noqa: E402,F401  - registers the mappers
from app.core.cache import reset_cache  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_process_state() -> Iterator[None]:
    """Reset the module-level singletons between tests.

    The cache and the loaded model are deliberately process-wide in production.
    That makes them shared mutable state in a test suite, so one test's cached
    feed could otherwise be served to the next.
    """
    from app.services import recommender

    def _clean() -> None:
        reset_cache()
        recommender._holder = recommender._ModelHolder()
        # The model directory is shared for the whole session, so without this
        # a version saved by one test leaks into the next and 'v1' becomes 'v7'.
        shutil.rmtree(_TEST_DIR / "models", ignore_errors=True)

    _clean()
    yield
    _clean()


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


@pytest.fixture
def model_root() -> Path:
    """The isolated model directory these tests read and write."""
    path = _TEST_DIR / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path
