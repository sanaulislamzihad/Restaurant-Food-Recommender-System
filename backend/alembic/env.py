"""Alembic environment.

The database URL is taken from application settings (``DATABASE_URL``) rather
than from alembic.ini, so migrations always target the same database the app
does — SQLite locally, Postgres in docker-compose and CI.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import get_settings
from app.db import models  # noqa: F401  - imported for its side effect of registering mappers
from app.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Autogenerate compares the live database against this metadata.
target_metadata = Base.metadata

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)


def _configure_common() -> dict[str, object]:
    """Options shared by the offline and online paths.

    ``render_as_batch`` matters: SQLite cannot ALTER most column properties, so
    Alembic must rebuild the table instead. Without it, any future migration that
    alters a column would run on Postgres and fail on SQLite.
    """
    return {
        "target_metadata": target_metadata,
        "compare_type": True,
        "compare_server_default": True,
        "render_as_batch": settings.is_sqlite,
    }


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a database."""
    context.configure(
        url=settings.database_url,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **_configure_common(),  # type: ignore[arg-type]
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and run migrations against the live database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, **_configure_common())  # type: ignore[arg-type]

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
