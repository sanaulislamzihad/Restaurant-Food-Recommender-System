"""Declarative base shared by every ORM model."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models.

    Kept in its own module so Alembic can import the metadata without pulling in
    the FastAPI application and its dependencies.
    """
