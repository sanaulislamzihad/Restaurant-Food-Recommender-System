"""Shared response shapes."""

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    """Base for schemas built directly from ORM rows."""

    model_config = ConfigDict(from_attributes=True)


class Page(BaseModel, Generic[T]):
    """One page of results.

    ``total`` is the count matching the filters, not the page size, so a client
    can render "showing 20 of 137" and size its pagination control without
    walking every page to find the end.
    """

    items: list[T]
    total: int = Field(description="total matching rows, ignoring limit/offset")
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total


class Message(BaseModel):
    detail: str
