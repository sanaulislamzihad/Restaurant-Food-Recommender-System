"""ORM models for the recommender.

Portability note: the schema deliberately avoids Postgres-only constructs so the
same migrations run on SQLite (local dev) and Postgres (docker-compose, CI).
Tag lists use the generic JSON type instead of ARRAY, and enums are stored as
VARCHAR + CHECK constraints (``native_enum=False``) rather than native Postgres
enums, which are painful to alter in later migrations.
"""

import enum
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from app.db.base import Base


class TZDateTime(TypeDecorator[datetime]):
    """A timestamp that stays timezone-aware across a SQLite round trip.

    SQLite has no native timestamp type. ``TZDateTime`` stores an
    ISO string and silently drops the offset, so values read back are naive and
    any client would interpret a UTC instant as local time. Postgres preserves
    the offset, so without this the two backends disagree about what a stored
    timestamp *means* - the worst kind of dialect difference, because nothing
    raises and the numbers merely drift by the reader's UTC offset.

    Everything is normalised to UTC on the way in and re-tagged as UTC on the
    way out. A naive value is assumed to be UTC rather than rejected: the
    impressions endpoint accepts client-supplied timestamps, and a 500 on
    fire-and-forget telemetry would be a worse outcome than the assumption.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def _utcnow() -> datetime:
    """Timezone-aware UTC now.

    Applied Python-side rather than via ``server_default`` because SQLite and
    Postgres disagree on how CURRENT_TIMESTAMP handles timezones.
    """
    return datetime.now(UTC)


class Gender(enum.StrEnum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class OrderStatus(enum.StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PREPARING = "preparing"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class Restaurant(Base):
    __tablename__ = "restaurants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    area: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    # e.g. ["bengali", "mughlai"] - JSON keeps this portable across dialects.
    cuisine_tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    food_items: Mapped[list["FoodItem"]] = relationship(
        back_populates="restaurant", cascade="all, delete-orphan"
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    # Not in the original data-model sketch, but auth needs somewhere to put the
    # bcrypt digest. Seeded users all share a known development password.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[Gender | None] = mapped_column(
        Enum(Gender, native_enum=False, length=16, validate_strings=True), nullable=True
    )
    area: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    # 0 = cannot handle any heat, 5 = wants it as hot as the kitchen can make it.
    spice_tolerance: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False, default=_utcnow)

    ratings: Mapped[list["Rating"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    orders: Mapped[list["Order"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("spice_tolerance BETWEEN 0 AND 5", name="ck_users_spice_tolerance"),
        CheckConstraint("age IS NULL OR age BETWEEN 10 AND 120", name="ck_users_age"),
    )


class FoodItem(Base):
    __tablename__ = "food_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    restaurant_id: Mapped[int] = mapped_column(
        ForeignKey("restaurants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    cuisine: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    spice_level: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_veg: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Rice-based dishes form their own taste cluster on a Bangladeshi menu, so
    # this earns a dedicated feature instead of hiding inside ingredient_tags.
    is_rice_based: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Numeric keeps money exact. SQLAlchemy warns that SQLite has no native
    # Decimal type; that is cosmetic, and the Postgres target stores it properly.
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    prep_time_min: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    image_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    ingredient_tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    is_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    # The ranking stage boosts promoted dishes. Without a column the rule would
    # be written but permanently inert, which is worse than not having it.
    is_promoted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False, default=_utcnow)

    restaurant: Mapped["Restaurant"] = relationship(back_populates="food_items")
    ratings: Mapped[list["Rating"]] = relationship(
        back_populates="food_item", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("spice_level BETWEEN 0 AND 5", name="ck_food_items_spice_level"),
        CheckConstraint("price >= 0", name="ck_food_items_price"),
        Index("ix_food_items_cuisine_available", "cuisine", "is_available"),
    )


class Rating(Base):
    __tablename__ = "ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    food_item_id: Mapped[int] = mapped_column(
        ForeignKey("food_items.id", ondelete="CASCADE"), nullable=False
    )
    rating: Mapped[Decimal] = mapped_column(Numeric(2, 1), nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False, default=_utcnow)

    user: Mapped["User"] = relationship(back_populates="ratings")
    food_item: Mapped["FoodItem"] = relationship(back_populates="ratings")

    __table_args__ = (
        # One rating per user per item; re-rating updates the existing row.
        UniqueConstraint("user_id", "food_item_id", name="uq_ratings_user_item"),
        CheckConstraint("rating BETWEEN 1 AND 5", name="ck_ratings_range"),
        # The evaluation harness splits chronologically per user, so this index
        # carries the "ratings for user X ordered by time" query.
        Index("ix_ratings_user_created", "user_id", "created_at"),
        Index("ix_ratings_item", "food_item_id"),
    )


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False, default=0)
    status: Mapped[OrderStatus] = mapped_column(
        Enum(OrderStatus, native_enum=False, length=16, validate_strings=True),
        nullable=False,
        default=OrderStatus.PENDING,
    )
    created_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False, default=_utcnow)

    user: Mapped["User"] = relationship(back_populates="orders")
    items: Mapped[list["OrderItem"]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # "What did this user order recently" powers both the retrieval stage and
        # the business rule that suppresses items ordered in the last 24 hours.
        Index("ix_orders_user_created", "user_id", "created_at"),
    )


class OrderItem(Base):
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True
    )
    food_item_id: Mapped[int] = mapped_column(
        ForeignKey("food_items.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # Captured at order time: the menu price may change afterwards.
    unit_price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)

    order: Mapped["Order"] = relationship(back_populates="items")

    __table_args__ = (CheckConstraint("quantity > 0", name="ck_order_items_quantity"),)


class Impression(Base):
    """Every item shown to a user, and whether it converted.

    This table is what makes implicit-feedback training possible: without logged
    non-conversions there is only ``y = 1`` data, and the model cannot learn what
    people actively reject.
    """

    __tablename__ = "impressions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    food_item_id: Mapped[int] = mapped_column(
        ForeignKey("food_items.id", ondelete="CASCADE"), nullable=False
    )
    was_ordered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    shown_at: Mapped[datetime] = mapped_column(TZDateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_impressions_user_shown", "user_id", "shown_at"),
        Index("ix_impressions_item", "food_item_id"),
    )


class ItemNeighbor(Base):
    """Precomputed nearest neighbours by squared distance on learned features.

    Populated offline after collaborative-filtering training so that
    "you liked X, try these" is a single indexed lookup rather than an O(n_m)
    distance computation inside a request handler.
    """

    __tablename__ = "item_neighbors"

    food_item_id: Mapped[int] = mapped_column(
        ForeignKey("food_items.id", ondelete="CASCADE"), primary_key=True
    )
    neighbor_id: Mapped[int] = mapped_column(
        ForeignKey("food_items.id", ondelete="CASCADE"), primary_key=True
    )
    distance: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_item_neighbors_item_rank", "food_item_id", "rank"),
        CheckConstraint("food_item_id <> neighbor_id", name="ck_item_neighbors_not_self"),
    )
