"""Authentication payloads."""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from app.core.security import MAX_PASSWORD_BYTES, MIN_PASSWORD_LENGTH
from app.db.models import Gender
from app.schemas.common import ORMModel


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    # The upper bound is bcrypt's, not a policy choice: it truncates silently
    # past 72 bytes, so anything longer would authenticate on its prefix.
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_BYTES)
    age: int | None = Field(default=None, ge=10, le=120)
    gender: Gender | None = None
    area: str | None = Field(default=None, max_length=80)
    spice_tolerance: int = Field(default=2, ge=0, le=5)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


class UserResponse(ORMModel):
    id: int
    name: str
    email: str
    age: int | None
    gender: Gender | None
    area: str | None
    spice_tolerance: int
    is_admin: bool
    created_at: datetime


class TasteProfile(BaseModel):
    """Summary of what a user's history says about them.

    Powers the profile page, and doubles as the explanation for why the feed
    looks the way it does.
    """

    total_ratings: int
    total_orders: int
    average_rating_given: float | None
    top_cuisines: list["CuisineAffinity"]
    is_cold_start: bool = Field(
        description="true when the user has too little history for collaborative filtering"
    )


class CuisineAffinity(BaseModel):
    cuisine: str
    order_count: int
    average_rating: float | None
