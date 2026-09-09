"""Registration and login."""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select

from app.api.deps import CurrentUser, DbSession
from app.core.config import get_settings
from app.core.security import (
    PasswordTooLongError,
    create_access_token,
    hash_password,
    verify_password,
)
from app.db.models import FoodItem, Order, OrderItem, Rating, User
from app.schemas.auth import (
    CuisineAffinity,
    LoginRequest,
    RegisterRequest,
    TasteProfile,
    TokenResponse,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: DbSession) -> TokenResponse:
    existing = db.scalar(select(User).where(func.lower(User.email) == payload.email.lower()))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="That email is already registered."
        )

    try:
        password_hash = hash_password(payload.password)
    except PasswordTooLongError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error

    user = User(
        name=payload.name,
        email=payload.email.lower(),
        password_hash=password_hash,
        age=payload.age,
        gender=payload.gender,
        area=payload.area,
        spice_tolerance=payload.spice_tolerance,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    settings = get_settings()
    return TokenResponse(
        access_token=create_access_token(user.id),
        expires_in_minutes=settings.jwt_expire_minutes,
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: DbSession) -> TokenResponse:
    user = db.scalar(select(User).where(func.lower(User.email) == payload.email.lower()))

    # The password is verified even when the account does not exist, against a
    # dummy hash, so that a missing account and a wrong password take the same
    # time. Returning early on an unknown email turns login into an oracle for
    # which addresses are registered.
    stored_hash = user.password_hash if user else _DUMMY_HASH
    password_ok = verify_password(payload.password, stored_hash)

    if user is None or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password."
        )

    settings = get_settings()
    return TokenResponse(
        access_token=create_access_token(user.id),
        expires_in_minutes=settings.jwt_expire_minutes,
    )


@router.get("/me", response_model=UserResponse)
def read_me(user: CurrentUser) -> User:
    return user


@router.get("/me/taste-profile", response_model=TasteProfile)
def taste_profile(user: CurrentUser, db: DbSession) -> TasteProfile:
    """What the user's history says about them.

    Shown on the profile page, and the honest answer to "why am I being shown
    this" - if the profile is empty, the feed is popularity, not personalisation.
    """
    settings = get_settings()

    total_ratings = (
        db.scalar(select(func.count()).select_from(Rating).where(Rating.user_id == user.id)) or 0
    )
    total_orders = (
        db.scalar(select(func.count()).select_from(Order).where(Order.user_id == user.id)) or 0
    )
    average = db.scalar(select(func.avg(Rating.rating)).where(Rating.user_id == user.id))

    rows = db.execute(
        select(
            FoodItem.cuisine,
            func.count(OrderItem.id),
            func.avg(Rating.rating),
        )
        .select_from(OrderItem)
        .join(Order, Order.id == OrderItem.order_id)
        .join(FoodItem, FoodItem.id == OrderItem.food_item_id)
        .outerjoin(
            Rating,
            (Rating.food_item_id == FoodItem.id) & (Rating.user_id == user.id),
        )
        .where(Order.user_id == user.id)
        .group_by(FoodItem.cuisine)
        .order_by(func.count(OrderItem.id).desc())
        .limit(5)
    ).all()

    return TasteProfile(
        total_ratings=total_ratings,
        total_orders=total_orders,
        average_rating_given=float(average) if average is not None else None,
        top_cuisines=[
            CuisineAffinity(
                cuisine=cuisine,
                order_count=int(count),
                average_rating=float(avg) if avg is not None else None,
            )
            for cuisine, count, avg in rows
        ],
        is_cold_start=total_ratings < settings.reco_min_user_ratings,
    )


#: A real bcrypt digest of a value nobody can log in with. Used only so that the
#: unknown-email path does the same work as the known-email path.
_DUMMY_HASH = hash_password("not-a-real-password-placeholder")
