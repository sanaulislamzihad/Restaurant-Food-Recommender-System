"""Shared FastAPI dependencies."""

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.models import User
from app.db.session import SessionLocal

# auto_error=False so the optional-auth dependency can see a missing header and
# return None instead of the framework raising a 403 before it runs.
bearer_scheme = HTTPBearer(auto_error=False)

CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DbSession = Annotated[Session, Depends(get_db)]
Credentials = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)]


def get_current_user(credentials: Credentials, db: DbSession) -> User:
    """The authenticated user, or 401.

    Every failure - no header, bad signature, expired, deleted account -
    produces the same generic 401. Distinguishing "no such user" from "wrong
    password" here would turn the endpoint into an account-existence oracle.
    """
    if credentials is None:
        raise CREDENTIALS_ERROR

    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise CREDENTIALS_ERROR

    user = db.get(User, user_id)
    if user is None:
        raise CREDENTIALS_ERROR
    return user


def get_optional_user(credentials: Credentials, db: DbSession) -> User | None:
    """The authenticated user if there is one, else None.

    For endpoints that work for anonymous visitors but personalise when they
    can - the home page being the obvious one.
    """
    if credentials is None:
        return None
    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        return None
    return db.get(User, user_id)


CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[User | None, Depends(get_optional_user)]
