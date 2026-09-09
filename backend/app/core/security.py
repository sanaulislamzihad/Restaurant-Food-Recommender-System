"""Password hashing and JWT access tokens."""

from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from app.core.config import get_settings

#: bcrypt silently truncates at 72 bytes. Rejecting longer input is safer than
#: accepting a password whose tail never mattered, which would let a user who
#: typed 100 characters authenticate with only the first 72.
MAX_PASSWORD_BYTES = 72

MIN_PASSWORD_LENGTH = 8


class PasswordTooLongError(ValueError):
    pass


def hash_password(password: str) -> str:
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise PasswordTooLongError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes when UTF-8 encoded."
        )
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time check of a password against a stored digest.

    Returns False rather than raising on a malformed digest: a corrupted row
    should fail the login, not return a 500 that tells an attacker the account
    exists.
    """
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(encoded, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: int, *, expires_minutes: int | None = None) -> str:
    """Signed JWT carrying the user id as its subject.

    ``sub`` is stringified because the JWT spec requires it to be a string, and
    some libraries reject an integer subject outright.
    """
    settings = get_settings()
    now = datetime.now(UTC)
    lifetime = expires_minutes if expires_minutes is not None else settings.jwt_expire_minutes

    payload: dict[str, Any] = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=lifetime),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> int | None:
    """Return the user id from a valid token, or None if it is not usable.

    Every failure mode - bad signature, expired, malformed, missing or
    non-numeric subject - collapses to None. The caller turns that into one
    generic 401; distinguishing them for the client would leak whether a token
    was forged or merely stale.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        return None

    subject = payload.get("sub")
    if subject is None:
        return None
    try:
        return int(subject)
    except (TypeError, ValueError):
        return None
