"""
auth.py

Minimal authentication + role-based authorization for the Apple
Trade-In app.

DESIGN (deliberately simple — internship-project appropriate):

    - Users are stored in a single JSON file (data/users.json).
      No database is introduced. This mirrors how the rest of the
      app already treats data/master_msrp.csv as its source of
      truth.

    - Passwords are never stored in plaintext. Each password is
      hashed with PBKDF2-HMAC-SHA256 (Python's stdlib `hashlib`,
      no extra dependency) using a random per-user salt.

    - After login, the server issues a signed, HTTP-only session
      cookie (via `itsdangerous`). The cookie contains the
      username and role and is cryptographically signed with a
      server-side secret key, so the browser cannot forge or
      tamper with it. The cookie itself carries no password data.

    - Two FastAPI dependencies are exposed:

          get_current_user   -> the logged-in user, or None
          require_admin       -> the logged-in user if role is
                                  "admin", otherwise raises 401/403

      Every /admin/* route (and the /admin page route) must be
      wired up with `Depends(require_admin)`. This is enforced
      on the server, not just hidden in the frontend -- a request
      with no cookie, an invalid cookie, or a customer's cookie
      is rejected before any admin logic runs.

SECURITY NOTE FOR THIS PROJECT'S SCALE:
    This is intentionally NOT a production-grade auth system
    (no password reset flow, no rate limiting, no email
    verification, no refresh tokens). It is sized for an
    internship project with a small number of internal admin
    users and self-serve customer signups, while still enforcing
    real server-side authorization -- which was the actual
    requirement.
"""

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Optional

from fastapi import Cookie, HTTPException, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

USERS_FILE = BASE_DIR / "data" / "users.json"

# Signs/verifies the session cookie. In production this should be
# set via an environment variable so it survives restarts and
# isn't visible in source control. If unset, a random key is
# generated at process start -- this means existing sessions are
# invalidated on every restart, which is an acceptable tradeoff
# for a project this size (users just log in again).
SESSION_SECRET = os.getenv(
    "SESSION_SECRET",
    secrets.token_hex(32),
)

SESSION_COOKIE_NAME = "tradein_session"

# Session cookies are valid for 7 days.
SESSION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60

VALID_ROLES = {"customer", "admin"}

_serializer = URLSafeTimedSerializer(SESSION_SECRET)


# ============================================================
# PASSWORD HASHING (PBKDF2-HMAC-SHA256, stdlib only)
# ============================================================

_PBKDF2_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    """
    Returns a string of the form:
        pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
    """

    salt = secrets.token_hex(16)

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        _PBKDF2_ITERATIONS,
    )

    return (
        f"pbkdf2_sha256${_PBKDF2_ITERATIONS}$"
        f"{salt}${derived.hex()}"
    )


def verify_password(password: str, stored_hash: str) -> bool:

    try:

        algorithm, iterations_str, salt, hash_hex = (
            stored_hash.split("$")
        )

        if algorithm != "pbkdf2_sha256":
            return False

        iterations = int(iterations_str)

        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            iterations,
        )

        # Constant-time comparison to avoid timing attacks.
        return hmac.compare_digest(derived.hex(), hash_hex)

    except (ValueError, AttributeError):

        return False


# ============================================================
# USER STORE (data/users.json)
# ============================================================

def _load_users() -> dict:

    if not USERS_FILE.exists():
        return {}

    try:

        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    except (json.JSONDecodeError, OSError):

        return {}


def _save_users(users: dict) -> None:

    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def get_user(username: str) -> Optional[dict]:

    users = _load_users()

    return users.get(username.lower().strip())


def create_user(
    username: str,
    password: str,
    role: str = "customer",
) -> dict:
    """
    Raises ValueError if the username already exists or the role
    is invalid. Returns the created user record (without the
    password hash) on success.
    """

    username_key = username.lower().strip()

    if not username_key:
        raise ValueError("Username is required.")

    if not password or len(password) < 6:
        raise ValueError(
            "Password must be at least 6 characters."
        )

    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role: {role}")

    users = _load_users()

    if username_key in users:
        raise ValueError("Username is already taken.")

    users[username_key] = {
        "username": username_key,
        "password_hash": hash_password(password),
        "role": role,
    }

    _save_users(users)

    return {
        "username": username_key,
        "role": role,
    }


def authenticate(
    username: str,
    password: str,
) -> Optional[dict]:
    """
    Returns the user record (without password hash) if the
    credentials are valid, otherwise None.
    """

    user = get_user(username)

    if user is None:
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    return {
        "username": user["username"],
        "role": user["role"],
    }


def seed_default_admin() -> None:
    """
    Called once at app startup. If no users exist yet, create a
    single default admin account so the admin dashboard is
    reachable on a fresh deploy without manual DB/file editing.

    The password is read from an environment variable so it is
    never hardcoded in source. If the env var is not set, no
    default admin is created and an admin must sign up normally
    with role escalation handled separately by whoever controls
    the server (see /signup below -- signups always default to
    "customer"; promoting someone to admin is a deliberate,
    separate action, e.g. editing data/users.json directly).
    """

    users = _load_users()

    if users:
        return

    default_username = os.getenv("DEFAULT_ADMIN_USERNAME")
    default_password = os.getenv("DEFAULT_ADMIN_PASSWORD")

    if not default_username or not default_password:
        return

    try:

        create_user(
            default_username,
            default_password,
            role="admin",
        )

        print(
            "Seeded default admin account "
            f"'{default_username.lower().strip()}' from "
            "environment variables."
        )

    except ValueError as exc:

        print(f"Could not seed default admin: {exc}")


# ============================================================
# SESSION COOKIES
# ============================================================

def create_session_token(username: str, role: str) -> str:

    return _serializer.dumps(
        {"username": username, "role": role}
    )


def read_session_token(token: str) -> Optional[dict]:

    try:

        data = _serializer.loads(
            token,
            max_age=SESSION_MAX_AGE_SECONDS,
        )

        if (
            not isinstance(data, dict)
            or "username" not in data
            or "role" not in data
        ):
            return None

        return data

    except (BadSignature, SignatureExpired):

        return None


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class SignupRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


# ============================================================
# FASTAPI DEPENDENCIES
# ============================================================

def get_current_user(
    tradein_session: Optional[str] = Cookie(
        default=None,
        alias=SESSION_COOKIE_NAME,
    ),
) -> Optional[dict]:
    """
    Returns the logged-in user's {"username", "role"} dict, or
    None if there is no valid session. Does NOT raise -- routes
    that should work for both logged-in and anonymous users (e.g.
    the customer page) can depend on this directly.
    """

    if not tradein_session:
        return None

    return read_session_token(tradein_session)


def require_admin(
    tradein_session: Optional[str] = Cookie(
        default=None,
        alias=SESSION_COOKIE_NAME,
    ),
) -> dict:
    """
    FastAPI dependency that PROTECTS admin routes.

    - No cookie / invalid cookie / expired cookie -> 401.
    - Valid cookie but role != "admin"             -> 403.
    - Valid admin session                           -> returns
      the user dict so the route can use it if needed.

    This is enforced entirely on the backend. It does not matter
    whether the request came from the admin UI, a customer who
    edited the page, or someone hitting the endpoint directly
    with curl -- without a valid signed admin session cookie,
    every /admin/* route and the /admin page itself will reject
    the request before any business logic runs.
    """

    if not tradein_session:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please log in.",
        )

    user = read_session_token(tradein_session)

    if user is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session is invalid or has expired. Please log in again.",
        )

    if user.get("role") != "admin":

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access is required for this action.",
        )

    return user