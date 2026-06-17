"""JWT and authentication utilities for AssuranceIA™."""
import os
import bcrypt
import json
import logging
from datetime import datetime, timedelta
import jwt

logger = logging.getLogger("assurancia")


# ============================================================================
# HARDCODED DEMO USERS - GUARANTEED FALLBACK
# ----------------------------------------------------------------------------
# These credentials ALWAYS work, even if MongoDB is down and even if the
# DEMO_USERS environment variable is missing or misconfigured.
#
# Plaintext passwords are stored here (demo accounts only) and hashed at
# runtime, so login is guaranteed to work end-to-end. This is the source of
# truth for the demo logins:
#     kevin / password123
#     demo  / demo123
#     assurance / pass456
# ============================================================================
HARDCODED_DEMO_USERS = [
    {"username": "kevin", "password": "password123", "insurer_id": "INSURER_TEST_001"},
    {"username": "demo", "password": "demo123", "insurer_id": "DEMO_INSURER"},
    {"username": "assurance", "password": "pass456", "insurer_id": "INSURER_TEST_002"},
]


def validate_jwt_secret():
    """Validate JWT_SECRET at startup.

    Raises RuntimeError if JWT_SECRET is not set or too weak.
    """
    jwt_secret = os.getenv("JWT_SECRET", "").strip()

    if not jwt_secret:
        raise RuntimeError(
            "CRITICAL: JWT_SECRET environment variable is not set. "
            "Set it to a random string >= 32 characters before deployment."
        )

    if len(jwt_secret) < 32:
        raise RuntimeError(
            f"CRITICAL: JWT_SECRET is too weak ({len(jwt_secret)} chars). "
            "Minimum 32 characters required. Use a strong random string."
        )

    logger.info("✅ JWT_SECRET validated (%d characters)", len(jwt_secret))


def _build_hardcoded_users() -> dict:
    """Build the hardcoded demo users dict with freshly-computed bcrypt hashes.

    This NEVER touches the network or environment, so it cannot fail. It is the
    last-resort guarantee that login works.
    """
    users = {}
    for u in HARDCODED_DEMO_USERS:
        users[u["username"]] = {
            "password_hash": hash_password(u["password"]),
            "insurer_id": u.get("insurer_id", "DEMO_INSURER"),
        }
    return users


def _users_from_env() -> dict:
    """Parse demo users from the DEMO_USERS environment variable (JSON list).

    Tolerant of BOTH key styles:
      - {"username": "...", "password": "plaintext", ...}      -> hashed at runtime
      - {"username": "...", "password_hash": "$2b$...", ...}   -> used as-is

    Returns {} if the variable is missing or invalid.
    """
    demo_users_json = os.getenv("DEMO_USERS", "").strip()
    if not demo_users_json:
        return {}

    try:
        users_list = json.loads(demo_users_json)
        users_dict = {}
        for user in users_list:
            username = user.get("username")
            if not username:
                continue

            # Prefer an explicit hash; otherwise hash the plaintext password.
            password_hash = user.get("password_hash", "").strip()
            if not password_hash:
                plaintext = user.get("password", "")
                if plaintext:
                    password_hash = hash_password(plaintext)

            if not password_hash:
                # Nothing usable for this user; skip it.
                continue

            users_dict[username] = {
                "password_hash": password_hash,
                "insurer_id": user.get("insurer_id", "UNKNOWN"),
            }
        return users_dict
    except json.JSONDecodeError as e:
        logger.error("❌ Failed to parse DEMO_USERS JSON: %s", e)
        return {}
    except Exception as e:
        logger.error("❌ Failed to load demo users from env: %s", e)
        return {}


def load_demo_users() -> dict:
    """Load demo users with a guaranteed, never-empty result.

    Strategy (merge, hardcoded wins so the known demo logins ALWAYS work):
      1. Start with the hardcoded demo users (always present, hashed at runtime).
      2. Layer in any extra users from the DEMO_USERS env var.

    The result ALWAYS contains kevin/demo/assurance, regardless of MongoDB or
    environment configuration. This eliminates the previous failure mode where a
    MongoDB outage or a malformed env var made login impossible.
    """
    users = _build_hardcoded_users()

    # Layer env users on top, but never let them remove the hardcoded demos.
    env_users = _users_from_env()
    for username, data in env_users.items():
        if username not in users:
            users[username] = data

    logger.info(
        "✅ Loaded %d demo user(s) (hardcoded fallback active): %s",
        len(users), ", ".join(sorted(users.keys())),
    )
    return users


def hash_password(password: str) -> str:
    """Hash a password using bcrypt (12 rounds)."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception as e:
        logger.error("Password verification failed: %s", e)
        return False


def generate_jwt_token(insurer_id: str, expires_in_hours: int = 24) -> str:
    """Generate a JWT token for an insurer."""
    jwt_secret = os.getenv("JWT_SECRET", "").strip()
    if not jwt_secret:
        raise RuntimeError("JWT_SECRET not configured")

    now = datetime.now()
    payload = {
        "insurer_id": insurer_id,
        "iat": now,
        "exp": now + timedelta(hours=expires_in_hours),
    }
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return token
