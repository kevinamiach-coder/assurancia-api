"""JWT and authentication utilities for AssuranceIA™."""
import os
import bcrypt
import json
import logging
from datetime import datetime, timedelta
import jwt

logger = logging.getLogger("assurancia")


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


def load_demo_users() -> dict:
    """Load demo users from DEMO_USERS environment variable (JSON format)."""
    demo_users_json = os.getenv("DEMO_USERS", "").strip()

    if not demo_users_json:
        logger.warning("⚠️ DEMO_USERS environment variable not set. Demo users unavailable.")
        return {}

    try:
        users_list = json.loads(demo_users_json)
        users_dict = {}
        for user in users_list:
            username = user.get("username")
            if username:
                users_dict[username] = {
                    "password_hash": user.get("password_hash", ""),
                    "insurer_id": user.get("insurer_id", "UNKNOWN")
                }
        logger.info("✅ Loaded %d demo user(s) from DEMO_USERS", len(users_dict))
        return users_dict
    except json.JSONDecodeError as e:
        logger.error("❌ Failed to parse DEMO_USERS JSON: %s", e)
        return {}
    except Exception as e:
        logger.error("❌ Failed to load demo users: %s", e)
        return {}


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
