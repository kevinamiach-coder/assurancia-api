from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, RedirectResponse
from pydantic import BaseModel
from fpdf import FPDF
from anthropic import Anthropic
import piexif
from PIL import Image
from io import BytesIO
import hashlib
import uuid
from datetime import datetime, timedelta
import secrets
import os
import base64
import json
import re
import requests
from pymongo import MongoClient
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
import jwt

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request

# Phase 1 Security Modules
from auth import validate_jwt_secret, load_demo_users, hash_password, verify_password, generate_jwt_token
from input_sanitizer import sanitize_email, sanitize_phone, sanitize_address, sanitize_text, is_safe
from audit_logger import audit_log, audit_authenticate, audit_create_declaration_link, audit_submit_declaration, audit_view_claim

app = FastAPI(title="AssuranceIA API", version="2.0")

# ========== STARTUP: PHASE 1 SECURITY VALIDATION ==========
@app.on_event("startup")
async def validate_startup_security():
    """Validate Phase 1 security configuration at startup."""
    logger.info("🔐 Starting Phase 1 Security Validation...")

    # 1. Validate JWT_SECRET
    try:
        validate_jwt_secret()
    except RuntimeError as e:
        logger.critical("FATAL: %s", e)
        raise

    # 2. Validate MONGODB_URI
    mongodb_uri = os.getenv("MONGODB_URI", "").strip()
    if not mongodb_uri:
        logger.critical("FATAL: MONGODB_URI environment variable is not set.")
        raise RuntimeError("MONGODB_URI is required for production deployment")
    logger.info("✅ MONGODB_URI configured")

    # 3. Validate ANTHROPIC_API_KEY
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip().strip('"').strip("'")
    if anthropic_key:
        logger.info("✅ ANTHROPIC_API_KEY configured (%d chars)", len(anthropic_key))
    else:
        logger.warning("⚠️  ANTHROPIC_API_KEY not set - Vision analysis disabled")

    # 4. Load DEMO_USERS from environment
    demo_users = load_demo_users()
    if demo_users:
        logger.info("✅ Loaded %d demo user(s) from DEMO_USERS", len(demo_users))
    else:
        logger.warning("⚠️  DEMO_USERS not configured - authentication may not work")

    logger.info("✅ Phase 1 Security Validation PASSED")

# Rate Limiting - use x-forwarded-for for proper client IP behind proxy
def get_client_ip(request: Request) -> str:
    """Get real client IP, accounting for reverse proxies (Render, CloudFlare, etc)"""
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else get_remote_address(request)

limiter = Limiter(key_func=get_client_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# JWT Security
security = HTTPBearer()

def verify_insurer_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token for insurer routes. Returns insurer_id if valid."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, os.getenv("JWT_SECRET"), algorithms=["HS256"])
        insurer_id = payload.get("insurer_id")
        if not insurer_id:
            raise HTTPException(status_code=401, detail="Invalid token: no insurer_id")
        return insurer_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ANTHROPIC_API_KEY = (os.getenv("ANTHROPIC_API_KEY") or "").strip().strip('"').strip("'")
CLAUDE_MODEL = "claude-opus-4-6"  # Vision model - using Opus for better analysis

# Initialize Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("assurancia")

# MongoDB Connection
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI environment variable is required. Set it in .env or on your deployment platform.")

# serverSelectionTimeoutMS keeps the app responsive if Mongo is unreachable
# instead of hanging for ~30s on every request.
mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
db = mongo_client["assurancia"]
declaration_links_collection = db["declaration_links"]
claims_collection = db["claims"]
token_to_claim_collection = db["token_to_claim"]

# Set to True when a real connection to MongoDB has been verified at startup.
MONGO_AVAILABLE = False

# In-memory storage (fallback for local dev). Production uses MongoDB above.
claims_db: dict = {}
token_to_claim: dict = {}  # Mapping: unique_token -> claim_id
declaration_links: dict = {}  # Declaration templates: token -> {insurer_email, client_email, created_at}

# Anthropic Vision API only accepts these media types.
SUPPORTED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_PHOTO_BYTES = 5 * 1024 * 1024  # 5 MB per photo (keeps memory safe on free tier)

# Display timezone for human-readable timestamps (Kevin's location = GMT+3).
DISPLAY_TZ_OFFSET_HOURS = 3
DISPLAY_TZ_LABEL = "GMT+3"

_FRENCH_MONTHS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


def format_french_datetime(iso_value, with_timezone: bool = True) -> str:
    """Format an ISO timestamp as a readable French date/time.

    Example: "10 juin 2026 à 13:45:32 (GMT+3)".

    ISO timestamps are stored as naive local time (datetime.now().isoformat()),
    so we display them as-is and simply append the configured timezone label.
    Returns "N/A" for anything unparseable (defensive against None / bad data).
    """
    if not iso_value:
        return "N/A"
    try:
        dt = datetime.fromisoformat(str(iso_value))
    except (ValueError, TypeError):
        # Fall back to a best-effort slice so we never crash the page.
        return str(iso_value)[:19].replace("T", " ")

    month = _FRENCH_MONTHS[dt.month - 1]
    base = f"{dt.day} {month} {dt.year} à {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"
    if with_timezone:
        base += f" ({DISPLAY_TZ_LABEL})"
    return base

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== STARTUP: Load declaration links from MongoDB ==========
@app.on_event("startup")
async def load_declaration_links_from_mongodb():
    """Verify MongoDB connection and load declaration links on app startup."""
    global declaration_links, MONGO_AVAILABLE

    # 1. Verify the connection with a ping (raises if Mongo is unreachable / SSL fails).
    try:
        mongo_client.admin.command("ping")
        MONGO_AVAILABLE = True
        logger.info("✅ MongoDB connection verified (ping OK).")
    except Exception as e:
        MONGO_AVAILABLE = False
        logger.error(f"❌ MongoDB connection FAILED at startup: {e!r}")
        logger.error("   → App will run in IN-MEMORY fallback mode. Data will NOT persist across restarts.")

    # 2. Load existing declaration links (only if Mongo is available).
    if not MONGO_AVAILABLE:
        return
    try:
        count = 0
        for doc in declaration_links_collection.find({}, {"_id": 0}):
            token = doc.get("token")
            if token:
                declaration_links[token] = {
                    "insurer_email": doc.get("insurer_email", ""),
                    "client_email": doc.get("client_email", ""),
                    "created_at": doc.get("created_at", ""),
                    "status": doc.get("status", "pending")
                }
                count += 1
        logger.info(f"✅ Loaded {count} declaration link(s) from MongoDB.")
    except Exception as e:
        logger.warning(f"⚠️  Could not load declaration links from MongoDB: {e!r}")


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class ClaimCreate(BaseModel):
    user_email: str
    damage_type: str
    address: str
    description: str
    phone_gps_lat: float | None = None  # GPS téléphone (latitude)
    phone_gps_lon: float | None = None  # GPS téléphone (longitude)


class DeclarationLinkRequest(BaseModel):
    insurer_email: str  # Email de l'assurance/courtier
    client_email: str | None = None  # Optional pre-filled client email


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def normalize_media_type(content_type: str | None) -> str | None:
    """Map common browser content types to Anthropic-supported media types."""
    if not content_type:
        return None
    ct = content_type.lower().strip()
    if ct == "image/jpg":
        ct = "image/jpeg"
    if ct in SUPPORTED_MEDIA_TYPES:
        return ct
    return None


def geocode_address(address: str) -> dict | None:
    """Geocode a French/EU address to lat/lon using the free OSM Nominatim API.

    Returns {"latitude": float, "longitude": float, "display_name": str} or None.
    """
    if not address or not address.strip():
        return None
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 1, "countrycodes": "fr"},
            headers={"User-Agent": "AssuranceIA/2.0 (claims-geocoding)"},
            timeout=10,
        )
        if resp.status_code == 200:
            results = resp.json()
            if results:
                r = results[0]
                return {
                    "latitude": float(r["lat"]),
                    "longitude": float(r["lon"]),
                    "display_name": r.get("display_name", address),
                }
    except Exception as e:  # geocoding is best-effort, never block claim creation
        print(f"Geocoding failed for '{address}': {e}")
    return None


def extract_json(text: str) -> dict | None:
    """Robustly extract a JSON object from a Claude text response."""
    if not text:
        return None
    candidate = text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present.
    if "```" in candidate:
        # Take content between the first pair of fences.
        parts = candidate.split("```")
        if len(parts) >= 3:
            block = parts[1]
            if block.lstrip().lower().startswith("json"):
                block = block.lstrip()[4:]
            candidate = block.strip()

    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Fallback: grab the first {...} balanced-ish block via regex.
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def extract_exif_datetime(image_data: bytes) -> dict | None:
    """Extract date/time from image EXIF metadata."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(BytesIO(image_data))
        exif_data = img._getexif()

        if not exif_data:
            return None

        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            # DateTime tags: 306 = DateTime, 36867 = DateTimeOriginal, 36868 = DateTimeDigitized
            if tag_name in ["DateTime", "DateTimeOriginal", "DateTimeDigitized"]:
                try:
                    # Format: "2026:06:08 14:30:45"
                    dt = datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
                    return {"datetime": dt, "timestamp": dt.isoformat()}
                except:
                    pass
        return None
    except Exception as e:
        print(f"EXIF datetime extraction error: {e}")
        return None


def extract_gps_from_exif(image_data: bytes) -> dict | None:
    """Extract GPS coordinates from image EXIF metadata."""
    try:
        img = Image.open(BytesIO(image_data))
        exif_dict = piexif.load(img.info.get("exif", b""))

        # Extract GPS IFD
        gps_ifd = exif_dict.get("GPS", {})
        if not gps_ifd:
            return None

        # Parse latitude
        lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef, [b"N"])[0]
        lat_data = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
        if not lat_data:
            return None

        # Parse longitude
        lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef, [b"E"])[0]
        lon_data = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
        if not lon_data:
            return None

        # Convert to decimal degrees
        def dms_to_decimal(dms_tuple, ref):
            degrees = dms_tuple[0][0] / dms_tuple[0][1]
            minutes = dms_tuple[1][0] / dms_tuple[1][1] / 60.0
            seconds = dms_tuple[2][0] / dms_tuple[2][1] / 3600.0
            decimal = degrees + minutes + seconds
            if ref in [b"S", b"W"]:
                decimal = -decimal
            return decimal

        latitude = dms_to_decimal(lat_data, lat_ref)
        longitude = dms_to_decimal(lon_data, lon_ref)

        return {"latitude": latitude, "longitude": longitude}
    except Exception as e:
        print(f"EXIF extraction error: {e}")
        return None


def calculate_image_hash(image_data: bytes) -> str:
    """Calculate MD5 hash of image for duplicate detection."""
    return hashlib.md5(image_data).hexdigest()


def check_image_duplicates(image_hash: str, claim_id: str) -> dict:
    """Check if same image has been used in other claims (reverse image search)."""
    duplicate_claims = []

    for cid, claim in claims_db.items():
        if cid == claim_id:
            continue
        for photo in claim.get("photos", []):
            # We'll store hash with photo if detected, compare
            if photo.get("image_hash") == image_hash:
                duplicate_claims.append(cid)

    result = {
        "is_duplicate": len(duplicate_claims) > 0,
        "duplicate_count": len(duplicate_claims),
        "duplicate_claims": duplicate_claims,
        "fraud_indicator": None
    }

    if len(duplicate_claims) > 0:
        result["fraud_indicator"] = f"🚨 IMAGE RECYCLÉE: Même photo détectée dans {len(duplicate_claims)} autre(s) sinistre(s)!"

    return result


def check_fraud_history(user_email: str, address: str) -> dict:
    """Check claim history to detect fraud networks."""
    email_claims = [c for c in claims_db.values() if c.get("user_email", "").lower() == user_email.lower()]
    address_claims = [c for c in claims_db.values() if c.get("address", "").lower() == address.lower()]

    fraud_flags = {
        "email_claim_count": len(email_claims),
        "address_claim_count": len(address_claims),
        "fraud_indicators": []
    }

    # RED FLAG: Same email, many claims (different addresses)
    if len(email_claims) >= 5:
        fraud_flags["fraud_indicators"].append(f"⚠️ HISTORIQUE: {len(email_claims)} sinistres avec cet email (possible réseau)")

    # RED FLAG: Same address, many claims (different people)
    if len(address_claims) >= 4:
        fraud_flags["fraud_indicators"].append(f"🚨 HISTORIQUE: {len(address_claims)} sinistres à cette adresse (réseau probable)")

    # RED FLAG: Same email + same address multiple times
    same_email_address = [c for c in email_claims if c.get("address", "").lower() == address.lower()]
    if len(same_email_address) >= 3:
        fraud_flags["fraud_indicators"].append(f"🚨 RÉSEAU: {len(same_email_address)} sinistres email+adresse identiques")
        fraud_flags["fraud_score_increase"] = 50

    return fraud_flags


def check_gps_location_match(claim_location: dict, photo_gps: dict) -> dict:
    """Check if photo GPS matches claim location (within ~2km tolerance)."""
    if not claim_location or not photo_gps:
        return {"matches": False, "distance_km": None, "flag": "NO_GPS_DATA"}

    try:
        # Haversine distance formula
        from math import radians, cos, sin, asin, sqrt

        lat1 = radians(claim_location["latitude"])
        lon1 = radians(claim_location["longitude"])
        lat2 = radians(photo_gps["latitude"])
        lon2 = radians(photo_gps["longitude"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        km = 6371 * c

        # Tolerance: 2km = normal GPS inaccuracy
        matches = km <= 2.0

        return {
            "matches": matches,
            "distance_km": round(km, 2),
            "flag": "OK" if matches else "LOCATION_MISMATCH"
        }
    except Exception as e:
        print(f"GPS match check error: {e}")
        return {"matches": False, "distance_km": None, "flag": "ERROR"}


def call_claude_vision(photo: dict, claim: dict) -> dict:
    """Call Claude Vision using Anthropic SDK and return a structured analysis dict."""
    if not client:
        return {
            "error": "api_not_configured",
            "damage_severity": "unknown",
            "recommendation": "Clé API Anthropic non configurée",
        }

    media_type = normalize_media_type(photo.get("content_type"))
    if media_type is None:
        return {
            "error": "unsupported_image",
            "damage_severity": "unknown",
            "recommendation": "Format d'image non supporte (utilisez JPEG, PNG, GIF ou WEBP).",
        }

    prompt = f"""Vous etes expert assurance multi-domaine (eau, automobile, cambriolage, incendie) + expert detection fraude.
Analysez cette photo de sinistre pour detecter fraude, incohérence, photo IA.

Contexte declare par l'assure:
- Type de degat DECLARE: {claim['damage_type']}
- Description: {claim['description']}
- Adresse: {claim['address']}

⚠️ ALERTE FRAUDE - Verifiez STRICTEMENT:
1. TYPE DÉGAT COHERENT?
   - La photo montre-t-elle le TYPE de dégât DÉCLARÉ?
   - Exemple: "accident_circulation" mais photo de fuite d'eau = FRAUDE!
   - Exemple: "cambriolage" mais photo de dégât grêle = FRAUDE!
   → TRÈS GRAVE si mismatch type = tentative fraude massive!

2. DETECTION IA/MANIPULATIONS:
   Signes d'image IA: textures lisses irrealistes, details flous/hyper-precis, artefacts, doigts bizarres, reflets anormaux,
   transitions non naturelles, details impossibles, textures repetitives, couleurs trop parfaites

3. Description COHERENTE avec photo visible?
   - La description correspond-elle aux dégâts visibles?

4. SOYEZ SEVERE: Mieux vaut false positive que laisser passer fraude!

Repondez UNIQUEMENT avec un objet JSON valide (aucun texte avant ou apres) avec EXACTEMENT ces champs:
{{
  "detected_damage_type": "fuite | inondation | rupture_canalisation | infiltration_toiture | moisissure | autre",
  "damage_severity": "low | medium | high | critical",
  "estimated_cost_eur": <nombre entier en euros>,
  "leak_location": "description courte du point d'impact ou de la source visible",
  "visible_damage": "description detaillee des degats visibles sur la photo",
  "is_ai_generated_or_manipulated": "genuine | suspicious | likely_ai | likely_manipulated",
  "ai_detection_confidence": "high | medium | low",
  "fraud_score": <entier 0-100, probabilite de fraude/incoherence>,
  "fraud_indicators": ["liste courte d'indices suspects, ou liste vide"],
  "consistency_with_declaration": "coherent | partiellement_coherent | incoherent",
  "recommendation": "recommandation d'action pour l'expert",
  "confidence": "high | medium | low"
}}"""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": photo["data"],
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
    except Exception as e:
        return {
            "error": str(type(e).__name__),
            "damage_severity": "unknown",
            "recommendation": f"Analyse echouee: {str(e)[:100]}",
        }

    response_text = response.content[0].text
    analysis = extract_json(response_text)

    if analysis is None:
        return {
            "damage_severity": "medium",
            "estimated_cost_eur": 3000,
            "visible_damage": response_text[:1000],
            "recommendation": "Inspection manuelle requise (reponse IA non structuree).",
            "fraud_score": 0,
            "confidence": "low",
        }

    # Check GPS location match (anti-fraud) - using phone GPS if available
    phone_gps = claim.get("phone_gps")
    gps_check = check_gps_location_match(claim.get("location"), phone_gps)
    analysis["location_verification"] = gps_check

    # Increase fraud score based on location mismatch (STRICT RULE)
    if gps_check["flag"] == "LOCATION_MISMATCH":
        distance = gps_check.get("distance_km", 0)
        if distance > 0.1:  # > 100 meters = AUTOMATIC +40 pts fraud
            fraud_increase = 40  # ALWAYS +40 for any mismatch > 100m
            analysis["fraud_score"] = min(100, analysis.get("fraud_score", 0) + fraud_increase)
            if "fraud_indicators" not in analysis or not isinstance(analysis["fraud_indicators"], list):
                analysis["fraud_indicators"] = []
            analysis["fraud_indicators"].append(f"🚨 GEOLOCALISATION MISMATCH: GPS a {distance}km de l'adresse declaree - SCORE +{fraud_increase}pts (SYSTÉMATIQUE)")

    # Check damage type coherence (CRITICAL FRAUD CHECK)
    detected_type = analysis.get("detected_damage_type", "").lower()
    declared_type = claim.get("damage_type", "").lower()

    # Map damage types to categories for fuzzy matching
    type_categories = {
        # Water damage
        "fuite": ["water", "fuite", "leak"],
        "inondation": ["water", "flood", "inondation"],
        "rupture_canalisation": ["water", "pipe", "canalisation"],
        "infiltration_toiture": ["water", "roof", "infiltration", "toiture"],
        # Automotive
        "accident_circulation": ["car", "accident", "collision", "vehicle"],
        "vandalisme_auto": ["car", "vandalism", "scratches", "automobile"],
        "vol_auto": ["car", "theft", "vol"],
        # Burglary
        "effraction": ["break-in", "effraction", "intrusion"],
        "vol": ["theft", "vol"],
        # Fire
        "incendie": ["fire", "incendie", "burn"],
        # Weather
        "grele": ["hail", "grele"],
        "tempete": ["storm", "tempete", "wind"],
    }

    # Check if declared type matches detected type
    declared_keywords = type_categories.get(declared_type, [declared_type.lower()])
    type_mismatch = not any(keyword in detected_type for keyword in declared_keywords)

    if type_mismatch and detected_type != "autre":
        # CRITICAL: Type mismatch = major fraud red flag
        mismatch_penalty = 70  # VERY HIGH penalty for type mismatch
        analysis["fraud_score"] = min(100, analysis.get("fraud_score", 0) + mismatch_penalty)
        if "fraud_indicators" not in analysis or not isinstance(analysis["fraud_indicators"], list):
            analysis["fraud_indicators"] = []
        analysis["fraud_indicators"].append(f"🚨 FRAUD MAJEURE: Type déclaré '{declared_type}' ≠ Type détecté '{detected_type}' - SCORE +{mismatch_penalty}pts")

    # Increase fraud score if AI-generated or manipulated photo detected (STRICT)
    ai_status = analysis.get("is_ai_generated_or_manipulated", "").lower()
    if ai_status in ["suspicious", "likely_ai", "likely_manipulated"]:
        # BE STRICT: suspicious=30, likely_ai=70, likely_manipulated=80
        ai_fraud_increase = {"suspicious": 30, "likely_ai": 70, "likely_manipulated": 80}.get(ai_status, 0)
        analysis["fraud_score"] = min(100, analysis.get("fraud_score", 0) + ai_fraud_increase)
        if "fraud_indicators" not in analysis or not isinstance(analysis["fraud_indicators"], list):
            analysis["fraud_indicators"] = []
        analysis["fraud_indicators"].append(f"⚠️ PHOTO SUSPECTE/FALSIFIEE: {ai_status.upper()} - SCORE +{ai_fraud_increase}pts")

    # Check EXIF timestamp coherence
    if claim.get("photos") and claim["photos"][0].get("exif_datetime"):
        photo_datetime = claim["photos"][0]["exif_datetime"]["datetime"]
        claim_datetime = datetime.fromisoformat(claim["created_at"])
        time_diff_hours = abs((claim_datetime - photo_datetime).total_seconds() / 3600)

        # If photo is older than 30 days before claim = SUSPICIOUS
        if time_diff_hours > 720:  # 30 days
            timestamp_penalty = 40
            analysis["fraud_score"] = min(100, analysis.get("fraud_score", 0) + timestamp_penalty)
            if "fraud_indicators" not in analysis or not isinstance(analysis["fraud_indicators"], list):
                analysis["fraud_indicators"] = []
            days_old = int(time_diff_hours / 24)
            analysis["fraud_indicators"].append(f"⚠️ PHOTO ANCIENNE: Photo prise {days_old} jours avant sinistre - SCORE +{timestamp_penalty}pts")

    # Check fraud history
    fraud_history = claim.get("fraud_history", {})
    if fraud_history.get("fraud_indicators"):
        history_penalty = fraud_history.get("fraud_score_increase", 20)
        analysis["fraud_score"] = min(100, analysis.get("fraud_score", 0) + history_penalty)
        if "fraud_indicators" not in analysis or not isinstance(analysis["fraud_indicators"], list):
            analysis["fraud_indicators"] = []
        for indicator in fraud_history["fraud_indicators"]:
            analysis["fraud_indicators"].append(indicator)

    return analysis


def _persist_claim_field(claim_id: str, fields: dict) -> bool:
    """Update a claim's fields in BOTH claims_db (in-memory) and MongoDB.

    Returns True if the MongoDB write succeeded. In-memory is always updated
    when the claim exists locally. Errors are caught and logged (never raised)
    so callers can keep responding even if Mongo is unreachable.
    """
    # In-memory update (fallback store)
    if claim_id in claims_db:
        claims_db[claim_id].update(fields)

    # MongoDB update
    mongo_ok = False
    try:
        result = claims_collection.update_one(
            {"claim_id": claim_id}, {"$set": fields}
        )
        mongo_ok = result.matched_count > 0
        if not mongo_ok:
            logger.warning("⚠️ _persist_claim_field: no Mongo doc matched claim_id=%s", claim_id)
    except Exception as e:
        logger.error("❌ _persist_claim_field Mongo update FAILED claim_id=%s: %r", claim_id, e)
    return mongo_ok


def _photo_to_vision_input(photo) -> dict | None:
    """Normalize a stored photo into the {data, content_type} shape that
    call_claude_vision() expects.

    Photos can be stored in two shapes:
      1. dict from /claims/{id}/photos -> {"data": <base64>, "content_type": ...}
      2. a data-URL string "data:image/jpeg;base64,...." (declaration viewer)
    Returns None if the photo cannot be normalized.
    """
    # Shape 1: dict already in the right format
    if isinstance(photo, dict):
        data = photo.get("data")
        if not data:
            return None
        # data may itself be a data-URL
        if isinstance(data, str) and data.startswith("data:"):
            try:
                header, b64 = data.split(",", 1)
                ctype = header.split(";")[0].replace("data:", "") or "image/jpeg"
                return {"data": b64, "content_type": ctype}
            except Exception:
                return None
        return {"data": data, "content_type": photo.get("content_type", "image/jpeg")}

    # Shape 2: bare data-URL string
    if isinstance(photo, str) and photo.startswith("data:"):
        try:
            header, b64 = photo.split(",", 1)
            ctype = header.split(";")[0].replace("data:", "") or "image/jpeg"
            return {"data": b64, "content_type": ctype}
        except Exception:
            return None

    return None


def analyze_claim_photos_with_claude(claim_id: str, photos: list) -> dict:
    """Run Claude Vision analysis on a claim's photos and persist the result.

    Uses the first usable photo. All API/parse errors are caught and returned
    as a structured analysis dict (never raised), so the declaration flow can
    never be broken by a Vision failure. The result is stored in MongoDB and
    claims_db under claim_data["analysis"].
    """
    claim = claims_db.get(claim_id)
    if claim is None:
        try:
            claim = claims_collection.find_one({"claim_id": claim_id}) or {}
        except Exception:
            claim = {}

    # Build a vision-ready photo from the first usable entry
    vision_photo = None
    for p in (photos or []):
        vision_photo = _photo_to_vision_input(p)
        if vision_photo:
            break

    if vision_photo is None:
        analysis = {
            "summary": "Aucune photo exploitable n'a été fournie pour l'analyse.",
            "detected_damage_type": claim.get("damage_type", "inconnu"),
            "damage_severity": "unknown",
            "fraud_score": 0,
            "recommendation": "Demander au client de fournir des photos nettes du sinistre.",
            "confidence": "low",
            "analyzed_at": datetime.now().isoformat(),
        }
        _persist_claim_field(claim_id, {"analysis": analysis})
        return analysis

    if not client:
        analysis = {
            "summary": "Clé API Anthropic non configurée : analyse Vision indisponible.",
            "detected_damage_type": claim.get("damage_type", "inconnu"),
            "damage_severity": "unknown",
            "fraud_score": 0,
            "recommendation": "Configurer ANTHROPIC_API_KEY puis relancer l'analyse.",
            "confidence": "low",
            "analyzed_at": datetime.now().isoformat(),
        }
        _persist_claim_field(claim_id, {"analysis": analysis})
        return analysis

    try:
        analysis = call_claude_vision(vision_photo, claim)
    except Exception as e:
        logger.error("❌ analyze_claim_photos_with_claude failed claim_id=%s: %r", claim_id, e)
        analysis = {
            "summary": f"L'analyse automatique a échoué ({type(e).__name__}).",
            "detected_damage_type": claim.get("damage_type", "inconnu"),
            "damage_severity": "unknown",
            "fraud_score": 0,
            "recommendation": "Inspection manuelle requise (erreur technique lors de l'analyse).",
            "confidence": "low",
        }

    if not isinstance(analysis, dict):
        analysis = {"summary": str(analysis)}

    # Build a human-readable summary if Claude didn't supply one.
    if not analysis.get("summary"):
        parts = []
        if analysis.get("detected_damage_type"):
            parts.append(f"Dégât détecté : {analysis['detected_damage_type']}")
        if analysis.get("damage_severity"):
            parts.append(f"gravité {analysis['damage_severity']}")
        if analysis.get("visible_damage"):
            parts.append(str(analysis["visible_damage"]))
        analysis["summary"] = ". ".join(parts) if parts else "Analyse Claude Vision effectuée."

    analysis["analyzed_at"] = datetime.now().isoformat()

    # Persist to Mongo + in-memory. Also sync the top-level fraud_score so the
    # claim viewer badge reflects the analysis.
    update_fields = {"analysis": analysis}
    if isinstance(analysis.get("fraud_score"), (int, float)):
        update_fields["fraud_score"] = analysis["fraud_score"]
    _persist_claim_field(claim_id, update_fields)

    logger.info("✅ Vision analysis stored for claim_id=%s (fraud_score=%s)",
                claim_id, analysis.get("fraud_score"))
    return analysis


def build_public_analysis(analysis: dict | None, claim_data: dict) -> dict:
    """Normalize the internal Vision analysis into the documented public shape.

    Output structure (always returns these keys, with safe fallbacks):
        {
          "summary": str,
          "damage_type": str,
          "severity": "High" | "Medium" | "Low",
          "recommendations": [str, ...],
          "analyzed_at": ISO timestamp,
          ... (internal fields preserved for fraud checks / viewer compat)
        }
    Never raises — falls back to an "analysis pending" object on bad input.
    """
    try:
        analysis = analysis if isinstance(analysis, dict) else {}

        # --- summary ---
        summary = analysis.get("summary") or analysis.get("visible_damage") \
            or "Analyse Claude Vision effectuée."

        # --- damage_type ---
        damage_type = analysis.get("detected_damage_type") \
            or claim_data.get("damage_type") or "Non déterminé"

        # --- severity (map low/medium/high/critical + numeric score → High/Medium/Low) ---
        raw_sev = str(analysis.get("damage_severity", "")).lower()
        sev_map = {
            "low": "Low", "faible": "Low",
            "medium": "Medium", "moderate": "Medium", "modérée": "Medium",
            "high": "High", "élevée": "High",
            "critical": "High", "critique": "High",
        }
        severity = sev_map.get(raw_sev)
        if severity is None:
            # Try a numeric 1-10 score if present.
            score_val = analysis.get("severity_score") or analysis.get("severity")
            try:
                n = float(score_val)
                severity = "High" if n >= 7 else "Medium" if n >= 4 else "Low"
            except (TypeError, ValueError):
                severity = "Medium" if analysis else "Low"

        # --- recommendations (list) ---
        recs = analysis.get("recommendations")
        if not isinstance(recs, list) or not recs:
            single = analysis.get("recommendation")
            recs = [single] if single else ["Inspection manuelle recommandée."]
        recs = [str(r) for r in recs if r]

        public = {
            "summary": str(summary),
            "damage_type": str(damage_type),
            "severity": severity,
            "recommendations": recs,
            "analyzed_at": analysis.get("analyzed_at") or datetime.now().isoformat(),
        }

        # Preserve internal fields so existing viewer code + fraud checks keep working.
        for key in ("detected_damage_type", "damage_severity",
                    "recommendation", "fraud_score", "fraud_indicators",
                    "is_ai_generated_or_manipulated", "confidence",
                    "consistency_with_declaration", "visible_damage",
                    "location_verification"):
            if key in analysis and key not in public:
                public[key] = analysis[key]
        return public
    except Exception as e:
        logger.error("❌ build_public_analysis failed: %r", e)
        return {
            "summary": "Analyse en attente.",
            "damage_type": claim_data.get("damage_type", "Non déterminé"),
            "severity": "Low",
            "recommendations": ["Inspection manuelle recommandée."],
            "analyzed_at": datetime.now().isoformat(),
        }


def _damage_types_consistent(declared: str, detected: str) -> bool:
    """Best-effort check that a declared damage type and a Vision-detected type
    belong to the same category. Returns True when consistent (or undeterminable).
    """
    declared = (declared or "").lower().strip()
    detected = (detected or "").lower().strip()
    if not declared or not detected or detected == "autre":
        return True  # Can't conclude inconsistency -> don't penalize.

    # Group everything into coarse categories.
    categories = {
        "water": ["water", "fuite", "leak", "inondation", "flood", "pipe",
                  "canalisation", "rupture", "infiltration", "toiture", "roof",
                  "moisissure", "appliance", "degat", "eau", "burst"],
        "car": ["car", "accident", "collision", "vehicle", "vehicule", "auto",
                "automobile", "vandalism", "vandalisme", "circulation"],
        "burglary": ["break-in", "effraction", "intrusion", "theft", "vol",
                     "cambriolage"],
        "fire": ["fire", "incendie", "burn", "feu"],
        "weather": ["hail", "grele", "storm", "tempete", "wind", "vent"],
    }

    def cat_of(value: str) -> set:
        found = set()
        for cat, kws in categories.items():
            if any(kw in value for kw in kws):
                found.add(cat)
        return found

    declared_cats = cat_of(declared)
    detected_cats = cat_of(detected)
    if not declared_cats or not detected_cats:
        return True  # One side unknown -> don't penalize.
    return bool(declared_cats & detected_cats)


def calculate_fraud_score(claim_data: dict, analysis_result: dict | None = None,
                          additional_factors: dict | None = None) -> int:
    """Compute a comprehensive fraud score (0-100+) for a claim.

    Combines claim data, the Claude Vision analysis, and database-derived
    checks (additional_factors). Each contributing factor is logged. Always
    returns an integer; on any unexpected error it returns 0 so the caller is
    never blocked.

    Scoring factors:
        +40  GPS mismatch with declared address
        +30  Damage type inconsistency (declared vs Vision-detected)
        +25  Multiple claims at same address in last 30 days
        +20  Multiple claims by same user in last 7 days
        +50  Photo quality / authenticity issues (blurry, fake, AI-generated)
        +15  Submitted at a suspicious hour (02:00-05:00)
        +10  Unusually short damage description (< 10 words)
        +35  Photo EXIF timestamp mismatch (if available)
    """
    try:
        analysis_result = analysis_result or {}
        additional_factors = additional_factors or {}
        score = 0

        def add(points: int, reason: str):
            nonlocal score
            score += points
            logger.info("🔎 fraud_score +%d → %s (running=%d)", points, reason, score)

        # +40 GPS mismatch with declared address
        gps_ver = claim_data.get("gps_verification") or {}
        gps_flag = gps_ver.get("flag")
        gps_matches = gps_ver.get("matches")
        if gps_flag == "LOCATION_MISMATCH" or (gps_matches is False and gps_flag not in ("NO_GPS_DATA", None)):
            dist = gps_ver.get("distance_km")
            add(40, f"GPS mismatch with address (distance={dist}km)")

        # +30 Damage type inconsistency (declared vs Vision-detected)
        declared_type = claim_data.get("damage_type", "")
        detected_type = analysis_result.get("detected_damage_type", "")
        if detected_type and not _damage_types_consistent(declared_type, detected_type):
            add(30, f"Damage type inconsistency (declared='{declared_type}' vs detected='{detected_type}')")

        # +25 Multiple claims same address in last 30 days
        if int(additional_factors.get("address_claims_30d", 0)) > 1:
            add(25, f"Multiple claims same address in 30 days ({additional_factors.get('address_claims_30d')})")

        # +20 Multiple claims same user in last 7 days
        if int(additional_factors.get("user_claims_7d", 0)) > 1:
            add(20, f"Multiple claims same user in 7 days ({additional_factors.get('user_claims_7d')})")

        # +50 Photo quality / authenticity issues
        ai_status = str(analysis_result.get("is_ai_generated_or_manipulated", "")).lower()
        confidence = str(analysis_result.get("confidence", "")).lower()
        photo_quality_flag = additional_factors.get("photo_quality_issue", False)
        if ai_status in ("suspicious", "likely_ai", "likely_manipulated") or photo_quality_flag:
            add(50, f"Photo quality/authenticity issue (ai_status='{ai_status}', flagged={photo_quality_flag})")
        elif confidence == "low" and analysis_result.get("detected_damage_type"):
            # Low confidence often signals blurry / unreadable photos.
            add(50, "Photo quality issue (low Vision confidence)")

        # +15 Suspicious submission time (02:00-05:00 local)
        created_at = claim_data.get("created_at")
        if created_at:
            try:
                hour = datetime.fromisoformat(str(created_at)).hour
                if 2 <= hour < 5:
                    add(15, f"Suspicious submission time ({hour:02d}h)")
            except (ValueError, TypeError):
                pass

        # +10 Unusually short description (< 10 words)
        desc = claim_data.get("description", "") or ""
        if 0 < len(desc.split()) < 10:
            add(10, f"Short description ({len(desc.split())} words)")

        # +35 EXIF timestamp mismatch (if available)
        if additional_factors.get("exif_timestamp_mismatch"):
            add(35, "Photo EXIF timestamp mismatch")

        logger.info("✅ Final fraud_score for claim_id=%s: %d",
                    claim_data.get("claim_id"), score)
        return int(score)
    except Exception as e:
        logger.error("❌ calculate_fraud_score failed: %r — returning 0", e)
        return 0


def compute_additional_fraud_factors(claim_data: dict) -> dict:
    """Derive database-backed fraud factors (multi-claim patterns, EXIF mismatch)
    from in-memory claims_db. All best-effort; returns safe defaults on error.
    """
    factors = {
        "address_claims_30d": 0,
        "user_claims_7d": 0,
        "exif_timestamp_mismatch": False,
        "photo_quality_issue": False,
    }
    try:
        now = datetime.now()
        addr = (claim_data.get("address") or "").lower().strip()
        email = (claim_data.get("user_email") or "").lower().strip()
        this_id = claim_data.get("claim_id")

        for cid, c in claims_db.items():
            if cid == this_id:
                continue
            created = c.get("created_at")
            if not created:
                continue
            try:
                dt = datetime.fromisoformat(str(created))
            except (ValueError, TypeError):
                continue
            days = (now - dt).total_seconds() / 86400.0
            if addr and (c.get("address") or "").lower().strip() == addr and days <= 30:
                factors["address_claims_30d"] += 1
            if email and (c.get("user_email") or "").lower().strip() == email and days <= 7:
                factors["user_claims_7d"] += 1

        # Count this claim itself so ">1" means at least one *other* matching claim.
        factors["address_claims_30d"] += 1
        factors["user_claims_7d"] += 1

        # EXIF timestamp mismatch: photo taken >30 days before the claim.
        photos = claim_data.get("photos") or []
        for p in photos:
            if isinstance(p, dict) and p.get("exif_datetime"):
                exif = p["exif_datetime"]
                photo_dt = exif.get("datetime") if isinstance(exif, dict) else None
                if isinstance(photo_dt, str):
                    try:
                        photo_dt = datetime.fromisoformat(photo_dt)
                    except (ValueError, TypeError):
                        photo_dt = None
                if isinstance(photo_dt, datetime):
                    if abs((now - photo_dt).total_seconds() / 86400.0) > 30:
                        factors["exif_timestamp_mismatch"] = True
                break
    except Exception as e:
        logger.warning("⚠️ compute_additional_fraud_factors failed: %r", e)
    return factors


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/")
async def read_root():
    """Serve index.html or fallback JSON."""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path, media_type="text/html")
    return {"message": "AssuranceIA API v2"}


@app.get("/health")
def health():
    """Lightweight health check (also useful to wake the Render dyno)."""
    key = ANTHROPIC_API_KEY or ""
    return {
        "status": "ok",
        "claims_count": len(claims_db),
        "claude_configured": bool(key),
        "key_length": len(key),
        "key_prefix": key[:7] if key else None,
        "model": CLAUDE_MODEL,
    }


@app.post("/claims")
def create_claim(claim: ClaimCreate):
    """Create a new claim, geocode its address, verify GPS phone vs address."""
    claim_id = f"CLM-{datetime.now().year}-{uuid.uuid4().hex[:6].upper()}"
    unique_token = secrets.token_urlsafe(32)  # Unique token for sharing

    location = geocode_address(claim.address)

    # Check GPS phone vs declared address (anti-fraud)
    phone_gps = None
    gps_verification = None
    if claim.phone_gps_lat is not None and claim.phone_gps_lon is not None:
        phone_gps = {"latitude": claim.phone_gps_lat, "longitude": claim.phone_gps_lon}
        gps_verification = check_gps_location_match(location, phone_gps)

    # Check fraud history (same email/address patterns)
    fraud_history = check_fraud_history(claim.user_email, claim.address)

    claim_data = {
        "claim_id": claim_id,
        "unique_token": unique_token,  # Unique shareable token
        "user_email": claim.user_email,
        "damage_type": claim.damage_type,
        "address": claim.address,
        "description": claim.description,
        "location": location,  # {latitude, longitude, display_name} from address geocoding
        "phone_gps": phone_gps,  # {latitude, longitude} from phone at claim creation
        "gps_verification": gps_verification,  # GPS phone vs declared address check
        "fraud_history": fraud_history,  # Check for email/address patterns
        "photos": [],
        "analysis": None,
        "status": "open",
        "created_at": datetime.now().isoformat(),
        # Attestation (safe defaults — set when the declarant attests)
        "attestation_confirmed": False,
        "attestation_timestamp": None,
    }

    claims_db[claim_id] = claim_data
    token_to_claim[unique_token] = claim_id  # Map token to claim

    # Log GPS verification result
    if gps_verification:
        status = "MATCH" if gps_verification["matches"] else "MISMATCH"
        distance = gps_verification.get("distance_km", "?")
        print(f"Claim created: {claim_id} | GPS Verification: {status} ({distance}km)")
    else:
        print(f"Claim created: {claim_id} (geocoded={location is not None}, no phone GPS)")

    return claim_data


@app.get("/claims")
def get_claims(insurer_id: str = Depends(verify_insurer_token)):
    """Return all claims (without heavy base64 photo payloads)."""
    result = []
    for claim in claims_db.values():
        light = {k: v for k, v in claim.items() if k != "photos"}
        light["photo_count"] = len(claim.get("photos", []))
        result.append(light)
    # Most recent first
    result.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    return result


@app.get("/claims/{claim_id}")
def get_claim(claim_id: str, insurer_id: str = Depends(verify_insurer_token)):
    """Return a single claim including its photos (base64)."""
    if claim_id not in claims_db:
        raise HTTPException(status_code=404, detail="Claim not found")
    return claims_db[claim_id]


@app.post("/claims/{claim_id}/photos")
async def upload_photo(claim_id: str, file: UploadFile = File(...)):
    """Upload a photo (base64-encoded) and attach it to the claim."""
    if claim_id not in claims_db:
        raise HTTPException(status_code=404, detail="Claim not found")

    contents = await file.read()
    if len(contents) > MAX_PHOTO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Photo trop volumineuse (max {MAX_PHOTO_BYTES // (1024 * 1024)} Mo).",
        )

    media_type = normalize_media_type(file.content_type)
    if media_type is None:
        raise HTTPException(
            status_code=415,
            detail="Format non supporte. Utilisez JPEG, PNG, GIF ou WEBP.",
        )

    # Extract GPS from EXIF
    gps_data = extract_gps_from_exif(contents)

    # Extract datetime from EXIF
    exif_datetime = extract_exif_datetime(contents)

    # Calculate image hash for duplicate detection
    image_hash = calculate_image_hash(contents)

    # Check for duplicate images
    duplicate_check = check_image_duplicates(image_hash, claim_id)

    base64_content = base64.b64encode(contents).decode("utf-8")
    photo_entry = {
        "filename": file.filename,
        "content_type": media_type,
        "data": base64_content,
        "gps": gps_data,  # {latitude, longitude} or None
        "exif_datetime": exif_datetime,  # datetime from EXIF
        "image_hash": image_hash,  # MD5 hash for duplicate detection
    }
    claims_db[claim_id]["photos"].append(photo_entry)

    # Persist the updated photos array to MongoDB so the analysis (and viewer)
    # see the photo even when reading from Mongo.
    _persist_claim_field(claim_id, {"photos": claims_db[claim_id]["photos"]})

    # Trigger Claude Vision analysis automatically now that photos exist.
    # Errors are swallowed inside analyze_claim_photos_with_claude (never raised),
    # so a Vision failure can't break photo upload.
    analysis_triggered = False
    try:
        analyze_claim_photos_with_claude(claim_id, claims_db[claim_id]["photos"])
        analysis_triggered = True
    except Exception as e:
        logger.error("❌ Auto-analysis after upload failed claim_id=%s: %r", claim_id, e)

    return {
        "status": "ok",
        "filename": file.filename,
        "photo_count": len(claims_db[claim_id]["photos"]),
        "gps_detected": gps_data is not None,
        "exif_datetime": exif_datetime.get("timestamp") if exif_datetime else None,
        "is_duplicate": duplicate_check["is_duplicate"],
        "duplicate_warning": duplicate_check["fraud_indicator"],
        "analysis_triggered": analysis_triggered,
        "message": "Photo uploadee avec succes",
    }


@app.post("/claims/{claim_id}/analyze")
@limiter.limit("10/minute")
def analyze_claim(request: Request, claim_id: str):
    """Analyze a claim's first photo using Claude Vision."""
    if claim_id not in claims_db:
        raise HTTPException(status_code=404, detail="Claim not found")

    claim = claims_db[claim_id]

    if not claim["photos"]:
        analysis = {
            "damage_severity": "unknown",
            "estimated_cost_eur": 0,
            "fraud_score": 0,
            "recommendation": "Veuillez uploader au moins une photo pour lancer l'analyse.",
        }
    elif not ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=503, detail="ANTHROPIC_API_KEY non configuree sur le serveur."
        )
    else:
        analysis = call_claude_vision(claim["photos"][0], claim)

    analysis["analyzed_at"] = datetime.now().isoformat()
    claims_db[claim_id]["analysis"] = analysis
    return analysis


@app.get("/claims/{claim_id}/report")
def download_report(claim_id: str):
    """Generate a PDF report for the claim and return it as a download."""
    if claim_id not in claims_db:
        raise HTTPException(status_code=404, detail="Claim not found")

    claim = claims_db[claim_id]
    pdf_bytes = build_claim_pdf(claim)
    filename = f"rapport_{claim_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ========== B2B WORKFLOW: ASSURANCE CREATES DECLARATION LINK ==========

@app.post("/create-declaration-link")
@limiter.limit("10/minute")
def create_declaration_link(request: Request, declaration_req: DeclarationLinkRequest, insurer_id: str = Depends(verify_insurer_token)):
    """
    Assurance/Courtier creates unique declaration link for client.
    Returns URL to send to client.

    Requires JWT authentication. Stores insurer_id for multi-tenant isolation.
    Phase 1: Sanitizes email + logs in audit trail.
    """
    client_ip = get_client_ip(request)

    # Sanitize insurer and client emails
    insurer_email = sanitize_email(declaration_req.insurer_email)
    if not insurer_email:
        raise HTTPException(status_code=400, detail="Email assurance invalide")

    client_email = sanitize_email(declaration_req.client_email) if declaration_req.client_email else ""

    token = secrets.token_urlsafe(32)

    link_data = {
        "token": token,
        "insurer_id": insurer_id,  # MULTI-TENANT: Link is tied to this insurer
        "insurer_email": insurer_email,
        "client_email": client_email,
        "created_at": datetime.now().isoformat(),
        "status": "pending"  # Not yet filled by client
    }

    declaration_links[token] = link_data

    # Save to MongoDB so token survives Render redeploys
    try:
        declaration_links_collection.insert_one(link_data)
    except:
        pass

    # Log link creation
    audit_create_declaration_link(insurer_id, token, client_email)

    base_url = os.getenv("APP_URL", "https://assurancia-api-2.onrender.com")
    declaration_url = f"{base_url}/declare/{token}"

    logger.info(f"✅ Declaration link created: {token[:8]}... for insurer_id={insurer_id} (IP={client_ip})")

    return {
        "token": token,
        "declaration_url": declaration_url,
        "insurer_email": insurer_email,
        "message": f"Envoyez ce lien au client: {declaration_url}",
        "qr_code_hint": f"QR code to generate: {declaration_url}"
    }


# ========== AUTHENTICATION HELPERS ==========

# Simple in-memory insurer credentials (in production, use a proper database)
INSURER_CREDENTIALS = {
    "kevin": {"password": "password123", "insurer_id": "INSURER_TEST_001"},
    "assurance1": {"password": "pass456", "insurer_id": "INSURER_TEST_002"},
    "demo": {"password": "demo123", "insurer_id": "DEMO_INSURER"},
}


def generate_jwt_token(insurer_id: str, expires_in_hours: int = 24) -> str:
    """Generate a JWT token for an insurer."""
    now = datetime.now()
    payload = {
        "insurer_id": insurer_id,
        "iat": now,
        "exp": now + timedelta(hours=expires_in_hours),
    }
    token = jwt.encode(payload, os.getenv("JWT_SECRET"), algorithm="HS256")
    return token


# ========== LOGIN & DASHBOARD ROUTES ==========

@app.get("/login-form")
def login_form_page():
    """Modern login form page with username/password authentication."""
    html = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Connexion - AssuranceIA™</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                padding: 20px;
                color: #e2e8f0;
            }
            .login-container {
                width: 100%;
                max-width: 420px;
            }
            .login-card {
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(51, 65, 85, 0.95) 100%);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 16px;
                padding: 50px 40px;
                backdrop-filter: blur(10px);
                box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
            }
            .logo-section {
                text-align: center;
                margin-bottom: 40px;
            }
            .logo {
                font-size: 48px;
                margin-bottom: 15px;
            }
            h1 {
                font-size: 28px;
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
                font-weight: 700;
            }
            .subtitle {
                color: #cbd5e1;
                font-size: 14px;
            }
            .form-group {
                margin-bottom: 25px;
            }
            label {
                display: block;
                margin-bottom: 10px;
                font-weight: 600;
                color: #f1f5f9;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            input[type="text"],
            input[type="password"] {
                width: 100%;
                padding: 14px 16px;
                background: rgba(15, 23, 42, 0.6);
                border: 2px solid rgba(148, 163, 184, 0.2);
                border-radius: 10px;
                font-family: inherit;
                font-size: 16px;
                color: #f1f5f9;
                transition: all 0.3s ease;
            }
            input[type="text"]::placeholder,
            input[type="password"]::placeholder {
                color: #64748b;
            }
            input[type="text"]:focus,
            input[type="password"]:focus {
                outline: none;
                border-color: #3b82f6;
                background: rgba(15, 23, 42, 0.8);
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
            }
            .login-button {
                width: 100%;
                padding: 14px;
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                border: none;
                border-radius: 10px;
                color: white;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
                margin-top: 10px;
            }
            .login-button:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 15px 40px rgba(59, 130, 246, 0.4);
            }
            .login-button:disabled {
                opacity: 0.7;
                cursor: not-allowed;
            }
            .login-button:active:not(:disabled) {
                transform: translateY(0);
            }
            #errorMessage {
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: #fca5a5;
                padding: 12px 16px;
                border-radius: 8px;
                margin-bottom: 20px;
                display: none;
                font-size: 14px;
                text-align: center;
            }
            #successMessage {
                background: rgba(34, 197, 94, 0.1);
                border: 1px solid rgba(34, 197, 94, 0.3);
                color: #86efac;
                padding: 12px 16px;
                border-radius: 8px;
                margin-bottom: 20px;
                display: none;
                font-size: 14px;
                text-align: center;
            }
            .demo-section {
                margin-top: 35px;
                padding-top: 25px;
                border-top: 1px solid rgba(148, 163, 184, 0.1);
                text-align: center;
            }
            .demo-title {
                font-size: 12px;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 15px;
                font-weight: 600;
            }
            .demo-cred {
                background: rgba(59, 130, 246, 0.05);
                border: 1px solid rgba(59, 130, 246, 0.15);
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 8px;
                font-size: 13px;
                color: #cbd5e1;
                font-family: 'Courier New', monospace;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            .demo-cred:hover {
                background: rgba(59, 130, 246, 0.1);
                border-color: rgba(59, 130, 246, 0.3);
            }
            .footer {
                text-align: center;
                margin-top: 30px;
                font-size: 12px;
                color: #64748b;
            }
            @media (max-width: 480px) {
                .login-card {
                    padding: 35px 25px;
                }
                h1 {
                    font-size: 24px;
                }
                .logo {
                    font-size: 40px;
                }
            }
        </style>
    </head>
    <body>
        <div class="login-container">
            <div class="login-card">
                <div class="logo-section">
                    <div class="logo">🔐</div>
                    <h1>AssuranceIA™</h1>
                    <p class="subtitle">Dashboard Assurance</p>
                </div>

                <div id="errorMessage"></div>
                <div id="successMessage"></div>

                <form id="loginForm" onsubmit="handleLogin(event)">
                    <div class="form-group">
                        <label for="username">Nom d'utilisateur</label>
                        <input
                            type="text"
                            id="username"
                            name="username"
                            placeholder="Entrez votre login"
                            autocomplete="username"
                            required
                            autofocus
                        >
                    </div>

                    <div class="form-group">
                        <label for="password">Mot de passe</label>
                        <input
                            type="password"
                            id="password"
                            name="password"
                            placeholder="Entrez votre mot de passe"
                            autocomplete="current-password"
                            required
                        >
                    </div>

                    <button type="submit" class="login-button" id="submitBtn">
                        📤 Se connecter
                    </button>
                </form>

                <div class="demo-section">
                    <div class="demo-title">Comptes de démonstration</div>
                    <div class="demo-cred" onclick="autofill('kevin', 'password123')">
                        kevin / password123
                    </div>
                    <div class="demo-cred" onclick="autofill('assurance1', 'pass456')">
                        assurance1 / pass456
                    </div>
                    <div class="demo-cred" onclick="autofill('demo', 'demo123')">
                        demo / demo123
                    </div>
                </div>

                <div class="footer">
                    🔒 Connexion sécurisée avec JWT
                </div>
            </div>
        </div>

        <script>
            const loginForm = document.getElementById('loginForm');
            const submitBtn = document.getElementById('submitBtn');
            const errorMsg = document.getElementById('errorMessage');
            const successMsg = document.getElementById('successMessage');

            function autofill(username, password) {
                document.getElementById('username').value = username;
                document.getElementById('password').value = password;
                document.getElementById('username').focus();
            }

            async function handleLogin(event) {
                event.preventDefault();

                const username = document.getElementById('username').value.trim();
                const password = document.getElementById('password').value;

                if (!username || !password) {
                    showError('Veuillez remplir tous les champs');
                    return;
                }

                submitBtn.disabled = true;
                submitBtn.textContent = '⏳ Connexion en cours...';
                hideMessages();

                try {
                    const response = await fetch('/authenticate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            username: username,
                            password: password,
                        })
                    });

                    const data = await response.json();

                    if (response.ok && data.token) {
                        showSuccess('Connexion réussie! Redirection...');
                        localStorage.setItem('assurance_jwt', data.token);
                        localStorage.setItem('insurer_id', data.insurer_id);

                        // Redirect after a brief delay for visual feedback
                        setTimeout(() => {
                            window.location.href = '/dashboard';
                        }, 500);
                    } else {
                        showError(data.detail || 'Identifiants invalides');
                        submitBtn.disabled = false;
                        submitBtn.textContent = '📤 Se connecter';
                    }
                } catch (error) {
                    showError('Erreur réseau: ' + error.message);
                    submitBtn.disabled = false;
                    submitBtn.textContent = '📤 Se connecter';
                }
            }

            function showError(message) {
                errorMsg.textContent = '❌ ' + message;
                errorMsg.style.display = 'block';
                successMsg.style.display = 'none';
            }

            function showSuccess(message) {
                successMsg.textContent = '✓ ' + message;
                successMsg.style.display = 'block';
                errorMsg.style.display = 'none';
            }

            function hideMessages() {
                errorMsg.style.display = 'none';
                successMsg.style.display = 'none';
            }

            // Allow Enter key to submit
            loginForm.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && e.target !== submitBtn) {
                    handleLogin(e);
                }
            });
        </script>
    </body>
    </html>
    """
    return Response(content=html, media_type="text/html")


@app.post("/authenticate")
@limiter.limit("5/minute")
async def authenticate(request: Request, username: str = "", password: str = ""):
    """Authenticate user with username/password and return JWT token.

    Accepts JSON: {"username": "...", "password": "..."}
    Returns: {"token": "<JWT>", "insurer_id": "..."}
    Rate limited to 5/minute per IP to prevent brute force.

    Phase 1: Uses bcrypt password verification + DEMO_USERS from environment.
    """
    # Get client IP for audit logging
    client_ip = get_client_ip(request)

    # Try to parse JSON body first
    try:
        body = await request.json()
        username = body.get("username", "").strip()
        password = body.get("password", "")
    except:
        # Fall back to form data if JSON parsing fails
        try:
            form_data = await request.form()
            username = form_data.get("username", "").strip()
            password = form_data.get("password", "")
        except:
            audit_authenticate(username or "unknown", client_ip, False)
            raise HTTPException(status_code=400, detail="Invalid request format")

    if not username or not password:
        audit_authenticate(username or "unknown", client_ip, False)
        raise HTTPException(status_code=400, detail="Username and password required")

    # Load demo users from environment
    demo_users = load_demo_users()

    # Check credentials against demo users
    if username not in demo_users and username not in INSURER_CREDENTIALS:
        audit_authenticate(username, client_ip, False)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Try demo users first (Phase 1)
    if username in demo_users:
        user_hash = demo_users[username].get("password_hash", "")
        if not user_hash or not verify_password(password, user_hash):
            audit_authenticate(username, client_ip, False)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        insurer_id = demo_users[username].get("insurer_id", "DEMO_INSURER")
    else:
        # Fallback to hardcoded credentials (legacy)
        cred = INSURER_CREDENTIALS[username]
        if cred["password"] != password:
            audit_authenticate(username, client_ip, False)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        insurer_id = cred["insurer_id"]

    # Generate JWT token
    token = generate_jwt_token(insurer_id)

    # Log successful authentication
    audit_authenticate(username, client_ip, True)
    logger.info(f"✅ User '{username}' (insurer_id={insurer_id}) authenticated successfully (IP={client_ip})")

    return {
        "token": token,
        "insurer_id": insurer_id,
        "message": "Authentication successful"
    }


@app.get("/login")
def login_page(token: str = ""):
    """Redirect to login-form. Legacy endpoint for backward compatibility.

    Usage: /login-form for the new form page
    """
    if not token:
        # Redirect to the new login form
        return RedirectResponse(url="/login-form", status_code=302)

    # Token provided: validate and store (legacy token-in-URL flow)
    try:
        payload = jwt.decode(token, os.getenv("JWT_SECRET"), algorithms=["HS256"])
        insurer_id = payload.get("insurer_id")
        if not insurer_id:
            raise ValueError("No insurer_id in token")
    except:
        return Response(content="""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <title>Erreur - AssuranceIA™</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    margin: 0;
                    color: #e2e8f0;
                }
                .container {
                    background: rgba(239, 68, 68, 0.1);
                    border: 1px solid rgba(239, 68, 68, 0.3);
                    border-radius: 12px;
                    padding: 40px;
                    max-width: 400px;
                    text-align: center;
                }
                h1 { color: #fca5a5; }
                a { color: #60a5fa; text-decoration: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>❌ Token invalide</h1>
                <p>Le token JWT n'est pas valide ou a expiré.</p>
                <a href="/login-form">← Retour à la connexion</a>
            </div>
        </body>
        </html>
        """, media_type="text/html")

    # Valid token: store in localStorage and redirect to dashboard
    return Response(content=f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Redirection...</title>
    </head>
    <body>
        <script>
            localStorage.setItem('assurance_jwt', '{token}');
            window.location.href = '/dashboard';
        </script>
        <p>Redirection vers le dashboard...</p>
    </body>
    </html>
    """, media_type="text/html")


@app.get("/dashboard")
def dashboard(request: Request):
    """Dashboard page that reads JWT token from localStorage.

    The page checks for a token in localStorage and displays claims.
    If no token, it redirects to /login.
    """
    html = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AssuranceIA™ Dashboard</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #e2e8f0;
                padding: 40px 20px;
                min-height: 100vh;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            header {
                margin-bottom: 40px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            h1 {
                font-size: 36px;
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .logout-btn {
                background: rgba(239, 68, 68, 0.2);
                border: 1px solid rgba(239, 68, 68, 0.5);
                color: #fca5a5;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
            }
            .logout-btn:hover {
                background: rgba(239, 68, 68, 0.3);
            }
            p {
                color: #cbd5e1;
                font-size: 16px;
            }
            .table-container {
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.9) 100%);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 12px;
                overflow: hidden;
                backdrop-filter: blur(10px);
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th {
                background: rgba(59, 130, 246, 0.1);
                padding: 16px;
                text-align: left;
                font-weight: 600;
                border-bottom: 1px solid rgba(148, 163, 184, 0.2);
                color: #60a5fa;
            }
            td {
                padding: 16px;
                border-bottom: 1px solid rgba(148, 163, 184, 0.1);
            }
            tr:hover {
                background: rgba(59, 130, 246, 0.05);
            }
            .reference {
                color: #60a5fa;
                font-weight: 600;
                cursor: pointer;
                text-decoration: none;
            }
            .reference:hover {
                text-decoration: underline;
            }
            .status {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }
            .status.low {
                background: rgba(34, 197, 94, 0.2);
                color: #86efac;
            }
            .status.medium {
                background: rgba(248, 113, 113, 0.2);
                color: #fca5a5;
            }
            .status.high {
                background: rgba(239, 68, 68, 0.2);
                color: #fca5a5;
            }
            .empty {
                text-align: center;
                padding: 60px 20px;
                color: #94a3b8;
            }
            .loading {
                text-align: center;
                padding: 60px 20px;
                color: #60a5fa;
            }
            .error {
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: #fca5a5;
                padding: 20px;
                border-radius: 6px;
                text-align: center;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div>
                    <h1>🔐 AssuranceIA™ Dashboard</h1>
                    <p>Tous les sinistres déclarés et analysés</p>
                </div>
                <button class="logout-btn" onclick="logout()">📤 Déconnexion</button>
            </header>

            <div class="table-container" id="tableContainer">
                <div class="loading">⏳ Chargement des sinistres...</div>
            </div>
        </div>

        <script>
            async function loadDashboard() {
                const token = localStorage.getItem('assurance_jwt');
                if (!token) {
                    window.location.href = '/login';
                    return;
                }

                try {
                    const response = await fetch('/api/dashboard', {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    });

                    if (!response.ok) {
                        if (response.status === 401 || response.status === 403) {
                            localStorage.removeItem('assurance_jwt');
                            window.location.href = '/login';
                            return;
                        }
                        throw new Error(`HTTP ${response.status}`);
                    }

                    const data = await response.json();
                    displayClaims(data.claims);
                } catch (error) {
                    document.getElementById('tableContainer').innerHTML = `
                        <div class="error">
                            ❌ Erreur de chargement: ${error.message}
                            <br><a href="/login" style="color: #60a5fa;">← Retour à la connexion</a>
                        </div>
                    `;
                }
            }

            function displayClaims(claims) {
                if (!claims || claims.length === 0) {
                    document.getElementById('tableContainer').innerHTML = `
                        <div class="empty">
                            <p>Aucun sinistre déclaré pour le moment</p>
                        </div>
                    `;
                    return;
                }

                let html = `
                    <table>
                        <thead>
                            <tr>
                                <th>Référence</th>
                                <th>Email</th>
                                <th>Prénom</th>
                                <th>Nom</th>
                                <th>Téléphone</th>
                                <th>Type</th>
                                <th>Adresse</th>
                                <th>Fraude</th>
                                <th>Date</th>
                            </tr>
                        </thead>
                        <tbody>
                `;

                claims.forEach(claim => {
                    const ref = claim.claim_id || 'N/A';
                    const email = claim.user_email || 'N/A';
                    const firstname = claim.firstname || 'N/A';
                    const lastname = claim.lastname || 'N/A';
                    const phone = claim.phone || 'N/A';
                    const damage = claim.damage_type || 'N/A';
                    const address = (claim.address || 'N/A').substring(0, 40);
                    const fraud_score = claim.fraud_score || 0;
                    const created = (claim.created_at || 'N/A').substring(0, 10);

                    let statusHtml;
                    if (fraud_score <= 20) {
                        statusHtml = '<span class="status low">✓ Fiable</span>';
                    } else if (fraud_score <= 50) {
                        statusHtml = '<span class="status medium">⚠ Attention</span>';
                    } else {
                        statusHtml = '<span class="status high">🚨 Suspect</span>';
                    }

                    html += `
                        <tr>
                            <td><a class="reference" onclick="viewClaim('${ref}')" title="Voir le détail">${ref}</a></td>
                            <td>${email}</td>
                            <td>${firstname}</td>
                            <td>${lastname}</td>
                            <td>${phone}</td>
                            <td>${damage}</td>
                            <td>${address}...</td>
                            <td>${statusHtml} (${fraud_score})</td>
                            <td>${created}</td>
                        </tr>
                    `;
                });

                html += `
                        </tbody>
                    </table>
                `;

                document.getElementById('tableContainer').innerHTML = html;
            }

            function viewClaim(claimId) {
                const token = localStorage.getItem('assurance_jwt');
                window.location.href = `/claim/${claimId}?token=${encodeURIComponent(token)}`;
            }

            function logout() {
                localStorage.removeItem('assurance_jwt');
                window.location.href = '/login';
            }

            // Load dashboard on page load
            window.onload = loadDashboard;
        </script>
    </body>
    </html>
    """

    return Response(content=html, media_type="text/html")


# ========== RATE LIMIT TEST ENDPOINT ==========

def ratelimit_test_key(request: Request) -> str:
    """Stable key for testing rate limiting behind a proxy that rotates IPs.

    Prefers the client_id query param so the 'client' is controllable and
    immune to per-request IP rotation. Falls back to client IP.
    """
    client_id = request.query_params.get("client_id")
    if client_id:
        return f"client_id:{client_id}"
    return get_client_ip(request)


@app.get("/ratelimit-test")
@limiter.limit("5/minute", key_func=ratelimit_test_key)
async def ratelimit_test(request: Request, client_id: str = "default"):
    return {"ok": True, "client_id": client_id}


# ========== TOKEN-BASED ROUTES (for sharing with clients/insurers) ==========

@app.get("/")
async def landing_page():
    """Beautiful landing page for AssuranceIA™."""
    html = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AssuranceIA™ - Validation Sinistres Intelligente</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #e2e8f0;
                line-height: 1.6;
                min-height: 100vh;
            }

            header {
                padding: 20px 40px;
                backdrop-filter: blur(10px);
                background: rgba(15, 23, 42, 0.8);
                border-bottom: 1px solid rgba(148, 163, 184, 0.1);
                position: sticky;
                top: 0;
                z-index: 100;
            }

            .logo {
                font-size: 28px;
                font-weight: 800;
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .hero {
                max-width: 1200px;
                margin: 0 auto;
                padding: 100px 40px;
                text-align: center;
            }

            .hero h1 {
                font-size: 56px;
                margin-bottom: 20px;
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-weight: 900;
            }

            .hero p {
                font-size: 20px;
                color: #cbd5e1;
                margin-bottom: 40px;
                max-width: 600px;
                margin-left: auto;
                margin-right: auto;
            }

            .hero-cta {
                display: inline-block;
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                color: white;
                padding: 16px 40px;
                border-radius: 12px;
                font-size: 18px;
                font-weight: 600;
                text-decoration: none;
                transition: all 0.3s ease;
                box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
            }

            .hero-cta:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 40px rgba(59, 130, 246, 0.4);
            }

            .features {
                max-width: 1200px;
                margin: 80px auto;
                padding: 0 40px;
            }

            .features h2 {
                text-align: center;
                font-size: 40px;
                margin-bottom: 50px;
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 30px;
                margin-bottom: 80px;
            }

            .feature-card {
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.8) 100%);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 12px;
                padding: 30px;
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }

            .feature-card:hover {
                transform: translateY(-5px);
                border-color: rgba(59, 130, 246, 0.5);
                background: linear-gradient(135deg, rgba(30, 41, 59, 1) 0%, rgba(51, 65, 85, 1) 100%);
            }

            .feature-icon {
                font-size: 40px;
                margin-bottom: 15px;
            }

            .feature-card h3 {
                font-size: 22px;
                margin-bottom: 10px;
                color: #f1f5f9;
            }

            .feature-card p {
                color: #cbd5e1;
                font-size: 16px;
            }

            .process {
                max-width: 1200px;
                margin: 80px auto;
                padding: 0 40px;
            }

            .process h2 {
                text-align: center;
                font-size: 40px;
                margin-bottom: 50px;
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .process-steps {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
            }

            .step {
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.8) 100%);
                border: 2px solid rgba(59, 130, 246, 0.3);
                border-radius: 12px;
                padding: 25px;
                text-align: center;
            }

            .step-number {
                display: inline-block;
                width: 50px;
                height: 50px;
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                color: white;
                border-radius: 50%;
                line-height: 50px;
                font-weight: bold;
                font-size: 22px;
                margin-bottom: 15px;
            }

            .step h3 {
                font-size: 18px;
                margin-bottom: 10px;
            }

            .step p {
                color: #cbd5e1;
                font-size: 14px;
            }

            footer {
                text-align: center;
                padding: 40px;
                color: #64748b;
                border-top: 1px solid rgba(148, 163, 184, 0.1);
                margin-top: 80px;
            }

            @media (max-width: 768px) {
                .hero h1 {
                    font-size: 36px;
                }

                .hero p {
                    font-size: 16px;
                }

                header {
                    padding: 15px 20px;
                }

                .hero {
                    padding: 50px 20px;
                }

                .features, .process {
                    padding: 0 20px;
                }
            }
        </style>
    </head>
    <body>
        <header>
            <div class="logo">🔐 AssuranceIA™</div>
        </header>

        <div class="hero">
            <h1>Validation Sinistres Intelligente</h1>
            <p>Analysez automatiquement les sinistres dégâts eaux avec l'IA et la géolocalisation anti-fraude</p>
            <a href="#declare" class="hero-cta">🚀 Déclarer un sinistre</a>
        </div>

        <div class="features">
            <h2>Pourquoi AssuranceIA™?</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">🤖</div>
                    <h3>IA Vision Avancée</h3>
                    <p>Analyse automatique des photos de sinistre avec Claude Vision pour une évaluation précise des dégâts</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📍</div>
                    <h3>Géolocalisation Anti-Fraude</h3>
                    <p>Vérification GPS des coordonnées photos vs adresse déclarée (2km tolérance) - impossible à falsifier</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <h3>Traitement Ultra-Rapide</h3>
                    <p>Analyse complète et génération de rapport PDF en moins de 2 minutes</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🛡️</div>
                    <h3>Détection Fraude Multicouche</h3>
                    <p>Scoring de fraude avec multiples vérifications: GPS, EXIF, IA photos, historique sinistres</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📱</div>
                    <h3>Mobile-First</h3>
                    <p>Interface responsive optimisée pour les déclarations depuis smartphone ou tablette</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <h3>Rapports Détaillés</h3>
                    <p>PDF complets avec analyse IA, photos, données GPS, score fraude et recommandations</p>
                </div>
            </div>
        </div>

        <div class="process">
            <h2>Comment ça marche?</h2>
            <div class="process-steps">
                <div class="step">
                    <div class="step-number">1</div>
                    <h3>Créer lien unique</h3>
                    <p>L'assurance génère un lien de déclaration personnalisé pour le client</p>
                </div>
                <div class="step">
                    <div class="step-number">2</div>
                    <h3>Remplir formulaire</h3>
                    <p>Le client accède au formulaire et capture son GPS (obligatoire)</p>
                </div>
                <div class="step">
                    <div class="step-number">3</div>
                    <h3>Envoyer photos</h3>
                    <p>Upload des photos de sinistre avec données EXIF intactes</p>
                </div>
                <div class="step">
                    <div class="step-number">4</div>
                    <h3>Analyse IA</h3>
                    <p>Système analyse automatiquement: type dégâts, sévérité, fraude</p>
                </div>
                <div class="step">
                    <div class="step-number">5</div>
                    <h3>Rapport PDF</h3>
                    <p>Rapport complet généré et envoyé au client + assurance</p>
                </div>
            </div>
        </div>

        <footer>
            <p>🔐 AssuranceIA™ 2026 | Propriété Intellectuelle Protégée</p>
        </footer>
    </body>
    </html>
    """
    return Response(content=html, media_type="text/html")


@app.get("/declare/{token}")
def get_declaration_form(token: str):
    """Beautiful declaration form for AssuranceIA™."""
    # Check if it's a declaration link (pending)
    if token in declaration_links:
        link_data = declaration_links[token]
        client_email = link_data.get("client_email", "")

        html = """
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Déclaration de Sinistre - AssuranceIA™</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }

                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    min-height: 100vh;
                    padding: 20px;
                    color: #e2e8f0;
                }

                .container {
                    max-width: 700px;
                    margin: 0 auto;
                }

                header {
                    text-align: center;
                    margin-bottom: 40px;
                    padding: 20px;
                }

                .logo {
                    font-size: 32px;
                    font-weight: 800;
                    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin-bottom: 10px;
                }

                .subtitle {
                    color: #cbd5e1;
                    font-size: 16px;
                }

                .card {
                    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.9) 100%);
                    border: 1px solid rgba(148, 163, 184, 0.2);
                    border-radius: 16px;
                    padding: 40px;
                    backdrop-filter: blur(10px);
                    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
                }

                .form-group {
                    margin-bottom: 25px;
                }

                label {
                    display: block;
                    margin-bottom: 10px;
                    font-weight: 600;
                    color: #f1f5f9;
                    font-size: 14px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }

                .required {
                    color: #ef4444;
                }

                input[type="text"],
                input[type="email"],
                input[type="tel"],
                select,
                textarea {
                    width: 100%;
                    padding: 14px;
                    background: rgba(15, 23, 42, 0.5);
                    border: 2px solid rgba(148, 163, 184, 0.2);
                    border-radius: 8px;
                    font-family: inherit;
                    font-size: 16px;
                    color: #f1f5f9;
                    transition: all 0.3s ease;
                }

                input[type="text"]::placeholder,
                input[type="email"]::placeholder,
                input[type="tel"]::placeholder,
                textarea::placeholder {
                    color: #64748b;
                }

                input[type="text"]:focus,
                input[type="email"]:focus,
                input[type="tel"]:focus,
                select:focus,
                textarea:focus {
                    outline: none;
                    border-color: #3b82f6;
                    background: rgba(15, 23, 42, 0.8);
                    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
                }

                textarea {
                    resize: vertical;
                    min-height: 120px;
                }

                .gps-section {
                    background: rgba(59, 130, 246, 0.1);
                    border: 2px dashed rgba(59, 130, 246, 0.3);
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 25px;
                    text-align: center;
                }

                .gps-button {
                    display: inline-block;
                    padding: 14px 30px;
                    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    margin-bottom: 12px;
                }

                .gps-button:hover:not(:disabled) {
                    transform: translateY(-2px);
                    box-shadow: 0 10px 25px rgba(59, 130, 246, 0.4);
                }

                .gps-button:disabled {
                    opacity: 0.7;
                    cursor: not-allowed;
                }

                #gpsStatus {
                    min-height: 24px;
                    font-size: 14px;
                    color: #cbd5e1;
                }

                .submit-button {
                    width: 100%;
                    padding: 16px;
                    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 18px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
                }

                .submit-button:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 15px 40px rgba(59, 130, 246, 0.4);
                }

                .submit-button:disabled {
                    opacity: 0.7;
                    cursor: not-allowed;
                }

                .attest-box {
                    background: rgba(34, 197, 94, 0.08);
                    border: 2px solid rgba(34, 197, 94, 0.35);
                    border-radius: 10px;
                    padding: 18px 20px;
                    margin-bottom: 25px;
                }

                .attest-label {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    cursor: pointer;
                    margin-bottom: 0;
                    text-transform: none;
                    letter-spacing: normal;
                    font-size: 15px;
                    color: #f1f5f9;
                    line-height: 1.5;
                }

                .attest-label input[type="checkbox"] {
                    width: 22px;
                    height: 22px;
                    min-width: 22px;
                    accent-color: #22c55e;
                    cursor: pointer;
                }

                .attest-check-icon {
                    font-size: 20px;
                }

                .attest-text {
                    font-weight: 600;
                }

                #status {
                    margin-top: 25px;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    display: none;
                }

                .success-message {
                    background: rgba(34, 197, 94, 0.1);
                    border: 2px solid rgba(34, 197, 94, 0.3);
                    color: #86efac;
                }

                .error-message {
                    background: rgba(239, 68, 68, 0.1);
                    border: 2px solid rgba(239, 68, 68, 0.3);
                    color: #fca5a5;
                }

                .info-text {
                    font-size: 13px;
                    color: #94a3b8;
                    margin-top: 8px;
                    font-style: italic;
                }

                @media (max-width: 768px) {
                    .card {
                        padding: 25px;
                    }

                    header {
                        margin-bottom: 30px;
                    }

                    .logo {
                        font-size: 24px;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <div class="logo">🔐 AssuranceIA™</div>
                    <p class="subtitle">Déclaration de sinistre - Analyse IA automatique</p>
                </header>

                <div class="card">
                    <form id="declarationForm" enctype="multipart/form-data">
                        <!-- GPS Capture - MANDATORY -->
                        <div class="form-group">
                            <label>📍 Géolocalisation <span class="required">*</span></label>
                            <div class="gps-section">
                                <button id="gpsBtn" type="button" class="gps-button">📍 Capturer mon GPS</button>
                                <div id="gpsStatus"></div>
                                <p class="info-text">La géolocalisation est obligatoire pour valider votre déclaration</p>
                            </div>
                        </div>

                        <!-- Email -->
                        <div class="form-group">
                            <label for="email">Email <span class="required">*</span></label>
                            <input type="email" id="email" name="email" value="{client_email}" placeholder="votre@email.com" required>
                        </div>

                        <!-- Prénom -->
                        <div class="form-group">
                            <label for="firstname">Prénom <span class="required">*</span></label>
                            <input type="text" id="firstname" name="firstname" placeholder="Jean" required>
                        </div>

                        <!-- Nom -->
                        <div class="form-group">
                            <label for="lastname">Nom <span class="required">*</span></label>
                            <input type="text" id="lastname" name="lastname" placeholder="Dupont" required>
                        </div>

                        <!-- Téléphone -->
                        <div class="form-group">
                            <label for="phone">Téléphone <span class="required">*</span></label>
                            <input type="tel" id="phone" name="phone" placeholder="06 12 34 56 78" required>
                        </div>

                        <!-- Damage Type -->
                        <div class="form-group">
                            <label for="damageType">Type de dégâts <span class="required">*</span></label>
                            <select id="damageType" name="damageType" required>
                                <option value="">-- Sélectionnez un type --</option>
                                <optgroup label="💧 Dégâts des Eaux">
                                    <option value="water_damage">Dégâts des eaux</option>
                                    <option value="flooding">Inondation</option>
                                    <option value="pipe_burst">Tuyauterie éclatée</option>
                                    <option value="roof_leak">Fuite de toiture</option>
                                    <option value="appliance_leak">Fuite électroménager</option>
                                </optgroup>
                                <optgroup label="🚗 Sinistres Automobile">
                                    <option value="vehicle_accident">Accident automobile</option>
                                    <option value="vehicle_vandalism">Vandalisme automobile</option>
                                </optgroup>
                            </select>
                        </div>

                        <!-- Address -->
                        <div class="form-group">
                            <label for="address">Adresse du sinistre <span class="required">*</span></label>
                            <input type="text" id="address" name="address" placeholder="123 rue de la Paix, 75000 Paris" required>
                            <p class="info-text">Doit correspondre à votre GPS (tolérance 2km)</p>
                        </div>

                        <!-- Description -->
                        <div class="form-group">
                            <label for="description">Description détaillée <span class="required">*</span></label>
                            <textarea id="description" name="description" placeholder="Décrivez précisément les dégâts constatés..." required></textarea>
                        </div>

                        <!-- Photos -->
                        <div class="form-group">
                            <label for="photos">📸 Photos des dégâts (jusqu'à 10)</label>
                            <input type="file" id="photos" name="photos" multiple accept="image/*">
                            <p class="info-text">Uploadez plusieurs photos pour une analyse meilleure (max 5 Mo / photo)</p>
                            <p id="photoCount" style="margin-top:8px;font-weight:600;color:#60a5fa;">0/10 photos</p>
                            <div id="photoPreviews" style="display:flex;flex-wrap:wrap;gap:10px;margin-top:12px;"></div>
                        </div>

                        <!-- Hidden GPS fields -->
                        <input type="hidden" id="gpsLat" name="gpsLat">
                        <input type="hidden" id="gpsLon" name="gpsLon">

                        <!-- Attestation sur l'honneur - OBLIGATOIRE -->
                        <div class="attest-box">
                            <label class="attest-label" for="attestation">
                                <input type="checkbox" id="attestation" name="attestation" required>
                                <span class="attest-check-icon">✅</span>
                                <span class="attest-text">J'atteste sur l'honneur la véracité des informations <span class="required">*</span></span>
                            </label>
                        </div>

                        <!-- Submit Button -->
                        <button type="submit" class="submit-button">🚀 Soumettre ma déclaration</button>
                    </form>

                    <!-- Status Messages -->
                    <div id="status"></div>
                </div>
            </div>

            <script>
            const token = '{token}';
            const apiUrl = window.location.origin;
            let currentGPS = null;

            // --- Multi-photo selection with previews, removal, and count ---
            const MAX_PHOTOS = 10;
            const MAX_PHOTO_BYTES = 5 * 1024 * 1024;
            let selectedPhotos = [];  // persistent list of File objects

            function renderPhotoPreviews() {
                const wrap = document.getElementById("photoPreviews");
                const counter = document.getElementById("photoCount");
                wrap.innerHTML = "";
                selectedPhotos.forEach(function(file, idx) {
                    const cell = document.createElement("div");
                    cell.style.cssText = "position:relative;width:90px;height:90px;border-radius:8px;overflow:hidden;border:2px solid rgba(148,163,184,0.3);";
                    const img = document.createElement("img");
                    img.style.cssText = "width:100%;height:100%;object-fit:cover;";
                    img.src = URL.createObjectURL(file);
                    img.onload = function() { URL.revokeObjectURL(img.src); };
                    const rm = document.createElement("button");
                    rm.type = "button";
                    rm.textContent = "✕";
                    rm.title = "Retirer";
                    rm.style.cssText = "position:absolute;top:2px;right:2px;width:22px;height:22px;border:none;border-radius:50%;background:rgba(239,68,68,0.9);color:#fff;font-weight:700;cursor:pointer;line-height:1;";
                    rm.addEventListener("click", function() {
                        selectedPhotos.splice(idx, 1);
                        renderPhotoPreviews();
                    });
                    cell.appendChild(img);
                    cell.appendChild(rm);
                    wrap.appendChild(cell);
                });
                counter.textContent = selectedPhotos.length + "/" + MAX_PHOTOS + " photos";
            }

            document.getElementById("photos").addEventListener("change", function(e) {
                const files = Array.from(e.target.files || []);
                files.forEach(function(f) {
                    if (selectedPhotos.length >= MAX_PHOTOS) {
                        alert("Maximum " + MAX_PHOTOS + " photos.");
                        return;
                    }
                    if (f.size > MAX_PHOTO_BYTES) {
                        alert("Photo trop volumineuse (max 5 Mo): " + f.name);
                        return;
                    }
                    selectedPhotos.push(f);
                });
                e.target.value = "";  // allow re-selecting the same file later
                renderPhotoPreviews();
            });

            document.getElementById("gpsBtn").addEventListener("click", function(e) {
                e.preventDefault();
                const gpsBtn = document.getElementById("gpsBtn");
                const gpsStatus = document.getElementById("gpsStatus");

                if (!navigator.geolocation) {
                    gpsStatus.innerHTML = '<span style="color: #fca5a5;">Geolocalisation non supportee</span>';
                    return;
                }

                gpsBtn.disabled = true;
                gpsBtn.textContent = "Localisation...";
                gpsStatus.innerHTML = "";

                navigator.geolocation.getCurrentPosition(function(position) {
                    currentGPS = {
                        lat: position.coords.latitude,
                        lon: position.coords.longitude
                    };
                    document.getElementById("gpsLat").value = currentGPS.lat;
                    document.getElementById("gpsLon").value = currentGPS.lon;

                    const lat = currentGPS.lat.toFixed(5);
                    const lon = currentGPS.lon.toFixed(5);
                    const acc = Math.round(position.coords.accuracy);

                    gpsBtn.textContent = "GPS: " + lat + ", " + lon;
                    gpsStatus.innerHTML = '<span style="color: #86efac;">Position: +/-' + acc + 'm</span>';
                    gpsBtn.disabled = false;
                }, function(error) {
                    gpsStatus.innerHTML = '<span style="color: #fca5a5;">Erreur: ' + error.message + '</span>';
                    gpsBtn.textContent = "Capturer mon GPS";
                    gpsBtn.disabled = false;
                }, { enableHighAccuracy: true, timeout: 10000 });
            });

            document.getElementById("declarationForm").addEventListener("submit", function(e) {
                e.preventDefault();

                if (!currentGPS) {
                    alert("GPS OBLIGATOIRE!");
                    return;
                }

                const attestEl = document.getElementById("attestation");
                if (!attestEl || !attestEl.checked) {
                    alert("Vous devez attester sur l'honneur la véracité des informations avant de soumettre.");
                    if (attestEl) { attestEl.focus(); }
                    return;
                }

                const submitBtn = document.querySelector(".submit-button");
                if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = "⏳ Analyse en cours..."; }

                const formData = new FormData();
                formData.append("user_email", document.getElementById("email").value);
                formData.append("firstname", document.getElementById("firstname").value);
                formData.append("lastname", document.getElementById("lastname").value);
                formData.append("phone", document.getElementById("phone").value);
                formData.append("damage_type", document.getElementById("damageType").value);
                formData.append("address", document.getElementById("address").value);
                formData.append("description", document.getElementById("description").value);
                formData.append("phone_gps_lat", currentGPS.lat);
                formData.append("phone_gps_lon", currentGPS.lon);
                formData.append("attestation_confirmed", "true");
                formData.append("token", token);

                // Attach photos from the curated list (previews/removal applied)
                selectedPhotos.forEach(function(f) {
                    formData.append("photos", f);
                });

                fetch(apiUrl + "/declare/" + token + "/submit", {
                    method: "POST",
                    body: formData
                }).then(function(response) {
                    return response.json();
                }).then(function(data) {
                    if (data.claim_id) {
                        const statusDiv = document.getElementById("status");
                        statusDiv.className = "success-message";
                        statusDiv.style.display = "block";
                        // Use token-authenticated client link (no claim_id, no PDF exposed).
                        var viewUrl = apiUrl + (data.client_view_url || ("/my-claim/" + data.unique_token));
                        statusDiv.innerHTML = "<h3>Declaration creee!</h3><p>Reference: " + data.claim_id + "</p>" +
                            "<p style='margin-top:12px;'>" + (data.photo_count || 0) + " photo(s) recue(s).</p>" +
                            "<p style='margin-top:12px;'>" +
                            "<a href='" + viewUrl + "' style='color:#60a5fa;font-weight:600;'>Voir ma declaration</a>" +
                            "</p>";
                        document.getElementById("declarationForm").style.display = "none";
                    } else {
                        alert("Erreur");
                        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = "🚀 Soumettre ma déclaration"; }
                    }
                }).catch(function(err) {
                    alert("Erreur reseau");
                    if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = "🚀 Soumettre ma déclaration"; }
                });
            });
            </script>
        </body>
        </html>
        """.replace("{token}", token).replace("{client_email}", client_email or "")
        return Response(content=html, media_type="text/html")

    # If token is an existing claim, show claim details
    if token in token_to_claim:
        claim_id = token_to_claim[token]
        claim = claims_db.get(claim_id)
        if claim:
            light_claim = {k: v for k, v in claim.items() if k != "photos"}
            light_claim["photo_count"] = len(claim.get("photos", []))
            return light_claim

    raise HTTPException(status_code=404, detail="Lien de déclaration invalide ou expiré")


@app.get("/claim/{claim_id}")
def view_claim_details(claim_id: str, token: str = "", request: Request = None):
    """INTERNAL (insurer/expert) view of a claim — full data incl. fraud score.

    Accepts JWT either via:
    1. Authorization header (classic JWT auth)
    2. Query param ?token=<JWT> (for dashboard links)

    Looks up by claim_id (used from the dashboard). Client-facing access goes
    through /my-claim/{unique_token} which renders the same page in client mode
    (fraud score + expert conclusion hidden).
    """
    # Validate JWT from either header or query param
    insurer_id = None

    if token:
        # Token from query param
        try:
            payload = jwt.decode(token, os.getenv("JWT_SECRET"), algorithms=["HS256"])
            insurer_id = payload.get("insurer_id")
        except:
            raise HTTPException(status_code=401, detail="Token invalide")
    else:
        # Try to get token from Authorization header
        if not request:
            raise HTTPException(status_code=401, detail="Not authenticated")
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                payload = jwt.decode(token, os.getenv("JWT_SECRET"), algorithms=["HS256"])
                insurer_id = payload.get("insurer_id")
            except:
                raise HTTPException(status_code=401, detail="Token invalide")
        else:
            raise HTTPException(status_code=401, detail="Not authenticated")

    if not insurer_id:
        raise HTTPException(status_code=401, detail="Invalid token: no insurer_id")

    # Try MongoDB first, then fall back to in-memory
    claim = None
    try:
        claim = claims_collection.find_one({"claim_id": claim_id})
    except:
        pass

    if not claim:
        claim = claims_db.get(claim_id)

    if not claim:
        raise HTTPException(status_code=404, detail="Sinistre non trouvé")

    # MULTI-TENANT: Verify insurer owns this claim
    claim_insurer_id = claim.get("insurer_id", "UNKNOWN")
    if claim_insurer_id != insurer_id:
        raise HTTPException(status_code=403, detail="Accès refusé: ce sinistre n'appartient pas à votre assurance")

    return _render_claim_details(claim_id, claim, client_view=False)


@app.get("/api/claim/{claim_id}")
def get_claim_json(claim_id: str, request: Request, insurer_id: str = Depends(verify_insurer_token)):
    """API endpoint: return claim details as JSON for programmatic access.

    This endpoint returns the raw claim data as JSON (not HTML).
    Requires JWT authentication.
    MULTI-TENANT: Only returns claims owned by this insurer.
    Phase 1: Logs access in audit trail.
    """
    client_ip = get_client_ip(request)
    claim = _load_claim(claim_id)
    if not claim:
        audit_view_claim(insurer_id, claim_id, client_ip)
        raise HTTPException(status_code=404, detail="Sinistre non trouvé")

    # MULTI-TENANT: Verify insurer owns this claim
    claim_insurer_id = claim.get("insurer_id", "UNKNOWN")
    if claim_insurer_id != insurer_id:
        audit_view_claim(insurer_id, claim_id, client_ip)
        raise HTTPException(status_code=403, detail="Accès refusé: ce sinistre n'appartient pas à votre assurance")

    # Log successful access
    audit_view_claim(insurer_id, claim_id, client_ip)

    # Return all claim data as JSON
    # Remove any non-serializable fields (like binary photo data if stored as bytes)
    serializable_claim = {}
    for key, value in claim.items():
        if key == "_id":  # MongoDB internal ID, skip it
            continue
        if isinstance(value, bytes):  # Skip binary data
            serializable_claim[key] = "[binary data]"
        elif isinstance(value, dict) or isinstance(value, list) or isinstance(value, str) or isinstance(value, (int, float, bool)) or value is None:
            serializable_claim[key] = value
        else:
            serializable_claim[key] = str(value)

    return serializable_claim


@app.get("/api/dashboard")
@limiter.limit("10/minute")
async def get_dashboard_json(request: Request, insurer_id: str = Depends(verify_insurer_token)):
    """API endpoint: return all claims as JSON for programmatic access.

    Returns a list of claims FOR THIS INSURER ONLY (multi-tenant isolation).
    Sorted by creation date (newest first).
    Requires JWT authentication.
    """
    try:
        # MULTI-TENANT: Filter by insurer_id
        claims_list = list(claims_collection.find({"insurer_id": insurer_id}, {"_id": 0}).sort("created_at", -1))
    except Exception as e:
        logger.warning("⚠️  Dashboard could not read from MongoDB: %r — using in-memory fallback.", e)
        claims_list = []

    # Fallback / merge: if Mongo returned nothing but we have in-memory claims
    if not claims_list and claims_db:
        claims_list = sorted(
            [{k: v for k, v in c.items() if k != "_id"} for c in claims_db.values() if c.get("insurer_id") == insurer_id],
            key=lambda c: c.get("created_at", ""),
            reverse=True,
        )

    # Make all values JSON-serializable
    serializable_claims = []
    for claim in claims_list:
        serializable_claim = {}
        for key, value in claim.items():
            if key == "_id":
                continue
            if isinstance(value, bytes):
                serializable_claim[key] = "[binary data]"
            elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                serializable_claim[key] = value
            else:
                serializable_claim[key] = str(value)
        serializable_claims.append(serializable_claim)

    return {
        "insurer_id": insurer_id,
        "total_claims": len(serializable_claims),
        "claims": serializable_claims
    }


def _load_claim_by_token(unique_token: str) -> tuple:
    """Resolve a claim from its unique_token (client authorization token).

    Returns (claim_id, claim_dict) or (None, None) if the token doesn't match
    any claim. Checks MongoDB first, then in-memory, then the token_to_claim map.
    Never raises — returns (None, None) on any backend error.
    """
    if not unique_token:
        return None, None
    # 1. MongoDB direct lookup by unique_token.
    try:
        doc = claims_collection.find_one({"unique_token": unique_token})
        if doc:
            return doc.get("claim_id"), doc
    except Exception:
        pass
    # 2. In-memory token map.
    claim_id = token_to_claim.get(unique_token)
    if claim_id and claim_id in claims_db:
        return claim_id, claims_db[claim_id]
    # 3. Scan in-memory claims as a last resort.
    for cid, c in claims_db.items():
        if isinstance(c, dict) and c.get("unique_token") == unique_token:
            return cid, c
    return None, None


@app.get("/my-claim/{unique_token}")
def view_my_claim(unique_token: str):
    """CLIENT view — access a claim via its unguessable unique_token.

    Shows only client-relevant sections (declaration, geolocation, photos,
    analysis, attestation). The internal fraud score and the expert conclusion
    are stripped out. Invalid/unknown tokens return 404.
    """
    claim_id, claim = _load_claim_by_token(unique_token)
    if not claim:
        raise HTTPException(status_code=404, detail="Lien invalide ou expiré")
    return _render_claim_details(claim_id, claim, client_view=True,
                                 unique_token=unique_token)


def _render_claim_details(claim_id: str, claim: dict, client_view: bool = False,
                          unique_token: str = "") -> Response:
    """Render the claim-detail HTML page.

    When client_view=True, internal data (fraud score, expert conclusion) is
    hidden and the PDF/email buttons point at the token-authenticated routes.
    """
    # --- GPS coordinates (defensive: phone_gps may be None or missing keys) ---
    phone_gps = claim.get("phone_gps") or {}
    gps_lat = phone_gps.get("latitude", "N/A")
    gps_lon = phone_gps.get("longitude", "N/A")
    if gps_lat is None:
        gps_lat = "N/A"
    if gps_lon is None:
        gps_lon = "N/A"

    # gps_verification may be None or a dict
    gps_verification = claim.get("gps_verification") or {}
    gps_match = bool(gps_verification.get("match")) if gps_verification else False

    # GPS status block (built outside the f-string to avoid backslash-in-f-string errors)
    gps_status_class = "verified" if gps_match else "mismatch"
    if gps_match:
        gps_status_text = "✓ GPS vérifié avec l'adresse"
    else:
        gps_status_text = "⚠ GPS ne correspond pas à l'adresse"
    if gps_verification and gps_verification.get("distance_km") is not None:
        gps_distance_text = f" (Distance: {gps_verification.get('distance_km')} km)"
    else:
        gps_distance_text = ""

    # JS-safe numeric strings for the map (avoid 'N/A' breaking parseFloat silently is fine,
    # but guard against None too)
    js_lat = gps_lat if gps_lat not in (None, "N/A") else ""
    js_lon = gps_lon if gps_lon not in (None, "N/A") else ""

    # --- Fraud score (defensive: may be None or non-numeric) ---
    fraud_score = claim.get("fraud_score", 0)
    try:
        fraud_score = float(fraud_score)
    except (TypeError, ValueError):
        fraud_score = 0
    if fraud_score <= 20:
        fraud_status = '<span class="badge low">✓ Fiable</span>'
    elif fraud_score <= 50:
        fraud_status = '<span class="badge medium">⚠ Attention</span>'
    else:
        fraud_status = '<span class="badge high">🚨 Suspect</span>'
    fraud_score_display = int(fraud_score) if fraud_score == int(fraud_score) else fraud_score

    # Fraud score is INTERNAL ONLY — never expose it in the client (token) view.
    if client_view:
        fraud_score_row = ""
    else:
        fraud_score_row = f"""<div class="info-item">
                        <div class="info-label">Score Fraude</div>
                        <div class="info-value">{fraud_status} ({fraud_score_display})</div>
                    </div>"""

    # --- Date (defensive: created_at may be None / missing / short) ---
    # Stored in ISO format; displayed as "10 juin 2026 à 13:45:32 (GMT+3)".
    created_at = claim.get("created_at")
    date_display = format_french_datetime(created_at)

    # --- Photos HTML (handles dict photos {data, content_type} AND data-URL strings) ---
    photos = claim.get("photos") or []
    photo_count = len(photos)
    photos_inner = ""
    for i, photo_data in enumerate(photos):
        src = None
        if isinstance(photo_data, str) and photo_data.startswith("data:"):
            src = photo_data
        elif isinstance(photo_data, dict):
            data = photo_data.get("data")
            if isinstance(data, str) and data.startswith("data:"):
                src = data
            elif isinstance(data, str) and data:
                ctype = photo_data.get("content_type", "image/jpeg")
                src = f"data:{ctype};base64,{data}"
        if src:
            photos_inner += f'<img src="{src}" alt="Photo {i+1}" class="claim-photo">'

    # The inline "add photos" button is an internal management action only.
    add_photos_btn = "" if client_view else \
        '<button class="action-btn add" onclick="openAddPhotos()">📥 Ajouter des photos</button>'
    photos_html = f"""
    <div class="section">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <h3 style="margin-bottom:0;">📸 Photos ({photo_count}/10)</h3>
            {add_photos_btn}
        </div>
        <div class="photos-grid">{photos_inner or '<p style="color:#94a3b8;">Aucune photo pour le moment.</p>'}</div>
    </div>
    """

    # --- Analysis HTML (defensive: analysis may be a dict, a string, or None) ---
    analysis_html = ""
    analysis = claim.get("analysis")
    if analysis:
        if isinstance(analysis, dict):
            a_summary = analysis.get("summary") or "Pas de résumé disponible."
            a_type = analysis.get("damage_type") or analysis.get("detected_damage_type", "Non déterminé")
            a_severity = analysis.get("severity") or analysis.get("damage_severity", "Non déterminée")
            # Prefer the public recommendations list; fall back to legacy single string.
            a_recs = analysis.get("recommendations")
            if isinstance(a_recs, list) and a_recs:
                a_reco = "<ul style='margin:0;padding-left:18px;'>" + "".join(
                    f"<li>{r}</li>" for r in a_recs) + "</ul>"
            else:
                a_reco = analysis.get("recommendation", "Aucune recommandation.")
            a_analyzed = format_french_datetime(analysis.get("analyzed_at")) if analysis.get("analyzed_at") else None

            # Severity badge styling (handles both public High/Medium/Low and legacy low/high)
            sev_map = {
                "low": ("badge low", "Faible"),
                "medium": ("badge medium", "Modérée"),
                "high": ("badge high", "Élevée"),
                "critical": ("badge high", "Critique"),
            }
            sev_class, sev_label = sev_map.get(str(a_severity).lower(), ("badge medium", str(a_severity)))

            analyzed_row = ""
            if a_analyzed:
                analyzed_row = f'<p style="margin-top:15px;color:#94a3b8;font-size:12px;">Analyse effectuée le {a_analyzed}</p>'

            analysis_html = f"""
            <div class="section">
                <h3>🔍 Détection Avancée Fraude</h3>
                <div class="analysis-content">
                    <p style="margin-bottom:18px;">{a_summary}</p>
                    <div class="info-grid">
                        <div class="info-item">
                            <div class="info-label">Type de dégât détecté</div>
                            <div class="info-value">{a_type}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Gravité estimée</div>
                            <div class="info-value"><span class="{sev_class}">{sev_label}</span></div>
                        </div>
                    </div>
                    <div style="margin-top:18px;">
                        <div class="info-label">Recommandations</div>
                        <div class="info-value" style="font-weight:400;line-height:1.6;">{a_reco}</div>
                    </div>
                    {analyzed_row}
                </div>
            </div>
            """
        else:
            analysis_html = f"""
            <div class="section">
                <h3>🔍 Détection Avancée Fraude</h3>
                <div class="analysis-content">{str(analysis)}</div>
            </div>
            """
    else:
        analysis_html = """
        <div class="section">
            <h3>🔍 Détection Avancée Fraude</h3>
            <div class="analysis-content" style="color:#94a3b8;">
                Analyse non encore disponible. Elle sera générée automatiquement
                dès réception des photos du sinistre.
            </div>
        </div>
        """

    # --- Attestation HTML ---
    attestation_confirmed = bool(claim.get("attestation_confirmed", False))
    attestation_ts = claim.get("attestation_timestamp")
    if attestation_confirmed and attestation_ts:
        attestation_block = f"""
        <div class="attest-confirmed">
            <span class="attest-check">✅</span>
            <div>
                <div style="font-weight:600;color:#86efac;">Déclaration attestée</div>
                <div style="font-size:13px;color:#cbd5e1;margin-top:4px;">
                    Attesté le {format_french_datetime(attestation_ts)}
                </div>
            </div>
        </div>
        """
    else:
        attestation_block = """
        <button id="attestBtn" class="attest-btn" onclick="attestClaim()">
            ✅ J'atteste de la véracité de cette déclaration
        </button>
        <p style="margin-top:10px;color:#94a3b8;font-size:12px;">Statut : Non attesté</p>
        """

    # --- Conclusion section (professional insurance wording) ---
    # Bands: 0-20 faible (vert), 21-50 modéré (orange), 51+ élevé (rouge).
    if fraud_score <= 20:
        risk_label = "FAIBLE"
        risk_color = "#86efac"
        risk_sentence = (
            "Les éléments transmis sont cohérents et ne présentent pas d'indice de fraude significatif. "
            "Le dossier peut suivre le circuit d'indemnisation standard."
        )
        next_steps = (
            "Procéder à la validation du dossier et à l'évaluation chiffrée des indemnités. "
            "Une expertise sur site reste recommandée pour les sinistres de gravité élevée."
        )
    elif fraud_score <= 50:
        risk_label = "MODÉRÉ"
        risk_color = "#fbbf24"
        risk_sentence = (
            "Certains éléments appellent à la vigilance et méritent une vérification complémentaire "
            "avant toute prise de décision d'indemnisation."
        )
        next_steps = (
            "Diligenter une expertise contradictoire sur site et solliciter des justificatifs additionnels "
            "(factures, témoignages, devis) auprès de l'assuré avant validation."
        )
    else:
        risk_label = "ÉLEVÉ"
        risk_color = "#fca5a5"
        risk_sentence = (
            "Le dossier présente plusieurs indices de fraude potentielle nécessitant un traitement renforcé. "
            "Aucune indemnisation ne devrait être engagée en l'état."
        )
        next_steps = (
            "Suspendre le traitement, transmettre le dossier au service anti-fraude et mandater une expertise "
            "approfondie. Documenter l'ensemble des anomalies relevées avant tout contact avec l'assuré."
        )

    if isinstance(analysis, dict):
        concl_damage = analysis.get("detected_damage_type", claim.get("damage_type", "non déterminé"))
        concl_sev = analysis.get("damage_severity", "non déterminée")
        damage_sentence = (
            f"L'analyse visuelle conclut à un sinistre de type « {concl_damage} » "
            f"d'une gravité estimée « {concl_sev} »."
        )
    else:
        damage_sentence = (
            f"Le sinistre déclaré est de type « {claim.get('damage_type', 'non déterminé')} ». "
            "L'évaluation automatique des dégâts est en attente des photos."
        )

    # The expert conclusion exposes the fraud score and internal assessment —
    # it is INTERNAL ONLY and must never be shown in the client (token) view.
    if client_view:
        conclusion_html = ""
    else:
        conclusion_html = f"""
    <div class="section">
        <h3>📑 Conclusion de l'expert</h3>
        <div class="conclusion-content">
            <p><strong>Évaluation des dégâts.</strong> {damage_sentence}</p>
            <p style="margin-top:14px;">
                <strong>Risque de fraude.</strong>
                <span style="color:{risk_color};font-weight:700;">{risk_label}</span>
                (score {fraud_score_display}/100). {risk_sentence}
            </p>
            <p style="margin-top:14px;"><strong>Prochaines étapes recommandées.</strong> {next_steps}</p>
            <p style="margin-top:18px;font-size:12px;color:#94a3b8;font-style:italic;">
                Cette synthèse est une aide à la décision générée automatiquement par AssuranceIA™
                et ne se substitue pas à l'appréciation d'un expert mandaté.
            </p>
        </div>
    </div>
    """

    # --- Build view-mode-dependent pieces (PDF link + action buttons) -------
    if client_view:
        page_title = "Ma Déclaration"
        # Client view has NO PDF access: no download button, no email button.
        # Client sees only their declaration data, photos and analysis.
        actions_bar = ""
    else:
        page_title = "Détail du Sinistre"
        pdf_url = f"/claim/{claim_id}/pdf"
        actions_bar = (
            f'<a class="action-btn pdf" href="{pdf_url}">📥 Télécharger le PDF</a>'
            '<button class="action-btn email" onclick="sendPdfEmail()">📧 Envoyer le PDF</button>'
            '<button class="action-btn add" onclick="openAddPhotos()">📥 Ajouter des photos</button>'
        )

    html = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Détail Sinistre - {claim_id}</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #e2e8f0;
                padding: 40px 20px;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            header {{
                margin-bottom: 40px;
            }}
            h1 {{
                font-size: 32px;
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
            }}
            .claim-info {{
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.9) 100%);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 12px;
                padding: 30px;
                margin-bottom: 30px;
                backdrop-filter: blur(10px);
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
            }}
            .info-item {{
                border-left: 3px solid #3b82f6;
                padding-left: 15px;
            }}
            .info-label {{
                font-size: 12px;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 5px;
            }}
            .info-value {{
                font-size: 16px;
                color: #f1f5f9;
                font-weight: 600;
            }}
            .badge {{
                display: inline-block;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }}
            .badge.low {{
                background: rgba(34, 197, 94, 0.2);
                color: #86efac;
            }}
            .badge.medium {{
                background: rgba(248, 113, 113, 0.2);
                color: #fca5a5;
            }}
            .badge.high {{
                background: rgba(239, 68, 68, 0.2);
                color: #fca5a5;
            }}
            .section {{
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.9) 100%);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 12px;
                padding: 30px;
                margin-bottom: 30px;
                backdrop-filter: blur(10px);
            }}
            h3 {{
                font-size: 20px;
                margin-bottom: 20px;
                color: #60a5fa;
            }}
            #map {{
                width: 100%;
                height: 400px;
                border-radius: 8px;
                border: 2px solid rgba(59, 130, 246, 0.3);
            }}
            .photos-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            .claim-photo {{
                width: 100%;
                height: 200px;
                object-fit: cover;
                border-radius: 8px;
                border: 2px solid rgba(148, 163, 184, 0.2);
            }}
            .analysis-content {{
                background: rgba(59, 130, 246, 0.05);
                border: 1px solid rgba(59, 130, 246, 0.2);
                border-radius: 8px;
                padding: 20px;
                color: #cbd5e1;
                line-height: 1.6;
            }}
            .gps-status {{
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
                font-size: 14px;
            }}
            .gps-status.verified {{
                background: rgba(34, 197, 94, 0.1);
                border: 1px solid rgba(34, 197, 94, 0.3);
                color: #86efac;
            }}
            .gps-status.mismatch {{
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: #fca5a5;
            }}
            .attest-btn {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
                color: #ffffff;
                border: none;
                border-radius: 10px;
                padding: 16px 28px;
                font-size: 16px;
                font-weight: 700;
                cursor: pointer;
                box-shadow: 0 6px 18px rgba(34, 197, 94, 0.35);
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }}
            .attest-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 10px 24px rgba(34, 197, 94, 0.45);
            }}
            .attest-btn:disabled {{
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }}
            .attest-confirmed {{
                display: flex;
                align-items: center;
                gap: 14px;
                background: rgba(34, 197, 94, 0.1);
                border: 1px solid rgba(34, 197, 94, 0.35);
                border-radius: 10px;
                padding: 18px 22px;
            }}
            .attest-check {{ font-size: 28px; }}
            .conclusion-content {{
                background: rgba(139, 92, 246, 0.06);
                border: 1px solid rgba(139, 92, 246, 0.25);
                border-left: 4px solid #8b5cf6;
                border-radius: 8px;
                padding: 24px;
                color: #cbd5e1;
                line-height: 1.6;
            }}
            .actions-bar {{
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                margin-top: 16px;
            }}
            .action-btn {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                border: none;
                border-radius: 8px;
                padding: 12px 18px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                color: #fff;
                text-decoration: none;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }}
            .action-btn:hover {{ transform: translateY(-2px); }}
            .action-btn.pdf {{ background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); }}
            .action-btn.email {{ background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%); }}
            .action-btn.add {{ background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); }}
            .modal-overlay {{
                display: none;
                position: fixed;
                inset: 0;
                background: rgba(0,0,0,0.6);
                z-index: 1000;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }}
            .modal-overlay.open {{ display: flex; }}
            .modal {{
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border: 1px solid rgba(148,163,184,0.3);
                border-radius: 14px;
                padding: 28px;
                max-width: 520px;
                width: 100%;
            }}
            .modal h3 {{ margin-bottom: 16px; }}
            .modal input[type="file"] {{
                width: 100%;
                padding: 12px;
                background: rgba(15,23,42,0.6);
                border: 2px dashed rgba(148,163,184,0.3);
                border-radius: 8px;
                color: #e2e8f0;
            }}
            .modal-actions {{ display:flex; gap:10px; margin-top:18px; }}
            .modal-actions button {{ flex:1; }}
            .btn-secondary {{ background: rgba(148,163,184,0.25); }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🔐 {page_title}</h1>
                <p style="color: #cbd5e1; font-size: 16px;">Référence: <strong>{claim_id}</strong></p>
                <div class="actions-bar">
                    {actions_bar}
                </div>
            </header>

            <div class="claim-info">
                <h3 style="color: #60a5fa; margin-bottom: 20px;">📋 Informations Principales</h3>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Référence</div>
                        <div class="info-value">{claim_id}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Email</div>
                        <div class="info-value">{claim.get('user_email', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Prénom</div>
                        <div class="info-value">{claim.get('firstname', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Nom</div>
                        <div class="info-value">{claim.get('lastname', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Téléphone</div>
                        <div class="info-value">{claim.get('phone', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Type de Dégâts</div>
                        <div class="info-value">{claim.get('damage_type', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Adresse</div>
                        <div class="info-value">{claim.get('address', 'N/A')}</div>
                    </div>
                    {fraud_score_row}
                    <div class="info-item">
                        <div class="info-label">Date</div>
                        <div class="info-value">{date_display}</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h3>📍 Géolocalisation</h3>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Latitude</div>
                        <div class="info-value">{gps_lat}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Longitude</div>
                        <div class="info-value">{gps_lon}</div>
                    </div>
                </div>
                <div id="map"></div>
                <div class="gps-status {gps_status_class}">
                    {gps_status_text}{gps_distance_text}
                </div>
            </div>

            <div class="section">
                <h3>📝 Description</h3>
                <p style="color: #cbd5e1; line-height: 1.6;">{claim.get('description', 'Pas de description')}</p>
            </div>

            {photos_html}
            {analysis_html}

            <div class="section">
                <h3>🖋️ Attestation sur l'honneur</h3>
                <p style="color:#cbd5e1;line-height:1.6;margin-bottom:18px;">
                    En attestant, le déclarant certifie l'exactitude et la sincérité
                    des informations fournies dans la présente déclaration de sinistre.
                </p>
                <div id="attestContainer">
                    {attestation_block}
                </div>
            </div>

            {conclusion_html}
        </div>

        <!-- Add Photos Modal -->
        <div class="modal-overlay" id="addPhotosModal">
            <div class="modal">
                <h3 style="color:#60a5fa;">📥 Ajouter des photos</h3>
                <p style="color:#cbd5e1;font-size:14px;margin-bottom:14px;">
                    Sélectionnez une ou plusieurs photos (max 5 Mo chacune). Les doublons sont ignorés.
                </p>
                <input type="file" id="addPhotosInput" multiple accept="image/*">
                <p id="addPhotosStatus" style="margin-top:12px;font-size:14px;color:#94a3b8;"></p>
                <div class="modal-actions">
                    <button class="action-btn add" id="addPhotosUploadBtn" onclick="uploadAddedPhotos()">Envoyer</button>
                    <button class="action-btn btn-secondary" onclick="closeAddPhotos()">Annuler</button>
                </div>
            </div>
        </div>

        <script>
        var CLAIM_ID = '{claim_id}';

        function openAddPhotos() {{
            document.getElementById("addPhotosModal").classList.add("open");
        }}
        function closeAddPhotos() {{
            document.getElementById("addPhotosModal").classList.remove("open");
            document.getElementById("addPhotosStatus").textContent = "";
            document.getElementById("addPhotosInput").value = "";
        }}

        function uploadAddedPhotos() {{
            var input = document.getElementById("addPhotosInput");
            var status = document.getElementById("addPhotosStatus");
            var btn = document.getElementById("addPhotosUploadBtn");
            if (!input.files || input.files.length === 0) {{
                status.textContent = "Veuillez sélectionner au moins une photo.";
                return;
            }}
            var fd = new FormData();
            for (var i = 0; i < input.files.length; i++) {{
                if (input.files[i].size > 5 * 1024 * 1024) {{
                    status.textContent = "Photo trop volumineuse (max 5 Mo): " + input.files[i].name;
                    return;
                }}
                fd.append("photos", input.files[i]);
            }}
            btn.disabled = true;
            status.textContent = "⏳ Envoi et analyse en cours...";
            fetch("/claim/" + CLAIM_ID + "/add-photos", {{ method: "POST", body: fd }})
                .then(function(r) {{ return r.json(); }})
                .then(function(data) {{
                    btn.disabled = false;
                    if (data && typeof data.photo_count !== "undefined") {{
                        status.innerHTML = "✅ " + (data.added || 0) + " photo(s) ajoutée(s). " +
                            "Total: " + data.photo_count + "/10. Score fraude: " + data.fraud_score + "/100." +
                            "<br>Rechargement...";
                        setTimeout(function() {{ window.location.reload(); }}, 1400);
                    }} else {{
                        status.textContent = (data && data.detail) || "Erreur lors de l'ajout.";
                    }}
                }})
                .catch(function() {{
                    btn.disabled = false;
                    status.textContent = "Erreur réseau.";
                }});
        }}

        function sendPdfEmail() {{
            if (!confirm("Envoyer le rapport PDF à l'assureur ?")) return;
            fetch("/claim/" + CLAIM_ID + "/email-pdf", {{ method: "POST" }})
                .then(function(r) {{ return r.json(); }})
                .then(function(data) {{
                    alert((data && data.message) || "Demande traitée.");
                }})
                .catch(function() {{ alert("Erreur réseau lors de l'envoi."); }});
        }}

        function attestClaim() {{
            if (!confirm("Confirmez-vous attester de la véracité de cette déclaration ? Cette action est définitive.")) {{
                return;
            }}
            var btn = document.getElementById("attestBtn");
            if (btn) {{ btn.disabled = true; btn.textContent = "Enregistrement..."; }}
            fetch("/claims/" + CLAIM_ID + "/attest", {{
                method: "POST",
                headers: {{ "Content-Type": "application/x-www-form-urlencoded" }},
                body: "attestation_confirmed=true"
            }}).then(function(r) {{ return r.json(); }}).then(function(data) {{
                if (data && data.attestation_confirmed) {{
                    var c = document.getElementById("attestContainer");
                    c.innerHTML = '<div class="attest-confirmed"><span class="attest-check">✅</span>' +
                        '<div><div style="font-weight:600;color:#86efac;">Déclaration attestée</div>' +
                        '<div style="font-size:13px;color:#cbd5e1;margin-top:4px;">Attesté le ' +
                        data.attestation_timestamp_display + '</div></div></div>';
                }} else {{
                    alert((data && data.detail) || "Erreur lors de l'attestation.");
                    if (btn) {{ btn.disabled = false; btn.textContent = "✅ J'atteste de la véracité de cette déclaration"; }}
                }}
            }}).catch(function() {{
                alert("Erreur réseau lors de l'attestation.");
                if (btn) {{ btn.disabled = false; btn.textContent = "✅ J'atteste de la véracité de cette déclaration"; }}
            }});
        }}

        // Initialize map
        var lat = parseFloat('{js_lat}');
        var lon = parseFloat('{js_lon}');

        if (!isNaN(lat) && !isNaN(lon)) {{
            const map = L.map('map').setView([lat, lon], 13);
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors',
                maxZoom: 19
            }}).addTo(map);
            L.marker([lat, lon]).addTo(map)
                .bindPopup('📍 Position du sinistre')
                .openPopup();
        }} else {{
            document.getElementById('map').innerHTML = '<p style="padding: 20px; text-align: center; color: #cbd5e1;">GPS non disponible</p>';
        }}
        </script>
    </body>
    </html>
    """

    return Response(content=html, media_type="text/html")


@app.get("/report/{token}")
def download_report_by_token(token: str):
    """Download PDF report using unique token (for client/insurer sharing)."""
    if token not in token_to_claim:
        raise HTTPException(status_code=404, detail="Lien invalide ou expiré")

    claim_id = token_to_claim[token]
    claim = claims_db.get(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Sinistre non trouvé")

    pdf_bytes = build_claim_pdf(claim)
    filename = f"rapport_{claim_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/claims/{claim_id}/send-links")
def send_claim_links(claim_id: str):
    """Generate and return shareable links for claim (PDF ready to send to client + insurer)."""
    if claim_id not in claims_db:
        raise HTTPException(status_code=404, detail="Sinistre non trouvé")

    claim = claims_db[claim_id]
    token = claim.get("unique_token")

    if not token:
        raise HTTPException(status_code=400, detail="Token non disponible")

    # Base URL (will be Render in prod)
    base_url = os.getenv("APP_URL", "https://assurancia-api-2.onrender.com")

    return {
        "claim_id": claim_id,
        "client_email": claim.get("user_email"),
        "links": {
            "declare": f"{base_url}/declare/{token}",
            "download_pdf": f"{base_url}/report/{token}",
        },
        "message": "Liens uniques générés. Envoyez ces liens au client et à l'assurance/courtier.",
        "emails": {
            "client": {
                "to": claim.get("user_email"),
                "subject": f"Votre sinistre {claim_id} - Accès à votre déclaration",
                "body": f"Cliquez ici pour accéder à votre déclaration: {base_url}/declare/{token}\nTélécharger le rapport PDF: {base_url}/report/{token}"
            },
            "insurer": {
                "to": "courtier@example.com",  # Replace with actual courtier email
                "subject": f"Sinistre reçu: {claim_id}",
                "body": f"Rapport disponible: {base_url}/report/{token}"
            }
        }
    }


@app.post("/declare/{token}/submit")
@limiter.limit("5/minute")
async def submit_declaration(request: Request,
    token: str,
    user_email: str = Form(""),
    firstname: str = Form(""),
    lastname: str = Form(""),
    phone: str = Form(""),
    damage_type: str = Form(""),
    address: str = Form(""),
    description: str = Form(""),
    phone_gps_lat: str = Form(None),
    phone_gps_lon: str = Form(None),
    attestation_confirmed: bool = Form(False),
    photos: list[UploadFile] = File(default=[]),
):
    """
    Client submits declaration via token link.
    Creates claim + stores insurer email + verifies GPS location.

    Phase 1: SANITIZES all user inputs before storing.
    NOTE: parameters use Form(...) because the frontend submits multipart/form-data
    (FormData). Plain str params would be read as query params and arrive empty.
    """
    client_ip = get_client_ip(request)

    # Log exactly what arrived so we can never again wonder if data was received.
    logger.info(
        "📥 submit_declaration token=%s | email=%r firstname=%r lastname=%r "
        "phone=%r type=%r address=%r desc_len=%d gps=(%s,%s) | IP=%s",
        token, user_email, firstname, lastname, phone, damage_type,
        address, len(description or ""), phone_gps_lat, phone_gps_lon, client_ip,
    )

    if token not in declaration_links:
        raise HTTPException(status_code=404, detail="Lien invalide ou expiré")

    link_data = declaration_links[token]
    insurer_email = link_data["insurer_email"]
    insurer_id = link_data.get("insurer_id", "UNKNOWN")  # MULTI-TENANT: Get insurer from link

    # --- PHASE 1: SANITIZE ALL USER INPUTS ---
    try:
        user_email = sanitize_email(user_email)
        if not user_email:
            raise HTTPException(status_code=400, detail="Email invalide")

        firstname = sanitize_text(firstname, max_length=100)
        if not firstname:
            raise HTTPException(status_code=400, detail="Prénom requis")

        lastname = sanitize_text(lastname, max_length=100)
        if not lastname:
            raise HTTPException(status_code=400, detail="Nom requis")

        phone = sanitize_phone(phone)
        if not phone:
            raise HTTPException(status_code=400, detail="Téléphone invalide")

        address = sanitize_address(address, max_length=500)
        if not address:
            raise HTTPException(status_code=400, detail="Adresse invalide")

        description = sanitize_text(description, max_length=2000)
        if not description:
            raise HTTPException(status_code=400, detail="Description requise")

        damage_type = sanitize_text(damage_type, max_length=100)

        # Validate no XSS/injection attempts
        if not is_safe(user_email) or not is_safe(firstname) or not is_safe(lastname) \
           or not is_safe(address) or not is_safe(description):
            audit_submit_declaration(insurer_id, "UNKNOWN", client_ip)
            raise HTTPException(status_code=400, detail="Contenu invalide détecté")

        logger.info("✅ All inputs sanitized for claim submission")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Input sanitization failed: %r", e)
        raise HTTPException(status_code=400, detail="Erreur lors du traitement des données")

    # Create claim
    claim_id = f"CLM-{datetime.now().year}-{uuid.uuid4().hex[:6].upper()}"
    unique_token = secrets.token_urlsafe(32)

    location = geocode_address(address)

    # Convert and store phone GPS
    phone_gps = None
    gps_verification = None
    if phone_gps_lat and phone_gps_lon:
        try:
            phone_gps = {"latitude": float(phone_gps_lat), "longitude": float(phone_gps_lon)}
            gps_verification = check_gps_location_match(location, phone_gps)
        except:
            pass

    fraud_history = check_fraud_history(user_email, address)

    claim_data = {
        "claim_id": claim_id,
        "unique_token": unique_token,
        "insurer_id": insurer_id,  # MULTI-TENANT: This claim belongs to this insurer
        "user_email": user_email,
        "firstname": firstname,
        "lastname": lastname,
        "phone": phone,
        "damage_type": damage_type,
        "address": address,
        "description": description,
        "location": location,
        "phone_gps": phone_gps,  # Store phone GPS from client
        "gps_verification": gps_verification,  # Verify against address
        "fraud_history": fraud_history,
        "insurer_email": insurer_email,  # Store for auto PDF sending
        "photos": [],
        "analysis": None,
        "status": "open",
        "created_at": datetime.now().isoformat(),
        # Attestation: may be confirmed at submission time, or later via /attest.
        "attestation_confirmed": bool(attestation_confirmed),
        "attestation_timestamp": datetime.now().isoformat() if attestation_confirmed else None,
    }

    # Always store in memory first (guarantees the claim is retrievable even if
    # Mongo is down — endpoints fall back to claims_db).
    claims_db[claim_id] = claim_data
    token_to_claim[unique_token] = claim_id
    # Token remains valid for reuse - don't mark as "completed"
    # This allows the same token to be used for multiple declarations

    # ---- PHASE 1: PHOTO RATE LIMITING ----
    # Check: max 10 photos, total size <= 100 MB
    photos_list = photos or []
    if len(photos_list) > 10:
        raise HTTPException(status_code=413, detail="Maximum 10 photos autorisées")

    total_photo_bytes = 0
    for upload in photos_list:
        try:
            contents = await upload.read()
            total_photo_bytes += len(contents)
            # Reset file pointer for later reading
            await upload.seek(0)
        except Exception as e:
            logger.warning("⚠️ Could not read upload size %r: %r", getattr(upload, "filename", "?"), e)
            continue

    MAX_TOTAL_PHOTOS_BYTES = 100 * 1024 * 1024  # 100 MB total
    if total_photo_bytes > MAX_TOTAL_PHOTOS_BYTES:
        size_mb = total_photo_bytes / (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Photos trop volumineuses (total {size_mb:.1f} MB, max 100 MB)")

    logger.info("📸 Rate limiting check passed: %d photos, %.1f MB total", len(photos_list), total_photo_bytes / (1024 * 1024))

    # ---- Process inline photos from the FormData (optional) ----------------
    # Photos uploaded with the declaration are read, validated, EXIF-parsed and
    # attached to the claim so Vision analysis + EXIF fraud checks can run.
    photo_count = 0
    for upload in (photos_list or []):
        try:
            contents = await upload.read()
        except Exception as e:
            logger.warning("⚠️ Could not read uploaded photo %r: %r", getattr(upload, "filename", "?"), e)
            continue
        if not contents:
            continue
        if len(contents) > MAX_PHOTO_BYTES:
            logger.warning("⚠️ Skipping oversized photo %r (%d bytes)", upload.filename, len(contents))
            continue
        media_type = normalize_media_type(upload.content_type)
        if media_type is None:
            logger.warning("⚠️ Skipping unsupported photo type %r (%s)", upload.filename, upload.content_type)
            continue
        try:
            photo_entry = {
                "filename": upload.filename,
                "content_type": media_type,
                "data": base64.b64encode(contents).decode("utf-8"),
                "gps": extract_gps_from_exif(contents),
                "exif_datetime": extract_exif_datetime(contents),
                "image_hash": calculate_image_hash(contents),
            }
            claim_data["photos"].append(photo_entry)
            photo_count += 1
        except Exception as e:
            logger.warning("⚠️ Failed to process photo %r: %r", upload.filename, e)

    logger.info("📸 submit_declaration attached %d photo(s) to %s", photo_count, claim_id)

    # Log declaration submission
    audit_submit_declaration(insurer_id, claim_id, client_ip)

    # ---- Claude Vision analysis (best-effort, never blocks claim creation) --
    # On any error, analyze_claim_photos_with_claude returns a structured
    # "analysis pending" dict — the claim always survives.
    analysis = None
    try:
        analysis = analyze_claim_photos_with_claude(claim_id, claim_data["photos"])
    except Exception as e:
        logger.error("❌ Vision analysis failed in submit_declaration claim_id=%s: %r", claim_id, e)
        analysis = {
            "summary": "Analyse en attente (erreur technique).",
            "detected_damage_type": damage_type,
            "damage_severity": "unknown",
            "analyzed_at": datetime.now().isoformat(),
        }
        claim_data["analysis"] = analysis

    # Normalize the analysis into the documented public structure and store it.
    public_analysis = build_public_analysis(analysis, claim_data)
    claim_data["analysis"] = public_analysis

    # ---- Fraud scoring (always returns a number) ---------------------------
    additional_factors = compute_additional_fraud_factors(claim_data)
    fraud_score = calculate_fraud_score(claim_data, analysis, additional_factors)
    claim_data["fraud_score"] = fraud_score
    claim_data["fraud_factors"] = additional_factors

    # ---- Persist analysis + fraud_score to MongoDB AND in-memory -----------
    _persist_claim_field(claim_id, {
        "photos": claim_data["photos"],
        "analysis": public_analysis,
        "fraud_score": fraud_score,
        "fraud_factors": additional_factors,
    })

    # Save the full claim document to MongoDB. insert_one() mutates the dict by
    # adding "_id" (non-JSON-serializable ObjectId), so insert a COPY.
    mongo_saved = False
    mongo_error = None
    try:
        # Remove any prior _id from a possible _persist_claim_field upsert path.
        doc = {k: v for k, v in claim_data.items() if k != "_id"}
        existing = claims_collection.find_one({"claim_id": claim_id})
        if existing:
            claims_collection.update_one({"claim_id": claim_id}, {"$set": doc})
        else:
            claims_collection.insert_one(dict(doc))
        mongo_saved = True
        logger.info("✅ MongoDB save OK: claim_id=%s", claim_id)
    except Exception as e:
        mongo_error = str(e)
        logger.error("❌ MongoDB save FAILED for claim_id=%s: %r", claim_id, e)
        logger.error("   → Claim is still available in-memory (fallback). Fix Mongo to persist.")

    base_url = os.getenv("APP_URL", "https://assurancia-api-2.onrender.com")
    client_view_url = f"/my-claim/{unique_token}"

    return {
        "mongo_saved": mongo_saved,
        "mongo_error": mongo_error,
        "claim_id": claim_id,  # internal reference (dashboard / insurer use)
        "unique_token": unique_token,  # client authorization token (unguessable)
        # Client-facing link (token-authenticated, no claim_id, no PDF exposed):
        "client_view_url": client_view_url,
        "client_links": {
            "view": f"{base_url}{client_view_url}",
        },
        "gps_verification": gps_verification,
        "photo_count": photo_count,
        "analysis": public_analysis,
        "fraud_score": fraud_score,
        "message": "✅ Déclaration créée ! Consultez votre déclaration.",
        "success_message_html": (
            f'Déclaration créée! '
            f'<a href="{client_view_url}">Voir ma déclaration</a>'
        ),
        "next_step": f"Consultez votre déclaration via {client_view_url}",
        "insurer_will_receive": f"PDF sera envoyé à {insurer_email} après analyse",
        "attestation_confirmed": bool(attestation_confirmed),
        "attestation_timestamp": claim_data["attestation_timestamp"],
    }


@app.post("/claims/{claim_id}/attest")
async def attest_claim(claim_id: str, attestation_confirmed: bool = Form(True)):
    """Record the declarant's attestation of truthfulness for a claim.

    Idempotent: once a claim is attested, re-posting returns the existing
    attestation timestamp without overwriting it (HTTP 200, not an error).
    Timestamp is stored in ISO format and returned both raw and formatted.
    """
    # Resolve the claim from in-memory first, then MongoDB.
    claim = claims_db.get(claim_id)
    if claim is None:
        try:
            claim = claims_collection.find_one({"claim_id": claim_id})
        except Exception:
            claim = None
    if not claim:
        raise HTTPException(status_code=404, detail="Sinistre non trouvé")

    if not attestation_confirmed:
        raise HTTPException(status_code=400, detail="Confirmation d'attestation requise.")

    # Idempotency: don't re-attest if already attested.
    if claim.get("attestation_confirmed") and claim.get("attestation_timestamp"):
        ts = claim["attestation_timestamp"]
        return {
            "claim_id": claim_id,
            "attestation_confirmed": True,
            "attestation_timestamp": ts,
            "attestation_timestamp_display": format_french_datetime(ts),
            "already_attested": True,
            "message": "Déclaration déjà attestée.",
        }

    timestamp = datetime.now().isoformat()
    _persist_claim_field(claim_id, {
        "attestation_confirmed": True,
        "attestation_timestamp": timestamp,
    })

    logger.info("🖋️ Claim %s attested at %s", claim_id, timestamp)
    return {
        "claim_id": claim_id,
        "attestation_confirmed": True,
        "attestation_timestamp": timestamp,
        "attestation_timestamp_display": format_french_datetime(timestamp),
        "already_attested": False,
        "message": "Déclaration attestée avec succès.",
    }


def _load_claim(claim_id: str) -> dict | None:
    """Resolve a claim from in-memory first, then MongoDB. Returns None if missing."""
    claim = claims_db.get(claim_id)
    if claim is not None:
        return claim
    try:
        doc = claims_collection.find_one({"claim_id": claim_id}, {"_id": 0})
        return doc
    except Exception:
        return None


@app.get("/claim/{claim_id}/pdf")
def download_claim_pdf(claim_id: str, insurer_id: str = Depends(verify_insurer_token)):
    """Generate and return the professional claim PDF as a file download."""
    claim = _load_claim(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Sinistre non trouvé")
    pdf_bytes = generate_claim_pdf(claim_id, claim)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="sinistre_{claim_id}.pdf"'},
    )


@app.post("/claim/{claim_id}/email-pdf")
def email_claim_pdf(claim_id: str, insurer_id: str = Depends(verify_insurer_token)):
    """Generate the claim PDF and send it to the insurer email via SMTP.

    Falls back to logging (and returns email_sent=False) when no SMTP service
    is configured, so the endpoint never breaks the flow.
    """
    claim = _load_claim(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Sinistre non trouvé")

    insurer_email = claim.get("insurer_email") or claim.get("user_email")
    if not insurer_email:
        raise HTTPException(status_code=400, detail="Aucune adresse email destinataire disponible.")

    pdf_bytes = generate_claim_pdf(claim_id, claim)

    smtp_host = os.getenv("SMTP_HOST")
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_from = os.getenv("SMTP_FROM", smtp_user or "noreply@assurancia.app")

    if not (smtp_host and smtp_user and smtp_pass):
        logger.info("📧 email-pdf requested for %s but no SMTP configured. "
                    "Would send to %s (%d bytes).", claim_id, insurer_email, len(pdf_bytes))
        return {
            "email_sent": False,
            "claim_id": claim_id,
            "insurer_email": insurer_email,
            "message": "Service email non configuré (SMTP_HOST/USER/PASS). PDF généré mais non envoyé.",
            "pdf_size_bytes": len(pdf_bytes),
        }

    try:
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = f"AssuranceIA - Rapport de sinistre {claim_id}"
        msg["From"] = smtp_from
        msg["To"] = insurer_email
        msg.set_content(
            f"Bonjour,\n\nVeuillez trouver ci-joint le rapport de sinistre {claim_id}.\n\n"
            f"Score de fraude: {claim.get('fraud_score', 'N/A')}/100\n\n"
            "Cordialement,\nAssuranceIA"
        )
        msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf",
                           filename=f"sinistre_{claim_id}.pdf")

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        logger.info("📧 PDF emailed for %s to %s", claim_id, insurer_email)
        return {
            "email_sent": True,
            "claim_id": claim_id,
            "insurer_email": insurer_email,
            "message": f"Rapport PDF envoyé à {insurer_email}.",
        }
    except Exception as e:
        logger.error("❌ email-pdf SMTP send failed for %s: %r", claim_id, e)
        return {
            "email_sent": False,
            "claim_id": claim_id,
            "insurer_email": insurer_email,
            "message": f"Échec de l'envoi email: {str(e)[:120]}",
        }


@app.post("/claim/{claim_id}/add-photos")
async def add_photos_to_claim(claim_id: str, photos: list[UploadFile] = File(default=[])):
    """Append new photos to an existing claim, dedup by MD5, re-run Vision + fraud.

    - Validates each photo (size <= 5MB, supported media type).
    - Skips duplicates already present in the claim (same MD5 hash).
    - Re-runs Claude Vision analysis and recomputes the fraud score.
    - Persists the updated photos / analysis / fraud_score to MongoDB + memory.
    """
    # Ensure claim is loaded into claims_db (so persistence + analysis work).
    claim = claims_db.get(claim_id)
    if claim is None:
        doc = _load_claim(claim_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Sinistre non trouvé")
        claims_db[claim_id] = doc
        claim = doc

    if not isinstance(claim.get("photos"), list):
        claim["photos"] = []

    existing_hashes = {
        p.get("image_hash") for p in claim["photos"]
        if isinstance(p, dict) and p.get("image_hash")
    }

    added = 0
    skipped_duplicate = 0
    skipped_invalid = 0

    for upload in (photos or []):
        try:
            contents = await upload.read()
        except Exception as e:
            logger.warning("⚠️ add-photos could not read %r: %r", getattr(upload, "filename", "?"), e)
            skipped_invalid += 1
            continue
        if not contents:
            skipped_invalid += 1
            continue
        if len(contents) > MAX_PHOTO_BYTES:
            logger.warning("⚠️ add-photos oversized %r (%d bytes)", upload.filename, len(contents))
            skipped_invalid += 1
            continue
        media_type = normalize_media_type(upload.content_type)
        if media_type is None:
            skipped_invalid += 1
            continue

        image_hash = calculate_image_hash(contents)
        if image_hash in existing_hashes:
            skipped_duplicate += 1
            continue
        existing_hashes.add(image_hash)

        try:
            claim["photos"].append({
                "filename": upload.filename,
                "content_type": media_type,
                "data": base64.b64encode(contents).decode("utf-8"),
                "gps": extract_gps_from_exif(contents),
                "exif_datetime": extract_exif_datetime(contents),
                "image_hash": image_hash,
            })
            added += 1
        except Exception as e:
            logger.warning("⚠️ add-photos failed to process %r: %r", upload.filename, e)
            skipped_invalid += 1

    # Persist the new photos array immediately.
    _persist_claim_field(claim_id, {"photos": claim["photos"]})

    # Re-run Vision + recompute fraud score (best-effort, never raises).
    analysis = None
    try:
        analysis = analyze_claim_photos_with_claude(claim_id, claim["photos"])
    except Exception as e:
        logger.error("❌ add-photos Vision analysis failed claim_id=%s: %r", claim_id, e)

    public_analysis = build_public_analysis(analysis, claim)
    claim["analysis"] = public_analysis

    additional_factors = compute_additional_fraud_factors(claim)
    fraud_score = calculate_fraud_score(claim, analysis, additional_factors)
    claim["fraud_score"] = fraud_score
    claim["fraud_factors"] = additional_factors

    _persist_claim_field(claim_id, {
        "photos": claim["photos"],
        "analysis": public_analysis,
        "fraud_score": fraud_score,
        "fraud_factors": additional_factors,
    })

    logger.info("📸 add-photos claim=%s added=%d dup=%d invalid=%d total=%d fraud=%s",
                claim_id, added, skipped_duplicate, skipped_invalid,
                len(claim["photos"]), fraud_score)

    return {
        "claim_id": claim_id,
        "added": added,
        "skipped_duplicate": skipped_duplicate,
        "skipped_invalid": skipped_invalid,
        "photo_count": len(claim["photos"]),
        "analysis": public_analysis,
        "fraud_score": fraud_score,
        "message": f"{added} photo(s) ajoutée(s). Total: {len(claim['photos'])} photo(s).",
    }


# ----------------------------------------------------------------------------
# PDF generation (pure-Python, no external binary needed)
# ----------------------------------------------------------------------------
def _decode_photo_to_image_bytes(photo) -> tuple | None:
    """Decode a stored photo into (PNG/JPEG bytes, ext) resized to max 800px width.

    Accepts photo dicts ({"data": <base64 or data-url>, "content_type": ...}) or
    bare data-URL strings. Returns None if the photo cannot be decoded.
    Always re-encodes through Pillow so fpdf2 gets a clean, supported image.
    """
    try:
        vinput = _photo_to_vision_input(photo)
        if not vinput or not vinput.get("data"):
            return None
        raw = base64.b64decode(vinput["data"])
        img = Image.open(BytesIO(raw))

        # Convert modes that fpdf2 / JPEG can't handle directly.
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Resize to a max width of 800px (keeps PDF small + renders cleanly).
        max_w = 800
        if img.width > max_w:
            ratio = max_w / float(img.width)
            new_h = max(1, int(img.height * ratio))
            img = img.resize((max_w, new_h))

        out = BytesIO()
        img.save(out, format="JPEG", quality=82)
        return out.getvalue(), "jpg"
    except Exception as e:
        logger.warning("⚠️ _decode_photo_to_image_bytes failed: %r", e)
        return None


def _build_qr_png(data: str) -> bytes | None:
    """Build a QR-code PNG for `data` if the optional `qrcode` lib is installed.

    Returns None when qrcode isn't available (no hard dependency added).
    """
    try:
        import qrcode  # optional dependency
        qr = qrcode.make(data)
        buf = BytesIO()
        qr.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def generate_claim_pdf(claim_id: str, claim_data: dict) -> bytes:
    """Build a professional multi-page PDF report for a claim using fpdf2.

    Includes: gradient-style header, claim reference, French-formatted date,
    all form fields, attestation status, embedded photos (max 800px), Claude
    Vision analysis, fraud score + assessment, conclusion, signature line,
    page numbers and an optional QR code linking to claim details.

    Never raises — on any error returns a minimal safe fallback PDF so the
    download endpoint can't break the claim flow.
    """
    try:
        return _generate_claim_pdf_impl(claim_id, claim_data or {})
    except Exception as e:
        logger.error("❌ generate_claim_pdf failed for %s: %r — returning fallback PDF", claim_id, e)
        try:
            fb = FPDF()
            fb.add_page()
            fb.set_font("Helvetica", "B", 16)
            fb.cell(0, 12, "AssuranceIA - Rapport de Sinistre", ln=True)
            fb.set_font("Helvetica", "", 11)
            fb.cell(0, 8, f"Reference: {claim_id}", ln=True)
            fb.multi_cell(0, 7, "Le rapport detaille n'a pas pu etre genere (erreur technique). "
                                "Veuillez reessayer ou contacter le support.")
            return bytes(fb.output())
        except Exception:
            # Absolute last resort: a tiny valid PDF.
            return b"%PDF-1.4\n%%EOF\n"


def _generate_claim_pdf_impl(claim_id: str, claim: dict) -> bytes:
    """Internal implementation for generate_claim_pdf (may raise; caller guards)."""

    APP_URL = os.getenv("APP_URL", "https://assurancia-api-2.onrender.com")

    def s(text) -> str:
        # fpdf2 core fonts are latin-1; strip unsupported chars (e.g. emojis) safely.
        return str(text).encode("latin-1", "replace").decode("latin-1")

    # Brand colors (blue -> purple gradient feel).
    BLUE = (59, 130, 246)
    PURPLE = (139, 92, 246)
    DARK = (30, 41, 59)
    GREY = (100, 116, 139)

    class ClaimPDF(FPDF):
        def header(self):
            # Gradient-style header band: draw thin vertical strips blue->purple.
            steps = 60
            band_h = 30
            for i in range(steps):
                t = i / float(steps - 1)
                r = int(BLUE[0] + (PURPLE[0] - BLUE[0]) * t)
                g = int(BLUE[1] + (PURPLE[1] - BLUE[1]) * t)
                b = int(BLUE[2] + (PURPLE[2] - BLUE[2]) * t)
                self.set_fill_color(r, g, b)
                x = i * (210.0 / steps)
                self.rect(x, 0, (210.0 / steps) + 0.5, band_h, "F")
            self.set_xy(10, 7)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 19)
            self.cell(0, 9, s("AssuranceIA(TM) - Rapport de Sinistre"), ln=True)
            self.set_xy(10, 18)
            self.set_font("Helvetica", "", 10)
            self.cell(0, 6, s(f"Reference: {claim.get('claim_id', claim_id)}"), ln=True)
            self.set_text_color(*DARK)
            self.set_y(38)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*GREY)
            self.cell(0, 5, s("AssuranceIA(TM) - Document genere automatiquement - aide a la decision"),
                      align="L")
            self.cell(0, 5, s(f"Page {self.page_no()}/{{nb}}"), align="R")

    pdf = ClaimPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.alias_nb_pages()
    pdf.add_page()

    def section(title: str):
        if pdf.get_y() > 250:
            pdf.add_page()
        pdf.ln(2)
        pdf.set_fill_color(238, 242, 255)
        pdf.set_draw_color(*PURPLE)
        pdf.set_line_width(0.3)
        pdf.set_text_color(*PURPLE)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, s(title), ln=True, fill=True, border="B")
        pdf.set_text_color(*DARK)
        pdf.set_font("Helvetica", "", 11)
        pdf.ln(2)

    def field(label: str, value):
        y = pdf.get_y()
        if y > 268:
            pdf.add_page()
            y = pdf.get_y()
        pdf.set_xy(12, y)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(48, 7, s(label), border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*DARK)
        pdf.set_xy(60, y)
        pdf.multi_cell(138, 7, s(value if value not in (None, "") else "-"))

    # ---- Date / reference summary -----------------------------------------
    created_at = claim.get("created_at")
    date_fr = format_french_datetime(created_at)

    section("Informations sur la declaration")
    field("Date de declaration", date_fr)
    field("Reference", claim.get("claim_id", claim_id))
    field("Statut", claim.get("status", "open"))

    # ---- Geolocation -------------------------------------------------------
    # Placed near the top (after the claim reference, before damage assessment).
    # Defensive: phone_gps / gps_verification may be None or have missing keys.
    section("Geolocalisation")  # rendered as "Geolocalisation" (PDF is latin-1, no emoji)

    phone_gps = claim.get("phone_gps") or {}
    gps_lat = phone_gps.get("latitude")
    gps_lon = phone_gps.get("longitude")
    try:
        coords_disp = f"Latitude: {float(gps_lat):.4f} deg | Longitude: {float(gps_lon):.4f} deg"
    except (TypeError, ValueError):
        coords_disp = "N/A"
    field("Coordonnees GPS", coords_disp)
    field("Adresse declaree", claim.get("address") or "N/A")

    gps_ver = claim.get("gps_verification") or {}
    # `matches` is the canonical key from check_gps_location_match; tolerate `match` too.
    gps_ok = gps_ver.get("matches")
    if gps_ok is None:
        gps_ok = gps_ver.get("match")
    distance_km = gps_ver.get("distance_km")
    if not gps_ver:
        verif_disp = "N/A (pas de donnees GPS)"
    elif gps_ok:
        if distance_km is not None:
            verif_disp = f"Verifie (Distance: {distance_km}km)"
        else:
            verif_disp = "Verifie"
    else:
        if distance_km is not None:
            verif_disp = f"Echoue - GPS ne correspond pas (Distance: {distance_km}km)"
        else:
            verif_disp = "Echoue - GPS ne correspond pas"
    field("Verification GPS", verif_disp)

    # ---- Form fields -------------------------------------------------------
    section("Informations du declarant")
    field("Email", claim.get("user_email"))
    field("Prenom", claim.get("firstname"))
    field("Nom", claim.get("lastname"))
    field("Telephone", claim.get("phone"))
    field("Type de degat", claim.get("damage_type"))
    field("Adresse", claim.get("address"))
    field("Description", claim.get("description"))

    # ---- Attestation -------------------------------------------------------
    section("Attestation sur l'honneur")
    if claim.get("attestation_confirmed"):
        ts = claim.get("attestation_timestamp")
        field("Statut", "Atteste")
        if ts:
            field("Atteste le", format_french_datetime(ts))
    else:
        field("Statut", "Non atteste")

    # ---- Vision analysis ---------------------------------------------------
    analysis = claim.get("analysis")
    section("Detection Avancee Fraude")
    if isinstance(analysis, dict) and analysis:
        summary = analysis.get("summary") or analysis.get("visible_damage")
        if summary:
            field("Resume", summary)
        field("Type detecte", analysis.get("damage_type") or analysis.get("detected_damage_type"))
        field("Gravite", analysis.get("severity") or analysis.get("damage_severity"))
        recs = analysis.get("recommendations")
        if isinstance(recs, list) and recs:
            field("Recommandations", " | ".join(str(r) for r in recs))
        elif analysis.get("recommendation"):
            field("Recommandation", analysis.get("recommendation"))
        indicators = analysis.get("fraud_indicators") or []
        if indicators:
            field("Indices fraude", "; ".join(str(i) for i in indicators))
        if analysis.get("analyzed_at"):
            field("Analyse le", format_french_datetime(analysis.get("analyzed_at")))
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(0, 7, s("Analyse non encore disponible."))
        pdf.set_font("Helvetica", "", 11)

    # ---- Fraud score + assessment -----------------------------------------
    fraud_score = claim.get("fraud_score", 0)
    try:
        fraud_score = float(fraud_score)
    except (TypeError, ValueError):
        fraud_score = 0.0
    fs_disp = int(fraud_score) if fraud_score == int(fraud_score) else fraud_score

    if fraud_score <= 20:
        risk_label, risk_rgb = "FAIBLE", (34, 197, 94)
        assessment = ("Les elements transmis sont coherents et ne presentent pas d'indice de fraude "
                      "significatif. Le dossier peut suivre le circuit d'indemnisation standard.")
    elif fraud_score <= 50:
        risk_label, risk_rgb = "MODERE", (234, 179, 8)
        assessment = ("Certains elements appellent a la vigilance et meritent une verification "
                      "complementaire avant toute decision d'indemnisation.")
    else:
        risk_label, risk_rgb = "ELEVE", (239, 68, 68)
        assessment = ("Le dossier presente plusieurs indices de fraude potentielle necessitant un "
                      "traitement renforce. Aucune indemnisation ne devrait etre engagee en l'etat.")

    section("Score de fraude")
    y = pdf.get_y()
    pdf.set_xy(12, y)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(48, 8, s("Niveau de risque"), border=0)
    pdf.set_fill_color(*risk_rgb)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(45, 8, s(f"{risk_label}  ({fs_disp}/100)"), align="C", fill=True, ln=True)
    pdf.set_text_color(*DARK)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(12)
    pdf.multi_cell(186, 6, s(assessment))

    # ---- Photos ------------------------------------------------------------
    photos = claim.get("photos") or []
    section(f"Photos du sinistre ({len(photos)})")
    if not photos:
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(0, 7, s("Aucune photo fournie."))
    else:
        for idx, photo in enumerate(photos, 1):
            decoded = _decode_photo_to_image_bytes(photo)
            fname = photo.get("filename", f"photo_{idx}") if isinstance(photo, dict) else f"photo_{idx}"

            # Estimate space; new page if not enough room for a photo block.
            if pdf.get_y() > 200:
                pdf.add_page()

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*DARK)
            pdf.cell(0, 6, s(f"Photo {idx}: {fname}"), ln=True)

            if isinstance(photo, dict) and photo.get("gps"):
                g = photo["gps"]
                try:
                    pdf.set_font("Helvetica", "", 8)
                    pdf.set_text_color(*GREY)
                    pdf.cell(0, 4, s(f"GPS: {g['latitude']:.5f}, {g['longitude']:.5f}"), ln=True)
                    pdf.set_text_color(*DARK)
                except Exception:
                    pass

            if decoded:
                img_bytes, ext = decoded
                try:
                    pdf.image(BytesIO(img_bytes), x=15, w=150)
                except Exception as e:
                    logger.warning("⚠️ pdf.image failed for photo %d: %r", idx, e)
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.set_text_color(150, 0, 0)
                    pdf.cell(0, 5, s("[Photo non affichable]"), ln=True)
                    pdf.set_text_color(*DARK)
            else:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(150, 0, 0)
                pdf.cell(0, 5, s("[Photo non affichable]"), ln=True)
                pdf.set_text_color(*DARK)
            pdf.ln(4)

    # ---- Conclusion --------------------------------------------------------
    section("Conclusion")
    pdf.set_font("Helvetica", "", 10)
    concl = (f"Le present rapport synthetise la declaration de sinistre {claim.get('claim_id', claim_id)} "
             f"de type \"{claim.get('damage_type', 'non determine')}\". "
             f"Le niveau de risque de fraude evalue est {risk_label} ({fs_disp}/100). {assessment} "
             "Cette synthese est une aide a la decision generee automatiquement par AssuranceIA(TM) "
             "et ne se substitue pas a l'appreciation d'un expert mandate.")
    pdf.set_x(12)
    pdf.multi_cell(186, 6, s(concl))
    pdf.ln(6)

    # ---- Optional QR code linking to claim details -------------------------
    qr_png = _build_qr_png(f"{APP_URL}/claim/{claim.get('claim_id', claim_id)}")
    if qr_png:
        try:
            if pdf.get_y() > 240:
                pdf.add_page()
            qy = pdf.get_y()
            pdf.image(BytesIO(qr_png), x=12, y=qy, w=28)
            pdf.set_xy(44, qy + 6)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*GREY)
            pdf.multi_cell(150, 5, s("Scannez pour acceder au detail du dossier en ligne."))
            pdf.set_text_color(*DARK)
            pdf.set_y(qy + 30)
        except Exception:
            pass

    # ---- Signature line ----------------------------------------------------
    if pdf.get_y() > 255:
        pdf.add_page()
    pdf.ln(4)
    pdf.set_draw_color(*GREY)
    pdf.set_line_width(0.2)
    y = pdf.get_y()
    pdf.line(12, y, 100, y)
    pdf.set_xy(12, y + 1)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 6, s(f"Declaration certifiee conforme le {format_french_datetime(datetime.now().isoformat())}"),
             ln=True)
    pdf.set_text_color(*DARK)

    out = pdf.output()
    return bytes(out)


def build_claim_pdf(claim: dict) -> bytes:
    """Build a simple one-page PDF report using fpdf2."""
    from fpdf import FPDF

    def s(text) -> str:
        # fpdf2 core fonts are latin-1; strip unsupported chars safely.
        return str(text).encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(102, 126, 234)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_xy(10, 8)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 10, s("AssuranceIA - Rapport de sinistre"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 19)
    pdf.cell(0, 6, s(f"Reference: {claim['claim_id']}"), ln=True)

    pdf.set_text_color(30, 30, 30)
    pdf.ln(20)

    def section(title: str):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(102, 126, 234)
        pdf.cell(0, 8, s(title), ln=True)
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "", 11)

    def field(label: str, value):
        y = pdf.get_y()
        pdf.set_xy(10, y)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(45, 7, s(label), border=0)
        pdf.set_font("Helvetica", "", 11)
        # Explicit width (A4 width 210 - left 10 - cell 45 - right 10 = 145).
        pdf.set_xy(55, y)
        pdf.multi_cell(145, 7, s(value if value not in (None, "") else "-"))

    # Claim details
    section("Informations sinistre")
    field("Email", claim.get("user_email"))
    field("Type declare", claim.get("damage_type"))
    field("Adresse", claim.get("address"))
    field("Description", claim.get("description"))
    field("Cree le", claim.get("created_at", "")[:19].replace("T", " "))
    loc = claim.get("location")
    if loc:
        field("Coordonnees", f"{loc['latitude']:.5f}, {loc['longitude']:.5f}")
    pdf.ln(3)

    # Location verification (anti-fraud GPS check)
    if len(claim.get("photos", [])) > 0:
        photo = claim["photos"][0]
        if photo.get("gps"):
            section("Verification geographique (EXIF GPS)")
            gps = photo.get("gps")
            field("GPS photo", f"{gps['latitude']:.5f}, {gps['longitude']:.5f}")
            analysis = claim.get("analysis")
            if analysis and analysis.get("location_verification"):
                loc_check = analysis["location_verification"]
                status = "✓ MATCH" if loc_check["matches"] else "✗ DESACCORD"
                field("Verification", status)
                if loc_check["distance_km"] is not None:
                    field("Distance", f"{loc_check['distance_km']} km de l'adresse declaree")
            pdf.ln(3)

    # Analysis
    analysis = claim.get("analysis")
    section("Detection Avancee Fraude")
    if not analysis:
        pdf.multi_cell(0, 7, s("Aucune analyse disponible. Lancez l'analyse depuis le tableau de bord."))
    else:
        field("Type detecte", analysis.get("detected_damage_type"))
        field("Gravite", analysis.get("damage_severity"))
        field("Localisation fuite", analysis.get("leak_location"))
        field("Degats visibles", analysis.get("visible_damage"))
        field("Score fraude", f"{analysis.get('fraud_score', '-')} / 100")
        field("Coherence", analysis.get("consistency_with_declaration"))
        indicators = analysis.get("fraud_indicators") or []
        if indicators:
            field("Indices fraude", "; ".join(str(i) for i in indicators))
        field("Recommandation", analysis.get("recommendation"))
        field("Confiance", analysis.get("confidence"))

    # Photos section
    photos = claim.get("photos", [])
    if photos:
        pdf.ln(6)
        section("Photos du sinistre")

        for idx, photo in enumerate(photos, 1):
            # Add page break if needed for multiple photos
            if pdf.get_y() > 250:
                pdf.add_page()
                pdf.ln(10)

            # Photo metadata
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, s(f"Photo {idx}: {photo.get('filename', 'unknown')}"), ln=True)

            # GPS info if available
            if photo.get("gps"):
                gps = photo["gps"]
                pdf.set_font("Helvetica", "", 9)
                pdf.cell(0, 4, s(f"GPS: {gps['latitude']:.5f}, {gps['longitude']:.5f}"), ln=True)

            # Try to display the photo (if base64 encoded image)
            try:
                import base64
                from io import BytesIO

                img_data = base64.b64decode(photo.get("data", ""))
                if img_data:
                    # Save temp image and add to PDF
                    img_file = BytesIO(img_data)
                    pdf.image(img_file, x=15, w=180, h=120)
                    pdf.ln(2)
            except Exception as e:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(150, 0, 0)
                pdf.cell(0, 4, s(f"[Photo non affichable]"), ln=True)
                pdf.set_text_color(30, 30, 30)

            pdf.ln(3)

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(
        0,
        5,
        s(
            "Document genere automatiquement par AssuranceIA. "
            "L'analyse IA est une aide a la decision et ne remplace pas l'expertise humaine."
        ),
    )

    out = pdf.output()
    return bytes(out)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
