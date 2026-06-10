from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from fpdf import FPDF
from anthropic import Anthropic
import piexif
from PIL import Image
from io import BytesIO
import hashlib
import uuid
from datetime import datetime
import secrets
import os
import base64
import json
import re
import requests
from pymongo import MongoClient

app = FastAPI(title="AssuranceIA API", version="2.0")

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
MONGODB_URI = os.getenv("MONGODB_URI") or "mongodb+srv://artisanpatrimoinefrancais_db_user:REMOVED@cluster0.ac2vlvx.mongodb.net/?appName=Cluster0"

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
def get_claims():
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
def get_claim(claim_id: str):
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
def analyze_claim(claim_id: str):
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
def create_declaration_link(request: DeclarationLinkRequest):
    """
    Assurance/Courtier creates unique declaration link for client.
    Returns URL to send to client.
    """
    token = secrets.token_urlsafe(32)

    link_data = {
        "token": token,
        "insurer_email": request.insurer_email,
        "client_email": request.client_email,
        "created_at": datetime.now().isoformat(),
        "status": "pending"  # Not yet filled by client
    }

    declaration_links[token] = link_data

    # Save to MongoDB so token survives Render redeploys
    try:
        declaration_links_collection.insert_one(link_data)
    except:
        pass

    base_url = os.getenv("APP_URL", "https://assurancia-api-2.onrender.com")
    declaration_url = f"{base_url}/declare/{token}"

    return {
        "token": token,
        "declaration_url": declaration_url,
        "insurer_email": request.insurer_email,
        "message": f"Envoyez ce lien au client: {declaration_url}",
        "qr_code_hint": f"QR code to generate: {declaration_url}"
    }


# ========== DASHBOARD ROUTE ==========

@app.get("/dashboard")
async def dashboard():
    """Dashboard to view all claims from MongoDB (falls back to in-memory)."""
    try:
        claims_list = list(claims_collection.find({}, {"_id": 0}).sort("created_at", -1))
    except Exception as e:
        logger.warning("⚠️  Dashboard could not read from MongoDB: %r — using in-memory fallback.", e)
        claims_list = []

    # Fallback / merge: if Mongo returned nothing but we have in-memory claims,
    # show those so the dashboard is never wrongly empty.
    if not claims_list and claims_db:
        claims_list = sorted(
            [{k: v for k, v in c.items() if k != "_id"} for c in claims_db.values()],
            key=lambda c: c.get("created_at", ""),
            reverse=True,
        )

    html = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AssuranceIA™ Dashboard</title>
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
                max-width: 1400px;
                margin: 0 auto;
            }}
            header {{
                margin-bottom: 40px;
            }}
            h1 {{
                font-size: 36px;
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
            }}
            p {{
                color: #cbd5e1;
                font-size: 16px;
            }}
            .table-container {{
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.9) 100%);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 12px;
                overflow: hidden;
                backdrop-filter: blur(10px);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th {{
                background: rgba(59, 130, 246, 0.1);
                padding: 16px;
                text-align: left;
                font-weight: 600;
                border-bottom: 1px solid rgba(148, 163, 184, 0.2);
                color: #60a5fa;
            }}
            td {{
                padding: 16px;
                border-bottom: 1px solid rgba(148, 163, 184, 0.1);
            }}
            tr:hover {{
                background: rgba(59, 130, 246, 0.05);
            }}
            .reference {{
                color: #60a5fa;
                font-weight: 600;
            }}
            .status {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }}
            .status.low {{
                background: rgba(34, 197, 94, 0.2);
                color: #86efac;
            }}
            .status.medium {{
                background: rgba(248, 113, 113, 0.2);
                color: #fca5a5;
            }}
            .status.high {{
                background: rgba(239, 68, 68, 0.2);
                color: #fca5a5;
            }}
            .empty {{
                text-align: center;
                padding: 60px 20px;
                color: #94a3b8;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🔐 AssuranceIA™ Dashboard</h1>
                <p>Tous les sinistres déclarés et analysés</p>
            </header>

            <div class="table-container">
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
    """

    if claims_list:
        for claim in claims_list:
            ref = claim.get("claim_id", "N/A")
            email = claim.get("user_email", "N/A")
            firstname = claim.get("firstname", "N/A")
            lastname = claim.get("lastname", "N/A")
            phone = claim.get("phone", "N/A")
            damage = claim.get("damage_type", "N/A")
            address = claim.get("address", "N/A")
            fraud_score = claim.get("fraud_score", 0)
            created = claim.get("created_at", "N/A")[:10]

            # Fraud status badge
            if fraud_score < 20:
                status = '<span class="status low">✓ Fiable</span>'
            elif fraud_score < 60:
                status = '<span class="status medium">⚠ Attention</span>'
            else:
                status = '<span class="status high">🚨 Suspect</span>'

            html += f"""
                        <tr>
                            <td class="reference"><a href="/claim/{ref}" style="color: #60a5fa; text-decoration: none; cursor: pointer; font-weight: 600;">{ref}</a></td>
                            <td>{email}</td>
                            <td>{firstname}</td>
                            <td>{lastname}</td>
                            <td>{phone}</td>
                            <td>{damage}</td>
                            <td>{address[:40]}...</td>
                            <td>{status} ({fraud_score})</td>
                            <td>{created}</td>
                        </tr>
            """
    else:
        html += """
                        <tr>
                            <td colspan="6" class="empty">
                                <p>Aucun sinistre déclaré pour le moment</p>
                            </td>
                        </tr>
        """

    html += """
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """

    return Response(content=html, media_type="text/html")


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
                            <label for="photos">📸 Photos des dégâts (optionnel)</label>
                            <input type="file" id="photos" name="photos" multiple accept="image/*">
                            <p class="info-text">Uploadez plusieurs photos pour une analyse meilleure</p>
                        </div>

                        <!-- Hidden GPS fields -->
                        <input type="hidden" id="gpsLat" name="gpsLat">
                        <input type="hidden" id="gpsLon" name="gpsLon">

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
                formData.append("token", token);

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
                        statusDiv.innerHTML = "<h3>Sinistre declare!</h3><p>Reference: " + data.claim_id + "</p>";
                        document.getElementById("declarationForm").style.display = "none";
                    } else {
                        alert("Erreur");
                    }
                }).catch(function(err) {
                    alert("Erreur reseau");
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
def view_claim_details(claim_id: str):
    """View complete claim details with photos, analysis, GPS map, and fraud score."""
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
    if fraud_score < 20:
        fraud_status = '<span class="badge low">✓ Fiable</span>'
    elif fraud_score < 60:
        fraud_status = '<span class="badge medium">⚠ Attention</span>'
    else:
        fraud_status = '<span class="badge high">🚨 Suspect</span>'
    fraud_score_display = int(fraud_score) if fraud_score == int(fraud_score) else fraud_score

    # --- Date (defensive: created_at may be None / missing / short) ---
    # Stored in ISO format; displayed as "10 juin 2026 à 13:45:32 (GMT+3)".
    created_at = claim.get("created_at")
    date_display = format_french_datetime(created_at)

    # --- Photos HTML ---
    photos_html = ""
    photos = claim.get("photos") or []
    if photos:
        photos_html = '<div class="section"><h3>📸 Photos</h3><div class="photos-grid">'
        for i, photo_data in enumerate(photos):
            if isinstance(photo_data, str) and photo_data.startswith("data:"):
                photos_html += f'<img src="{photo_data}" alt="Photo {i+1}" class="claim-photo">'
        photos_html += '</div></div>'

    # --- Analysis HTML (defensive: analysis may be a dict, a string, or None) ---
    analysis_html = ""
    analysis = claim.get("analysis")
    if analysis:
        if isinstance(analysis, dict):
            a_summary = analysis.get("summary") or "Pas de résumé disponible."
            a_type = analysis.get("detected_damage_type", "Non déterminé")
            a_severity = analysis.get("damage_severity", "Non déterminée")
            a_reco = analysis.get("recommendation", "Aucune recommandation.")
            a_cost = analysis.get("estimated_cost_eur", analysis.get("estimated_cost"))
            a_analyzed = format_french_datetime(analysis.get("analyzed_at")) if analysis.get("analyzed_at") else None

            # Severity badge styling
            sev_map = {
                "low": ("badge low", "Faible"),
                "medium": ("badge medium", "Modérée"),
                "high": ("badge high", "Élevée"),
                "critical": ("badge high", "Critique"),
            }
            sev_class, sev_label = sev_map.get(str(a_severity).lower(), ("badge medium", str(a_severity)))

            cost_row = ""
            if a_cost not in (None, "", 0):
                cost_row = f"""
                <div class="info-item">
                    <div class="info-label">Coût estimé</div>
                    <div class="info-value">{a_cost} €</div>
                </div>"""

            analyzed_row = ""
            if a_analyzed:
                analyzed_row = f'<p style="margin-top:15px;color:#94a3b8;font-size:12px;">Analyse effectuée le {a_analyzed}</p>'

            analysis_html = f"""
            <div class="section">
                <h3>🤖 Analyse Claude Vision</h3>
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
                        </div>{cost_row}
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
                <h3>🤖 Analyse Claude Vision</h3>
                <div class="analysis-content">{str(analysis)}</div>
            </div>
            """
    else:
        analysis_html = """
        <div class="section">
            <h3>🤖 Analyse Claude Vision</h3>
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
    if fraud_score < 20:
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
    elif fraud_score < 60:
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
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🔐 Détail du Sinistre</h1>
                <p style="color: #cbd5e1; font-size: 16px;">Référence: <strong>{claim_id}</strong></p>
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
                    <div class="info-item">
                        <div class="info-label">Score Fraude</div>
                        <div class="info-value">{fraud_status} ({fraud_score_display})</div>
                    </div>
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

        <script>
        var CLAIM_ID = '{claim_id}';

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
async def submit_declaration(
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
):
    """
    Client submits declaration via token link.
    Creates claim + stores insurer email + verifies GPS location.

    NOTE: parameters use Form(...) because the frontend submits multipart/form-data
    (FormData). Plain str params would be read as query params and arrive empty.
    """
    # Log exactly what arrived so we can never again wonder if data was received.
    logger.info(
        "📥 submit_declaration token=%s | email=%r firstname=%r lastname=%r "
        "phone=%r type=%r address=%r desc_len=%d gps=(%s,%s)",
        token, user_email, firstname, lastname, phone, damage_type,
        address, len(description or ""), phone_gps_lat, phone_gps_lon,
    )

    if token not in declaration_links:
        raise HTTPException(status_code=404, detail="Lien invalide ou expiré")

    link_data = declaration_links[token]
    insurer_email = link_data["insurer_email"]

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

    # Save to MongoDB. insert_one() mutates the dict by adding "_id" (a non-JSON-
    # serializable ObjectId), so insert a COPY to keep claims_db clean.
    mongo_saved = False
    mongo_error = None
    try:
        result = claims_collection.insert_one(dict(claim_data))
        mongo_saved = True
        logger.info("✅ MongoDB insert OK: claim_id=%s _id=%s", claim_id, result.inserted_id)
    except Exception as e:
        mongo_error = str(e)
        logger.error("❌ MongoDB insert FAILED for claim_id=%s: %r", claim_id, e)
        logger.error("   → Claim is still available in-memory (fallback). Fix Mongo to persist.")

    return {
        "mongo_saved": mongo_saved,
        "mongo_error": mongo_error,
        "claim_id": claim_id,
        "unique_token": unique_token,
        "gps_verification": gps_verification,
        "message": "✅ Sinistre créé avec GPS vérifié. Uploadez les photos pour lancer l'analyse automatique.",
        "next_step": f"Uploadez photos via POST /claims/{claim_id}/photos",
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


# ----------------------------------------------------------------------------
# PDF generation (pure-Python, no external binary needed)
# ----------------------------------------------------------------------------
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
    section("Analyse IA (Claude Vision)")
    if not analysis:
        pdf.multi_cell(0, 7, s("Aucune analyse disponible. Lancez l'analyse depuis le tableau de bord."))
    else:
        field("Type detecte", analysis.get("detected_damage_type"))
        field("Gravite", analysis.get("damage_severity"))
        cost = analysis.get("estimated_cost_eur", analysis.get("estimated_cost"))
        field("Cout estime", f"{cost} EUR" if cost not in (None, "") else "-")
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
