from fastapi import FastAPI, File, UploadFile, HTTPException
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

app = FastAPI(title="AssuranceIA API", version="2.0")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ANTHROPIC_API_KEY = (os.getenv("ANTHROPIC_API_KEY") or "").strip().strip('"').strip("'")
CLAUDE_MODEL = "claude-sonnet-4-6"  # Vision model that works

# Initialize Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# In-memory storage. NOTE: Render free tier sleeps after inactivity and wipes
# this dict on restart. Acceptable for a demo; migrate to a real DB for prod.
claims_db: dict = {}
token_to_claim: dict = {}  # Mapping: unique_token -> claim_id
declaration_links: dict = {}  # Declaration templates: token -> {insurer_email, client_email, created_at}

# Anthropic Vision API only accepts these media types.
SUPPORTED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_PHOTO_BYTES = 5 * 1024 * 1024  # 5 MB per photo (keeps memory safe on free tier)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    return {
        "status": "ok",
        "filename": file.filename,
        "photo_count": len(claims_db[claim_id]["photos"]),
        "gps_detected": gps_data is not None,
        "exif_datetime": exif_datetime.get("timestamp") if exif_datetime else None,
        "is_duplicate": duplicate_check["is_duplicate"],
        "duplicate_warning": duplicate_check["fraud_indicator"],
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

    declaration_links[token] = {
        "insurer_email": request.insurer_email,
        "client_email": request.client_email,
        "created_at": datetime.now().isoformat(),
        "status": "pending"  # Not yet filled by client
    }

    base_url = os.getenv("APP_URL", "https://assurancia-api-2.onrender.com")
    declaration_url = f"{base_url}/declare/{token}"

    return {
        "token": token,
        "declaration_url": declaration_url,
        "insurer_email": request.insurer_email,
        "message": f"Envoyez ce lien au client: {declaration_url}",
        "qr_code_hint": f"QR code to generate: {declaration_url}"
    }


# ========== TOKEN-BASED ROUTES (for sharing with clients/insurers) ==========

@app.get("/declare/{token}")
def get_declaration_form(token: str):
    """
    Client access: Simple declaration form.
    If token is a pending declaration link, show form.
    If token is an existing claim, show claim details.
    """
    # Check if it's a declaration link (pending)
    if token in declaration_links:
        link_data = declaration_links[token]
        client_email = link_data.get("client_email", "")

        html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Déclaration de Sinistre - AssuranceIA™</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
                .container {{ max-width: 600px; margin: 0 auto; }}
                .card {{ background: white; border-radius: 12px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .header h1 {{ color: #667eea; font-size: 2em; margin-bottom: 8px; }}
                .header p {{ color: #666; }}
                .form-group {{ margin-bottom: 20px; }}
                label {{ display: block; margin-bottom: 8px; font-weight: 600; color: #333; }}
                input, select, textarea {{ width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 6px; font-family: inherit; font-size: inherit; }}
                textarea {{ resize: vertical; min-height: 100px; }}
                input:focus, select:focus, textarea:focus {{ outline: none; border-color: #667eea; box-shadow: 0 0 0 3px rgba(102,126,234,0.1); }}
                button {{ width: 100%; padding: 14px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 6px; font-size: 16px; font-weight: 600; cursor: pointer; transition: transform 0.2s; }}
                button:hover {{ transform: translateY(-2px); }}
                .required {{ color: red; }}
                .status {{ text-align: center; margin-bottom: 20px; padding: 12px; background: #e8f4f8; border-radius: 6px; color: #0066cc; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="card">
                    <div class="header">
                        <h1>🔍 Déclaration de Sinistre</h1>
                        <p>AssuranceIA™ - Validation automatisée</p>
                    </div>

                    <div class="status">
                        ✅ Lien de déclaration valide - Remplissez le formulaire ci-dessous
                    </div>

                    <form id="declarationForm">
                        <div class="form-group">
                            <label for="email">Email <span class="required">*</span></label>
                            <input type="email" id="email" name="email" value="{client_email}" required>
                        </div>

                        <div class="form-group">
                            <label for="damageType">Type de dégât <span class="required">*</span></label>
                            <select id="damageType" name="damageType" required>
                                <option value="">-- Sélectionner --</option>
                                <optgroup label="💧 Dégâts des Eaux">
                                    <option value="fuite">Fuite d'eau</option>
                                    <option value="inondation">Inondation</option>
                                    <option value="rupture_canalisation">Rupture de canalisation</option>
                                    <option value="infiltration_toiture">Infiltration toiture</option>
                                </optgroup>
                                <optgroup label="🚗 Sinistres Automobile">
                                    <option value="accident_circulation">Accident de circulation</option>
                                    <option value="vandalisme_auto">Vandalisme/Rayures</option>
                                </optgroup>
                                <optgroup label="🔓 Cambriolage">
                                    <option value="effraction">Effraction/Intrusion</option>
                                    <option value="vol">Vol/Cambriolage</option>
                                </optgroup>
                            </select>
                        </div>

                        <div class="form-group">
                            <label for="address">Adresse du sinistre <span class="required">*</span></label>
                            <input type="text" id="address" name="address" placeholder="12 Rue de Rivoli, 75001 Paris" required>
                        </div>

                        <div class="form-group">
                            <label for="description">Description des dégâts <span class="required">*</span></label>
                            <textarea id="description" name="description" placeholder="Décrivez les dégâts constatés..." required></textarea>
                        </div>

                        <div class="form-group">
                            <label for="photos">Photos (optionnel)</label>
                            <input type="file" id="photos" name="photos" multiple accept="image/*">
                        </div>

                        <button type="submit">📤 Soumettre la déclaration</button>
                    </form>

                    <div id="status" style="margin-top: 20px; text-align: center; display: none;"></div>
                </div>
            </div>

            <script>
                const token = '{token}';
                const apiUrl = window.location.origin;

                document.getElementById('declarationForm').addEventListener('submit', async (e) => {
                    e.preventDefault();

                    const formData = new FormData();
                    formData.append('user_email', document.getElementById('email').value);
                    formData.append('damage_type', document.getElementById('damageType').value);
                    formData.append('address', document.getElementById('address').value);
                    formData.append('description', document.getElementById('description').value);
                    formData.append('token', token);

                    try {{
                        const response = await fetch(`${{apiUrl}}/declare/${{token}}/submit`, {{
                            method: 'POST',
                            body: formData
                        }});

                        const data = await response.json();

                        if (response.ok) {{
                            document.getElementById('status').style.display = 'block';
                            document.getElementById('status').innerHTML = `
                                <div style="color: green; padding: 15px; background: #e8f5e9; border-radius: 6px;">
                                    <h3>✅ Sinistre déclaré avec succès!</h3>
                                    <p>Référence: ${{data.claim_id}}</p>
                                    <p>Vous recevrez le rapport d'analyse par email.</p>
                                </div>
                            `;
                            document.getElementById('declarationForm').style.display = 'none';
                        }} else {{
                            alert('Erreur: ' + data.detail);
                        }}
                    }} catch (err) {{
                        alert('Erreur réseau: ' + err.message);
                    }}
                }});
            </script>
        </body>
        </html>
        """
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
async def submit_declaration(token: str, user_email: str = "", damage_type: str = "", address: str = "", description: str = ""):
    """
    Client submits declaration via token link.
    Creates claim + stores insurer email for auto PDF sending.
    """
    if token not in declaration_links:
        raise HTTPException(status_code=404, detail="Lien invalide ou expiré")

    link_data = declaration_links[token]
    insurer_email = link_data["insurer_email"]

    # Create claim
    claim_id = f"CLM-{datetime.now().year}-{uuid.uuid4().hex[:6].upper()}"
    unique_token = secrets.token_urlsafe(32)

    location = geocode_address(address)
    fraud_history = check_fraud_history(user_email, address)

    claim_data = {
        "claim_id": claim_id,
        "unique_token": unique_token,
        "user_email": user_email,
        "damage_type": damage_type,
        "address": address,
        "description": description,
        "location": location,
        "phone_gps": None,
        "gps_verification": None,
        "fraud_history": fraud_history,
        "insurer_email": insurer_email,  # Store for auto PDF sending
        "photos": [],
        "analysis": None,
        "status": "open",
        "created_at": datetime.now().isoformat(),
    }

    claims_db[claim_id] = claim_data
    token_to_claim[unique_token] = claim_id
    declaration_links[token]["status"] = "completed"
    declaration_links[token]["claim_id"] = claim_id

    return {
        "claim_id": claim_id,
        "unique_token": unique_token,
        "message": "✅ Sinistre créé. Uploadez les photos pour lancer l'analyse automatique.",
        "next_step": f"Uploadez photos via POST /claims/{claim_id}/photos",
        "insurer_will_receive": f"PDF sera envoyé à {insurer_email} après analyse"
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
