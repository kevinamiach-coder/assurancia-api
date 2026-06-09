from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from fpdf import FPDF
from anthropic import Anthropic
import piexif
from PIL import Image
from io import BytesIO
import uuid
from datetime import datetime
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
    phone_gps_lon: float | None = None  # GPS téléphone (longitude) = ""


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

    # Increase fraud score based on location mismatch
    if gps_check["flag"] == "LOCATION_MISMATCH":
        distance = gps_check.get("distance_km", 0)
        if distance > 0.1:  # > 100 meters
            fraud_increase = min(40, int(distance * 2))  # Increase by 2 points per km, max 40
            analysis["fraud_score"] = min(100, analysis.get("fraud_score", 0) + fraud_increase)
            if "fraud_indicators" not in analysis or not isinstance(analysis["fraud_indicators"], list):
                analysis["fraud_indicators"] = []
            analysis["fraud_indicators"].append(f"GEOLOCALISATION: GPS a {distance}km de l'adresse (MISMATCH CRITIQUE)")

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

    location = geocode_address(claim.address)

    # Check GPS phone vs declared address (anti-fraud)
    phone_gps = None
    gps_verification = None
    if claim.phone_gps_lat is not None and claim.phone_gps_lon is not None:
        phone_gps = {"latitude": claim.phone_gps_lat, "longitude": claim.phone_gps_lon}
        gps_verification = check_gps_location_match(location, phone_gps)

    claim_data = {
        "claim_id": claim_id,
        "user_email": claim.user_email,
        "damage_type": claim.damage_type,
        "address": claim.address,
        "description": claim.description,
        "location": location,  # {latitude, longitude, display_name} from address geocoding
        "phone_gps": phone_gps,  # {latitude, longitude} from phone at claim creation
        "gps_verification": gps_verification,  # GPS phone vs declared address check
        "photos": [],
        "analysis": None,
        "status": "open",
        "created_at": datetime.now().isoformat(),
    }

    claims_db[claim_id] = claim_data

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

    base64_content = base64.b64encode(contents).decode("utf-8")
    photo_entry = {
        "filename": file.filename,
        "content_type": media_type,
        "data": base64_content,
        "gps": gps_data,  # {latitude, longitude} or None
    }
    claims_db[claim_id]["photos"].append(photo_entry)

    return {
        "status": "ok",
        "filename": file.filename,
        "photo_count": len(claims_db[claim_id]["photos"]),
        "gps_detected": gps_data is not None,
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
