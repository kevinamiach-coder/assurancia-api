from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

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
    description: str = ""


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


def call_claude_vision(photo: dict, claim: dict) -> dict:
    """Call Claude Vision and return a structured analysis dict."""
    media_type = normalize_media_type(photo.get("content_type"))
    if media_type is None:
        return {
            "error": "unsupported_image",
            "damage_severity": "unknown",
            "recommendation": "Format d'image non supporte (utilisez JPEG, PNG, GIF ou WEBP).",
        }

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    prompt = f"""Vous etes un expert en assurance specialise dans les degats des eaux.
Analysez cette photo de sinistre.

Contexte declare par l'assure:
- Type de degat: {claim['damage_type']}
- Description: {claim['description']}
- Adresse: {claim['address']}

Repondez UNIQUEMENT avec un objet JSON valide (aucun texte avant ou apres) avec EXACTEMENT ces champs:
{{
  "detected_damage_type": "fuite | inondation | rupture_canalisation | infiltration_toiture | moisissure | autre",
  "damage_severity": "low | medium | high | critical",
  "estimated_cost_eur": <nombre entier en euros>,
  "leak_location": "description courte du point d'impact ou de la source visible",
  "visible_damage": "description detaillee des degats visibles sur la photo",
  "fraud_score": <entier 0-100, probabilite de fraude/incoherence>,
  "fraud_indicators": ["liste courte d'indices suspects, ou liste vide"],
  "consistency_with_declaration": "coherent | partiellement_coherent | incoherent",
  "recommendation": "recommandation d'action pour l'expert",
  "confidence": "high | medium | low"
}}"""

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 2048,
        "messages": [
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
    }

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=120,
        )
    except requests.RequestException as e:
        return {
            "error": "network_error",
            "damage_severity": "unknown",
            "recommendation": f"Connexion a Claude impossible: {e}",
        }

    if response.status_code != 200:
        return {
            "error": f"api_error_{response.status_code}",
            "damage_severity": "unknown",
            "recommendation": "Analyse echouee - veuillez reessayer.",
            "details": response.text[:300],
        }

    data = response.json()
    response_text = data["content"][0]["text"]
    analysis = extract_json(response_text)

    if analysis is None:
        # Could not parse: return the raw text so nothing is lost.
        return {
            "damage_severity": "medium",
            "estimated_cost_eur": 3000,
            "visible_damage": response_text[:1000],
            "recommendation": "Inspection manuelle requise (reponse IA non structuree).",
            "fraud_score": 0,
            "confidence": "low",
        }
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
    return {
        "status": "ok",
        "claims_count": len(claims_db),
        "claude_configured": bool(ANTHROPIC_API_KEY),
    }


@app.post("/claims")
def create_claim(claim: ClaimCreate):
    """Create a new claim, geocode its address, and store it in memory."""
    claim_id = f"CLM-{datetime.now().year}-{uuid.uuid4().hex[:6].upper()}"

    location = geocode_address(claim.address)

    claim_data = {
        "claim_id": claim_id,
        "user_email": claim.user_email,
        "damage_type": claim.damage_type,
        "address": claim.address,
        "description": claim.description,
        "location": location,  # {latitude, longitude, display_name} or None
        "photos": [],
        "analysis": None,
        "status": "open",
        "created_at": datetime.now().isoformat(),
    }

    claims_db[claim_id] = claim_data
    print(f"Claim created: {claim_id} (geocoded={location is not None})")
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

    base64_content = base64.b64encode(contents).decode("utf-8")
    claims_db[claim_id]["photos"].append(
        {
            "filename": file.filename,
            "content_type": media_type,
            "data": base64_content,
        }
    )

    return {
        "status": "ok",
        "filename": file.filename,
        "photo_count": len(claims_db[claim_id]["photos"]),
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
