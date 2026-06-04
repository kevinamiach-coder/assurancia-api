from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
from datetime import datetime
import os
import base64
import json
import requests

app = FastAPI(title="AssuranceIA API")

# Get API key from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClaimCreate(BaseModel):
    user_email: str
    damage_type: str
    address: str
    description: str = ""

claims_db = {}

@app.get("/")
async def read_root():
    """Serve index.html"""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path, media_type="text/html")
    return {"message": "AssuranceIA API v1"}

@app.post("/claims")
def create_claim(claim: ClaimCreate):
    """Create claim"""
    claim_id = f"CLM-{datetime.now().year}-{uuid.uuid4().hex[:6].upper()}"
    claims_db[claim_id] = {
        "claim_id": claim_id,
        "user_email": claim.user_email,
        "damage_type": claim.damage_type,
        "address": claim.address,
        "description": claim.description,
        "created_at": datetime.now().isoformat(),
        "photos": [],
        "analysis": None
    }
    return claims_db[claim_id]

@app.get("/claims")
def get_claims():
    """Get all claims"""
    return list(claims_db.values())

@app.post("/claims/{claim_id}/photos")
async def upload_photo(claim_id: str, file: UploadFile = File(...)):
    """Upload photo - store as base64"""
    if claim_id not in claims_db:
        return {"error": "Claim not found"}

    try:
        # Read file content
        contents = await file.read()
        # Convert to base64
        base64_content = base64.b64encode(contents).decode('utf-8')

        # Store photo with metadata
        claims_db[claim_id]["photos"].append({
            "filename": file.filename,
            "content_type": file.content_type,
            "data": base64_content
        })

        return {
            "status": "ok",
            "filename": file.filename,
            "message": "Photo uploadée avec succès"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/claims/{claim_id}/analyze")
def analyze_claim(claim_id: str):
    """Analyze claim with Claude Vision API"""
    if claim_id not in claims_db:
        return {"error": "Claim not found"}

    claim = claims_db[claim_id]

    # If no photos, return message
    if not claim["photos"]:
        analysis = {
            "damage_severity": "unknown",
            "estimated_cost": 0,
            "recommendation": "Veuillez uploader des photos pour l'analyse"
        }
    else:
        # Use Claude Vision API via HTTP
        try:
            if not ANTHROPIC_API_KEY:
                return {"error": "API key not configured"}

            photo = claim["photos"][0]

            # Prepare the request to Claude API
            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": photo["content_type"],
                                    "data": photo["data"],
                                },
                            },
                            {
                                "type": "text",
                                "text": f"""Vous êtes un expert en assurance dégâts d'eau. Analysez cette photo de sinistre eau.

Contexte du sinistre:
- Type: {claim['damage_type']}
- Description: {claim['description']}
- Adresse: {claim['address']}

Donnez une analyse JSON avec:
1. damage_severity: "low" / "medium" / "high" / "critical"
2. estimated_cost: estimation en euros
3. visible_damage: description des dégâts visibles
4. recommendation: recommandation d'action
5. confidence: "high" / "medium" / "low"

Répondez UNIQUEMENT en JSON valide."""
                            }
                        ],
                    }
                ]
            }

            # Call Claude API
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                response_text = data["content"][0]["text"]

                # Try to extract JSON from response
                try:
                    analysis = json.loads(response_text)
                except:
                    # If not valid JSON, create structured response
                    analysis = {
                        "damage_severity": "medium",
                        "estimated_cost": 3000,
                        "visible_damage": response_text,
                        "recommendation": "Inspection manuelle requise",
                        "confidence": "medium"
                    }
            else:
                analysis = {
                    "error": f"API Error {response.status_code}",
                    "damage_severity": "unknown",
                    "recommendation": "Analyse échouée - veuillez réessayer",
                    "details": response.text[:200]
                }
        except Exception as e:
            analysis = {
                "error": str(e),
                "damage_severity": "unknown",
                "recommendation": "Analyse échouée - veuillez réessayer"
            }

    claim["analysis"] = analysis
    return analysis

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
