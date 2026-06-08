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

# Environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# In-memory storage
claims_db: dict = {}

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

@app.get("/")
async def read_root():
    """Serve index.html or fallback"""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path, media_type="text/html")
    return {"message": "AssuranceIA API v1"}

@app.post("/claims")
def create_claim(claim: ClaimCreate):
    """Create a new claim and store it in memory"""
    try:
        claim_id = f"CLM-{datetime.now().year}-{uuid.uuid4().hex[:6].upper()}"

        claim_data = {
            "claim_id": claim_id,
            "user_email": claim.user_email,
            "damage_type": claim.damage_type,
            "address": claim.address,
            "description": claim.description,
            "photos": [],
            "analysis": None,
            "created_at": datetime.now().isoformat()
        }

        claims_db[claim_id] = claim_data
        print(f"Claim created: {claim_id}")

        return claim_data
    except Exception as e:
        return {"error": str(e)}

@app.get("/claims")
def get_claims():
    """Return all claims from memory"""
    try:
        return list(claims_db.values())
    except Exception as e:
        return {"error": str(e)}

@app.post("/claims/{claim_id}/photos")
async def upload_photo(claim_id: str, file: UploadFile = File(...)):
    """Upload a photo as base64 and attach it to the claim"""
    try:
        if claim_id not in claims_db:
            return {"error": "Claim not found"}

        # Read and encode file as base64
        contents = await file.read()
        base64_content = base64.b64encode(contents).decode("utf-8")

        claims_db[claim_id]["photos"].append({
            "filename": file.filename,
            "content_type": file.content_type,
            "data": base64_content
        })

        return {
            "status": "ok",
            "filename": file.filename,
            "message": "Photo uploadee avec succes"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/claims/{claim_id}/analyze")
def analyze_claim(claim_id: str):
    """Analyze a claim using Claude Vision API via HTTP"""
    try:
        if claim_id not in claims_db:
            return {"error": "Claim not found"}

        claim = claims_db[claim_id]

        # No photos: return default message
        if not claim["photos"]:
            analysis = {
                "damage_severity": "unknown",
                "estimated_cost": 0,
                "recommendation": "Veuillez uploader des photos pour l'analyse"
            }
        else:
            if not ANTHROPIC_API_KEY:
                return {"error": "ANTHROPIC_API_KEY not configured"}

            photo = claim["photos"][0]

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
                                    "data": photo["data"]
                                }
                            },
                            {
                                "type": "text",
                                "text": f"""Vous etes un expert en assurance degats d'eau. Analysez cette photo de sinistre eau.

Contexte du sinistre:
- Type: {claim['damage_type']}
- Description: {claim['description']}
- Adresse: {claim['address']}

Donnez une analyse JSON avec:
1. damage_severity: "low" / "medium" / "high" / "critical"
2. estimated_cost: estimation en euros
3. visible_damage: description des degats visibles
4. recommendation: recommandation d'action
5. confidence: "high" / "medium" / "low"

Repondez UNIQUEMENT en JSON valide."""
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                response_text = data["content"][0]["text"]

                # Extract JSON block if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()

                try:
                    analysis = json.loads(response_text)
                except Exception:
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
                    "recommendation": "Analyse echouee - veuillez reessayer",
                    "details": response.text[:200]
                }

        # Save analysis back to memory
        claims_db[claim_id]["analysis"] = analysis

        return analysis
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
