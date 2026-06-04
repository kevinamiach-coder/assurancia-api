from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
from datetime import datetime
import os
import base64
import requests
import json

app = FastAPI(title="AssuranceIA API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anthropic API configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Pydantic models
class ClaimCreate(BaseModel):
    user_email: str
    damage_type: str
    address: str
    description: str = ""

# In-memory storage
claims_db = {}
photos_db = {}

@app.get("/")
async def read_root():
    """Serve index.html for root path"""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path, media_type="text/html")
    return {"message": "AssuranceIA API v1"}

@app.post("/claims")
def create_claim(claim: ClaimCreate):
    """Create a new insurance claim"""
    claim_id = f"CLM-{datetime.now().year}-{uuid.uuid4().hex[:6].upper()}"
    
    claim_data = {
        "claim_id": claim_id,
        "user_email": claim.user_email,
        "damage_type": claim.damage_type,
        "address": claim.address,
        "description": claim.description,
        "created_at": datetime.now().isoformat(),
        "photos": [],
        "analysis": None
    }
    
    claims_db[claim_id] = claim_data
    return claim_data

@app.get("/claims")
def get_claims():
    """Get all claims"""
    return list(claims_db.values())

@app.post("/claims/{claim_id}/photos")
async def upload_photo(claim_id: str, file: UploadFile = File(...)):
    """Upload a photo for a claim"""
    if claim_id not in claims_db:
        return {"error": "Claim not found"}
    
    photo_id = f"PHO-{uuid.uuid4().hex[:8].upper()}"
    photo_data = await file.read()
    photo_base64 = base64.b64encode(photo_data).decode('utf-8')
    
    photos_db[photo_id] = {
        "photo_id": photo_id,
        "claim_id": claim_id,
        "filename": file.filename,
        "size": len(photo_data),
        "content_base64": photo_base64,
        "media_type": file.content_type or "image/jpeg",
        "uploaded_at": datetime.now().isoformat()
    }
    
    claims_db[claim_id]["photos"].append(photo_id)
    
    return {
        "photo_id": photo_id,
        "claim_id": claim_id,
        "filename": file.filename,
        "uploaded_at": photos_db[photo_id]["uploaded_at"]
    }

@app.post("/claims/{claim_id}/analyze")
def analyze_claim(claim_id: str):
    """Analyze claim with AI using Claude Vision API"""
    if claim_id not in claims_db:
        return {"error": "Claim not found"}
    
    claim = claims_db[claim_id]
    
    # If no API key or no photos, return default analysis
    if not ANTHROPIC_API_KEY or not claim["photos"]:
        analysis = {
            "claim_id": claim_id,
            "damage_severity": "medium",
            "estimated_cost": 3000,
            "recommendation": "Manual inspection required",
            "analyzed_at": datetime.now().isoformat(),
            "note": "No Claude Vision API available or no photos"
        }
        claim["analysis"] = analysis
        return analysis
    
    try:
        # Build image content for Claude Vision
        image_content = []
        
        # Add text description
        image_content.append({
            "type": "text",
            "text": f"""Analyze this water damage insurance claim:
Type: {claim['damage_type']}
Address: {claim['address']}
Description: {claim['description']}

Provide: 1) Damage severity (low/medium/high), 2) Estimated cost in euros, 3) Recommendation"""
        })
        
        # Add first photo
        if claim["photos"] and claim["photos"][0] in photos_db:
            photo = photos_db[claim["photos"][0]]
            image_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": photo["media_type"],
                    "data": photo["content_base64"]
                }
            })
        
        # Call Claude Vision API directly
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": image_content
                }
            ]
        }
        
        response = requests.post(ANTHROPIC_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        claude_response = result["content"][0]["text"] if result.get("content") else "Analysis failed"
        
        analysis = {
            "claim_id": claim_id,
            "damage_severity": "high" if "high" in claude_response.lower() else "medium",
            "estimated_cost": 5000,
            "recommendation": "Approve for investigation",
            "claude_analysis": claude_response[:300],
            "analyzed_at": datetime.now().isoformat()
        }
        
        claim["analysis"] = analysis
        return analysis
        
    except Exception as e:
        return {
            "claim_id": claim_id,
            "error": f"Analysis failed: {str(e)}",
            "analyzed_at": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
