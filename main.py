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
from supabase import create_client, Client

app = FastAPI(title="AssuranceIA API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase initialization
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://fpddknsrhkadtethohrf.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_uoos9HB_meSo8CrcEweF-g_7o2Vg1_9")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Anthropic API configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Pydantic models
class ClaimCreate(BaseModel):
    user_email: str
    damage_type: str
    address: str
    description: str = ""

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
        "claim_number": claim_id,
        "user_email": claim.user_email,
        "damage_type": claim.damage_type,
        "address": claim.address,
        "description": claim.description,
        "created_at": datetime.now().isoformat(),
    }
    
    # Insert into Supabase
    response = supabase.table("claims").insert(claim_data).execute()
    
    if response.data:
        return {
            "claim_id": claim_id,
            "user_email": claim.user_email,
            "damage_type": claim.damage_type,
            "address": claim.address,
            "description": claim.description,
            "created_at": claim_data["created_at"],
            "photos": [],
            "analysis": None
        }
    return {"error": "Failed to create claim"}

@app.get("/claims")
def get_claims():
    """Get all claims"""
    try:
        response = supabase.table("claims").select("*").order("created_at", desc=True).execute()
        claims = []
        for claim in response.data:
            # Get photos for this claim
            photos_response = supabase.table("claim_photos").select("*").eq("claim_id", claim["id"]).execute()
            
            claims.append({
                "claim_id": claim.get("claim_number", ""),
                "user_email": claim.get("user_email", ""),
                "damage_type": claim.get("damage_type", ""),
                "address": claim.get("address", ""),
                "description": claim.get("description", ""),
                "created_at": claim.get("created_at", ""),
                "photos": [p.get("id") for p in photos_response.data] if photos_response.data else [],
                "analysis": None
            })
        return claims
    except Exception as e:
        return {"error": str(e)}

@app.post("/claims/{claim_id}/photos")
async def upload_photo(claim_id: str, file: UploadFile = File(...)):
    """Upload a photo for a claim"""
    try:
        photo_id = f"PHO-{uuid.uuid4().hex[:8].upper()}"
        photo_data = await file.read()
        photo_base64 = base64.b64encode(photo_data).decode('utf-8')
        
        # First find the claim by claim_number
        claims_response = supabase.table("claims").select("id").eq("claim_number", claim_id).execute()
        if not claims_response.data:
            return {"error": "Claim not found"}
        
        claim_db_id = claims_response.data[0]["id"]
        
        # Insert photo
        photo_data_db = {
            "claim_id": claim_db_id,
            "photo_url": photo_id,
            "filename": file.filename,
            "content_base64": photo_base64,
            "media_type": file.content_type or "image/jpeg",
        }
        
        response = supabase.table("claim_photos").insert(photo_data_db).execute()
        
        if response.data:
            return {
                "photo_id": photo_id,
                "claim_id": claim_id,
                "filename": file.filename,
                "uploaded_at": datetime.now().isoformat()
            }
        return {"error": "Failed to upload photo"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/claims/{claim_id}/analyze")
def analyze_claim(claim_id: str):
    """Analyze claim with AI"""
    try:
        # Get claim from Supabase
        claims_response = supabase.table("claims").select("*").eq("claim_number", claim_id).execute()
        if not claims_response.data:
            return {"error": "Claim not found"}
        
        claim = claims_response.data[0]
        
        # Default analysis
        analysis = {
            "claim_id": claim_id,
            "damage_severity": "medium",
            "estimated_cost": 3000,
            "recommendation": "Manual inspection required",
            "analyzed_at": datetime.now().isoformat(),
            "note": "Default analysis (Claude Vision API not yet integrated)"
        }
        
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
