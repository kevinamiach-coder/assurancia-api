from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
from datetime import datetime
import os

app = FastAPI(title="AssuranceIA API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    photos_db[photo_id] = {
        "photo_id": photo_id,
        "claim_id": claim_id,
        "filename": file.filename,
        "size": len(photo_data),
        "uploaded_at": datetime.now().isoformat()
    }
    
    claims_db[claim_id]["photos"].append(photo_id)
    
    return photos_db[photo_id]

@app.post("/claims/{claim_id}/analyze")
def analyze_claim(claim_id: str):
    """Analyze claim with AI"""
    if claim_id not in claims_db:
        return {"error": "Claim not found"}
    
    claim = claims_db[claim_id]
    
    # Dummy analysis (replace with actual Claude Vision API call)
    analysis = {
        "claim_id": claim_id,
        "damage_severity": "high",
        "estimated_cost": 5000,
        "recommendation": "Approve for investigation",
        "analyzed_at": datetime.now().isoformat()
    }
    
    claim["analysis"] = analysis
    return analysis

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
