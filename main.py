from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
from datetime import datetime
import os

app = FastAPI(title="AssuranceIA API")

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
    """Upload photo"""
    if claim_id in claims_db:
        claims_db[claim_id]["photos"].append(file.filename)
    return {"status": "ok"}

@app.post("/claims/{claim_id}/analyze")
def analyze_claim(claim_id: str):
    """Analyze claim"""
    if claim_id in claims_db:
        claims_db[claim_id]["analysis"] = {
            "damage_severity": "medium",
            "estimated_cost": 3000,
            "recommendation": "Manual inspection required"
        }
        return claims_db[claim_id]["analysis"]
    return {"error": "Not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
