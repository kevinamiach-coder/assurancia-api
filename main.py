from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic
import uuid
from datetime import datetime
import os
import base64

app = FastAPI(title="AssuranceIA API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anthropic client
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

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
    
    # Default analysis if no API key or no photos
    if not client or not claim["photos"]:
        analysis = {
            "claim_id": claim_id,
            "damage_severity": "medium",
            "estimated_cost": 3000,
            "recommendation": "Manual inspection required",
            "analyzed_at": datetime.now().isoformat(),
            "note": "No API key or photos available for AI analysis"
        }
        claim["analysis"] = analysis
        return analysis
    
    try:
        # Build image content for Claude Vision
        image_content = []
        
        # Add text description
        image_content.append({
            "type": "text",
            "text": f"""Analyze this water damage insurance claim and provide:
1. Damage severity (low/medium/high/critical)
2. Estimated repair cost in euros
3. Recommendations for approval/denial
4. Key observations from the images

Claim details:
- Type: {claim['damage_type']}
- Address: {claim['address']}
- Description: {claim['description']}"""
        })
        
        # Add photos
        for photo_id in claim["photos"]:
            if photo_id in photos_db:
                photo = photos_db[photo_id]
                image_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": photo["media_type"],
                        "data": photo["content_base64"]
                    }
                })
        
        # Call Claude Vision API
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": image_content
                }
            ]
        )
        
        # Parse Claude's response
        claude_response = message.content[0].text
        
        # Extract key info from Claude's analysis
        analysis = {
            "claim_id": claim_id,
            "damage_severity": "high" if "critical" in claude_response.lower() or "high" in claude_response.lower() else "medium",
            "estimated_cost": 5000,  # Parse from Claude response if needed
            "recommendation": "Approve for investigation" if "approve" in claude_response.lower() else "Manual review required",
            "claude_analysis": claude_response,
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
