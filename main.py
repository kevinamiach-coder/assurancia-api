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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

# Gmail configuration
GMAIL_EMAIL = os.environ.get("GMAIL_EMAIL", "artisanpatrimoinefrancais@gmail.com")
GMAIL_PASSWORD = os.environ.get("GMAIL_PASSWORD", "")

# Pydantic models
class ClaimCreate(BaseModel):
    user_email: str
    damage_type: str
    address: str
    description: str = ""

def send_email(to_email: str, claim_id: str, claim_data: dict, analysis: dict):
    """Send email recap of claim and analysis"""
    try:
        if not GMAIL_PASSWORD:
            print("Gmail password not configured")
            return False
        
        # Create email
        subject = f"AssuranceIA™ - Sinistre {claim_id} analysé"
        
        body = f"""
Bonjour,

Votre sinistre a été enregistré et analysé par AssuranceIA™.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 DÉTAILS DU SINISTRE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ID: {claim_id}
Email: {claim_data.get('user_email', '')}
Adresse: {claim_data.get('address', '')}
Type: {claim_data.get('damage_type', '')}
Description: {claim_data.get('description', '')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 ANALYSE IA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sévérité: {analysis.get('damage_severity', 'N/A')}
Coût estimé: €{analysis.get('estimated_cost', 0)}
Recommandation: {analysis.get('recommendation', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Merci d'avoir utilisé AssuranceIA™
https://assurancia-api-2.onrender.com/

Cordialement,
L'équipe AssuranceIA™
"""
        
        # Send via Gmail SMTP
        msg = MIMEMultipart()
        msg['From'] = GMAIL_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_EMAIL, GMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print(f"Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"Email error: {str(e)}")
        return False

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
        
        # Send email with analysis
        send_email(
            claim.get("user_email", ""),
            claim_id,
            claim,
            analysis
        )
        
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
