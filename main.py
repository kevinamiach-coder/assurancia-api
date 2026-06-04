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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from supabase import create_client, Client

app = FastAPI(title="AssuranceIA API")

# Get environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = None
print(f"DEBUG: SUPABASE_URL = {SUPABASE_URL}")
print(f"DEBUG: SUPABASE_KEY = {SUPABASE_KEY[:20] if SUPABASE_KEY else 'None'}...")
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("DEBUG: Supabase client initialized successfully ✅")
    except Exception as e:
        print(f"DEBUG: Supabase initialization failed: {str(e)} ❌")
else:
    print("DEBUG: SUPABASE_URL or SUPABASE_KEY not set ❌")

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

# Note: Using Supabase instead of in-memory database

# Email function
def send_email(to_email: str, subject: str, html_body: str, attachment_data: bytes = None, attachment_filename: str = None):
    """Send email via Gmail SMTP"""
    try:
        if not GMAIL_EMAIL or not GMAIL_PASSWORD:
            print("Gmail not configured")
            return False

        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = GMAIL_EMAIL
        msg["To"] = to_email

        # Add HTML body
        msg.attach(MIMEText(html_body, "html"))

        # Add attachment if provided
        if attachment_data and attachment_filename:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment_data)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename= {attachment_filename}")
            msg.attach(part)

        # Send via Gmail
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(GMAIL_EMAIL, GMAIL_PASSWORD)
        server.sendmail(GMAIL_EMAIL, to_email, msg.as_string())
        server.quit()

        return True
    except Exception as e:
        print(f"Email error: {str(e)}")
        return False

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
    try:
        claim_id = f"CLM-{datetime.now().year}-{uuid.uuid4().hex[:6].upper()}"

        # Insert into Supabase
        claim_data = {
            "claim_id": claim_id,
            "user_email": claim.user_email,
            "damage_type": claim.damage_type,
            "address": claim.address,
            "description": claim.description,
            "photos": [],
            "analysis": None
        }

        if supabase:
            print(f"DEBUG: Inserting claim {claim_id} into Supabase...")
            try:
                response = supabase.table("claims").insert(claim_data).execute()
                print(f"DEBUG: Supabase insert response: {response}")
                if response.data:
                    claim_data = response.data[0]
                    print(f"DEBUG: Claim inserted successfully ✅")
                else:
                    print(f"DEBUG: No data returned from insert ❌")
            except Exception as e:
                print(f"DEBUG: Supabase insert error: {str(e)} ❌")
        else:
            print(f"DEBUG: Supabase client not initialized, skipping insert")

        # Send confirmation email
        email_subject = f"🔍 Nouveau sinistre créé - AssuranceIA™ ({claim_id})"
        email_body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; color: #333;">
                <h2>Sinistre Enregistré</h2>
                <p><strong>Numéro de dossier :</strong> {claim_id}</p>
                <hr>
                <h3>Détails du sinistre</h3>
                <ul>
                    <li><strong>Email :</strong> {claim.user_email}</li>
                    <li><strong>Adresse :</strong> {claim.address}</li>
                    <li><strong>Type de dégât :</strong> {claim.damage_type}</li>
                    <li><strong>Description :</strong> {claim.description}</li>
                    <li><strong>Date de création :</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</li>
                </ul>
                <hr>
                <p>Prochaine étape : <strong>Uploadez les photos</strong> et demandez une analyse.</p>
                <p style="color: #666; font-size: 12px;">AssuranceIA™ - Validation automatique des sinistres eau</p>
            </body>
        </html>
        """
        send_email(claim.user_email, email_subject, email_body)

        return claim_data
    except Exception as e:
        return {"error": str(e)}

@app.get("/claims")
def get_claims():
    """Get all claims"""
    try:
        if supabase:
            response = supabase.table("claims").select("*").execute()
            return response.data if response.data else []
        return []
    except Exception as e:
        return {"error": str(e)}

@app.post("/claims/{claim_id}/photos")
async def upload_photo(claim_id: str, file: UploadFile = File(...)):
    """Upload photo - store as base64"""
    try:
        if not supabase:
            return {"error": "Database not configured"}

        # Get existing claim
        response = supabase.table("claims").select("*").eq("claim_id", claim_id).execute()
        if not response.data:
            return {"error": "Claim not found"}

        claim = response.data[0]

        # Read file content
        contents = await file.read()
        # Convert to base64
        base64_content = base64.b64encode(contents).decode('utf-8')

        # Add photo to photos array
        photos = claim.get("photos", []) or []
        photos.append({
            "filename": file.filename,
            "content_type": file.content_type,
            "data": base64_content
        })

        # Update claim in Supabase
        supabase.table("claims").update({"photos": photos}).eq("claim_id", claim_id).execute()

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
    try:
        if not supabase:
            return {"error": "Database not configured"}

        # Get claim from Supabase
        response = supabase.table("claims").select("*").eq("claim_id", claim_id).execute()
        if not response.data:
            return {"error": "Claim not found"}

        claim = response.data[0]

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
                "Authorization": f"Bearer {ANTHROPIC_API_KEY}",
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

            # Call Claude API (with longer timeout for image analysis)
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=120
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

    # Update claim in Supabase with analysis results
    if supabase:
        supabase.table("claims").update({"analysis": analysis}).eq("claim_id", claim_id).execute()

    # Send analysis email with photo attachment
    try:
        severity_emoji = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🔴",
            "critical": "⛔"
        }.get(analysis.get("damage_severity", "unknown"), "❓")

        email_subject = f"{severity_emoji} Analyse du sinistre {claim_id} - AssuranceIA™"

        analysis_html = f"""
        <html>
            <body style="font-family: Arial, sans-serif; color: #333;">
                <h2>📋 Résultats de l'analyse - {claim_id}</h2>
                <hr>

                <h3>📊 Analyse</h3>
                <ul>
                    <li><strong>Sévérité :</strong> {severity_emoji} {analysis.get('damage_severity', 'unknown').upper()}</li>
                    <li><strong>Coût estimé :</strong> {analysis.get('estimated_cost', 'N/A')} €</li>
                    <li><strong>Confiance :</strong> {analysis.get('confidence', 'N/A')}</li>
                </ul>

                <h3>🔍 Dégâts visibles</h3>
                <p>{analysis.get('visible_damage', 'Aucune description disponible')}</p>

                <h3>✅ Recommandation</h3>
                <p><strong>{analysis.get('recommendation', 'Aucune recommandation')}</strong></p>

                <hr>
                <p style="color: #666; font-size: 12px;">Photo jointe en pièce jointe</p>
                <p style="color: #666; font-size: 12px;">AssuranceIA™ - Validation automatique des sinistres eau</p>
            </body>
        </html>
        """

        # Attach the first photo if available
        if claim["photos"]:
            photo_data = claim["photos"][0]["data"]
            photo_bytes = base64.b64decode(photo_data)
            photo_filename = claim["photos"][0]["filename"]
            send_email(claim["user_email"], email_subject, analysis_html, photo_bytes, photo_filename)
        else:
            send_email(claim["user_email"], email_subject, analysis_html)
    except Exception as e:
        print(f"Error sending analysis email: {str(e)}")

        return analysis
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
