"""Audit logging utilities for AssuranceIA™."""
import os
import logging
from datetime import datetime
from pymongo import MongoClient

logger = logging.getLogger("assurancia")


def get_audit_collection():
    """Get MongoDB audit_logs collection."""
    try:
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            logger.error("MONGODB_URI not set - audit logging disabled")
            return None
        
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        db = client["assurancia"]
        return db["audit_logs"]
    except Exception as e:
        logger.error("Failed to get audit collection: %s", e)
        return None


def audit_log(
    insurer_id: str,
    action: str,
    claim_id: str = None,
    ip_address: str = None,
    status: str = "success",
    details: dict = None,
):
    """Log an action to MongoDB audit_logs collection.
    
    Args:
        insurer_id: ID of the insurer performing the action
        action: Action name (authenticate, create_declaration_link, etc.)
        claim_id: Optional claim ID (for claim-related actions)
        ip_address: Client IP address
        status: "success" or "failure"
        details: Optional additional details dict
    
    Returns:
        True if logged to MongoDB, False if fallback-only
    """
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "insurer_id": insurer_id,
        "action": action,
        "claim_id": claim_id,
        "ip_address": ip_address,
        "status": status,
        "details": details or {},
    }
    
    # Always log locally
    log_msg = f"AUDIT [{action}] insurer={insurer_id} status={status}"
    if claim_id:
        log_msg += f" claim={claim_id}"
    if ip_address:
        log_msg += f" ip={ip_address}"
    logger.info(log_msg)
    
    # Try to persist to MongoDB
    try:
        collection = get_audit_collection()
        if collection:
            collection.insert_one(audit_entry)
            return True
    except Exception as e:
        logger.error("Failed to write audit log to MongoDB: %s", e)
    
    return False


# Convenience functions for specific actions

def audit_authenticate(username: str, ip_address: str, success: bool):
    """Log an authentication attempt."""
    audit_log(
        insurer_id=username,
        action="authenticate",
        ip_address=ip_address,
        status="success" if success else "failure",
        details={"username": username}
    )


def audit_create_declaration_link(insurer_id: str, token: str, client_email: str):
    """Log declaration link creation."""
    audit_log(
        insurer_id=insurer_id,
        action="create_declaration_link",
        status="success",
        details={"token": token[:8], "client_email": client_email}
    )


def audit_submit_declaration(insurer_id: str, claim_id: str, ip_address: str):
    """Log declaration submission."""
    audit_log(
        insurer_id=insurer_id,
        action="submit_declaration",
        claim_id=claim_id,
        ip_address=ip_address,
        status="success",
    )


def audit_upload_photo(insurer_id: str, claim_id: str, ip_address: str, filename: str):
    """Log photo upload."""
    audit_log(
        insurer_id=insurer_id,
        action="upload_photo",
        claim_id=claim_id,
        ip_address=ip_address,
        status="success",
        details={"filename": filename}
    )


def audit_analyze_claim(insurer_id: str, claim_id: str):
    """Log Claude Vision analysis."""
    audit_log(
        insurer_id=insurer_id,
        action="analyze_claim",
        claim_id=claim_id,
        status="success",
    )


def audit_attest_claim(insurer_id: str, claim_id: str):
    """Log claim attestation."""
    audit_log(
        insurer_id=insurer_id,
        action="attest_claim",
        claim_id=claim_id,
        status="success",
    )


def audit_view_claim(insurer_id: str, claim_id: str, ip_address: str):
    """Log claim view."""
    audit_log(
        insurer_id=insurer_id,
        action="view_claim",
        claim_id=claim_id,
        ip_address=ip_address,
        status="success",
    )


def audit_download_pdf(insurer_id: str, claim_id: str, ip_address: str):
    """Log PDF download."""
    audit_log(
        insurer_id=insurer_id,
        action="download_pdf",
        claim_id=claim_id,
        ip_address=ip_address,
        status="success",
    )


def audit_email_pdf(insurer_id: str, claim_id: str, recipient_email: str, success: bool):
    """Log PDF email send."""
    audit_log(
        insurer_id=insurer_id,
        action="email_pdf",
        claim_id=claim_id,
        status="success" if success else "failure",
        details={"recipient": recipient_email}
    )


def audit_delete_account(insurer_id: str, ip_address: str):
    """Log account deletion (GDPR)."""
    audit_log(
        insurer_id=insurer_id,
        action="delete_account",
        ip_address=ip_address,
        status="success",
        details={"deleted_at": datetime.now().isoformat()}
    )


def get_audit_logs(insurer_id: str = None, action: str = None, limit: int = 100) -> list:
    """Retrieve audit logs from MongoDB.
    
    Args:
        insurer_id: Filter by insurer (optional)
        action: Filter by action (optional)
        limit: Max results (default 100)
    
    Returns:
        List of audit entries
    """
    try:
        collection = get_audit_collection()
        if not collection:
            return []
        
        query = {}
        if insurer_id:
            query["insurer_id"] = insurer_id
        if action:
            query["action"] = action
        
        return list(
            collection.find(query, {"_id": 0})
            .sort("timestamp", -1)
            .limit(limit)
        )
    except Exception as e:
        logger.error("Failed to retrieve audit logs: %s", e)
        return []
