"""Input sanitization utilities for XSS/SQL injection prevention."""
import re
import logging

logger = logging.getLogger("assurancia")


def sanitize_text(text: str, max_length: int = 500) -> str:
    """Sanitize generic text to prevent XSS injection.
    
    - Removes HTML tags
    - Escapes dangerous characters
    - Limits length
    """
    if not text:
        return ""
    
    text = str(text).strip()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Escape dangerous characters
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&#x27;')
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    
    return text


def sanitize_email(email: str) -> str:
    """Sanitize email address.
    
    - Basic email format validation
    - Lowercase normalization
    - Length check
    - Removes dangerous characters
    """
    if not email:
        return ""
    
    email = str(email).strip().lower()
    
    # Basic email validation (RFC 5322 simplified)
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email, re.IGNORECASE):
        logger.warning("Invalid email format: %s", email[:20])
        return ""
    
    if len(email) > 254:  # RFC 5321
        logger.warning("Email too long: %s", email[:20])
        return ""
    
    return email


def sanitize_phone(phone: str) -> str:
    """Sanitize phone number.
    
    - Removes non-digit/plus characters
    - Validates basic format (10+ digits)
    - Prevents SQL/script injection
    """
    if not phone:
        return ""
    
    phone = str(phone).strip()
    
    # Remove common formatting characters
    phone = re.sub(r'[^\d+\-\s().]', '', phone)
    
    # Extract only digits and plus
    digits = re.sub(r'[^\d+]', '', phone)
    
    # Must have at least 10 digits
    digit_count = len(re.sub(r'\D', '', digits))
    if digit_count < 10:
        logger.warning("Phone number too short: %s", phone[:20])
        return ""
    
    # Max length
    if len(digits) > 20:
        logger.warning("Phone number too long: %s", phone[:20])
        return ""
    
    return digits


def sanitize_address(address: str, max_length: int = 500) -> str:
    """Sanitize address field.
    
    - Removes HTML tags
    - Removes script injection attempts
    - Keeps alphanumeric, spaces, and common address characters
    - Limits length
    """
    if not address:
        return ""
    
    address = str(address).strip()
    
    # Remove HTML tags
    address = re.sub(r'<[^>]+>', '', address)
    
    # Remove common script injection patterns
    dangerous_patterns = [
        r'javascript:',
        r'on\w+\s*=',
        r'<script',
        r'eval\(',
        r'expression\(',
    ]
    
    for pattern in dangerous_patterns:
        address = re.sub(pattern, '', address, flags=re.IGNORECASE)
    
    # Escape quotes
    address = address.replace('"', '&quot;')
    address = address.replace("'", '&#x27;')
    
    # Limit length
    if len(address) > max_length:
        address = address[:max_length]
    
    return address


def sanitize_username(username: str, max_length: int = 50) -> str:
    """Sanitize username for authentication.
    
    - Alphanumeric and underscores only
    - No special characters
    - Prevents injection attacks
    """
    if not username:
        return ""
    
    username = str(username).strip()
    
    # Allow only alphanumeric and underscore
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        logger.warning("Invalid username format: %s", username[:20])
        return ""
    
    # Limit length
    if len(username) > max_length:
        username = username[:max_length]
    
    if len(username) < 3:
        logger.warning("Username too short: %s", username)
        return ""
    
    return username


def is_safe(text: str) -> bool:
    """Quick check if text contains obvious injection attempts."""
    if not text:
        return True
    
    dangerous_patterns = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'onclick=',
        r"'.*or.*'",
        r'"; DROP TABLE',
        r"union.*select",
        r'--.*comment',
    ]
    
    text_lower = str(text).lower()
    
    for pattern in dangerous_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.warning("Potential injection detected: %s", text[:50])
            return False
    
    return True
