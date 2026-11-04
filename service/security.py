import os
import hmac
import hashlib
import json
import base64
import time
import secrets
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from jose import JWTError, jwt
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# pbkdf2_sha256: Effectively resists brute-force and rainbow table attacks.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + \
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# --- URL-Safe Cryptographic Signed Tokens for Resource References ---
_LEGACY_XOR_KEY_HEX = "A387ED14"
_LEGACY_XOR_KEY = bytes.fromhex(_LEGACY_XOR_KEY_HEX)


def _get_signing_key() -> bytes:
    key_str = SECRET_KEY or "fallback_secret_key_lamp_2026"
    return hashlib.sha256(key_str.encode("utf-8")).digest()


def generate_signed_token(id_val: int, purpose: str = "report", expires_in: int = 86400) -> str:
    """Generate a tamper-proof URL-safe token containing id, timestamp, purpose, and HMAC-SHA256 signature."""
    now = int(time.time())
    payload = {
        "id": int(id_val),
        "ts": now,
        "exp": now + expires_in,
        "pur": str(purpose),
    }
    raw_payload = json.dumps(payload, separators=(',', ':'), ensure_ascii=True).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(raw_payload).decode("ascii").rstrip("=")

    key = _get_signing_key()
    signature = hmac.new(key, payload_b64.encode("ascii"), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")

    return f"{payload_b64}.{sig_b64}"


def verify_signed_token(token: str, expected_purpose: Optional[str] = "report", max_age: int = 86400) -> Optional[int]:
    """Verify HMAC signature, purpose, and timestamp of the signed token.
    Falls back to legacy XOR token if token matches legacy format, ensuring zero breakage for in-flight requests.
    Returns integer ID if valid, or None if invalid/expired.
    """
    if not token or not isinstance(token, str):
        return None

    # Check for modern signed token (payload.signature)
    if "." in token:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64, sig_b64 = parts

        # Verify signature in constant time
        key = _get_signing_key()
        expected_sig = hmac.new(key, payload_b64.encode("ascii"), hashlib.sha256).digest()
        expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode("ascii").rstrip("=")

        if not secrets.compare_digest(sig_b64, expected_sig_b64):
            return None

        # Decode payload
        try:
            padded = payload_b64 + "=" * (-len(payload_b64) % 4)
            raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
            data = json.loads(raw)
        except Exception:
            return None

        # Check timestamp & expiration
        now = int(time.time())
        exp = data.get("exp")
        ts = data.get("ts", 0)
        if exp and now > exp:
            return None
        if now - ts > max_age:
            return None

        # Check purpose if expected_purpose provided
        if expected_purpose and data.get("pur") != expected_purpose:
            return None

        return int(data.get("id"))

    # Legacy XOR fallback (16-char hex string)
    if len(token) == 16:
        try:
            raw = bytes.fromhex(token)
            b = bytes([raw[i] ^ _LEGACY_XOR_KEY[i % len(_LEGACY_XOR_KEY)] for i in range(len(raw))])
            return int.from_bytes(b, "big")
        except Exception:
            pass

    return None
