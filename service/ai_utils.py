import base64
import json
import os
import re
from urllib import request as urllib_request
from urllib.error import URLError, HTTPError
from io import BytesIO
from typing import Any

from PIL import Image
from zhipuai import ZhipuAI


def _resolve_zhipu_key() -> str | None:
    return os.getenv("ZHIPUAI_API_KEY") or os.getenv("zhipuAPI")


def get_zhipu_client() -> ZhipuAI:
    api_key = _resolve_zhipu_key()
    if not api_key:
        raise RuntimeError("ZHIPUAI_API_KEY is not set in environment")
    return ZhipuAI(api_key=api_key)


def image_to_base64(image_bytes: bytes) -> str:
    img = Image.open(BytesIO(image_bytes))
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img.thumbnail((1024, 1024))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_prompt_from_image_and_text(image_b64: str | None, user_text: str | None) -> str:
    base = (
        "Please generate a lost-and-found description no longer than 256 characters, including item category, "
        "color, material, distinguishing features, and scene/location. "
        "Do not include emotions or extended narrative - only factual attributes."
    )
    if user_text:
        combined = f"{base}\nUser description: {user_text}"
    else:
        combined = base

    if image_b64:
        combined = combined + \
            "\n(Image attached; supplement features from the image.)"

    return combined


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    content = text.strip()

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\\s*", "", content)
        content = re.sub(r"\\s*```$", "", content)

    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_markdown_text(text: str) -> str:
    normalized = to_str(text)
    normalized = normalized.replace("\\r\\n", "\n")
    normalized = normalized.replace("\\n", "\n")
    normalized = normalized.replace("\\t", "\t")
    return normalized


def image_to_text_external_api(
    image_b64: str,
    user_text: str | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: int = 20,
) -> str | None:
    """Call external image-to-text API and return text description.

    Expected response JSON can be one of:
    - {"text": "..."}
    - {"description": "..."}
    - {"result": "..."}
    """
    resolved_url = api_url or os.getenv("EXTERNAL_IMAGE2TEXT_URL")
    if not resolved_url:
        return None

    resolved_key = api_key or os.getenv(
        "EXTERNAL_IMAGE2TEXT_API_KEY") or _resolve_zhipu_key()
    payload = {
        "image_base64": image_b64,
        "user_text": user_text or "",
    }

    headers = {
        "Content-Type": "application/json",
    }
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"

    req = urllib_request.Request(
        resolved_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError):
        return None

    try:
        parsed = json.loads(body)
    except Exception:
        return None

    if not isinstance(parsed, dict):
        return None

    for key in ("text", "description", "result"):
        value = to_str(parsed.get(key))
        if value:
            return value
    return None
