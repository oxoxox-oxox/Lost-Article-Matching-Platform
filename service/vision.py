"""Production multimodal vision and vector embedding service for LAMP.

Consolidates:
- Image compression and thumbnail Base64 encoding (Pillow)
- Lost-and-found prompt construction and LLM vision summarization (glm-4v-flash)
- Text embedding generation (embedding-3)
- Vector serialization and cosine similarity computation (NumPy)
"""

import os
import json
import base64
from io import BytesIO
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image
from zhipuai import ZhipuAI

_ZHIPU_CLIENT: Optional[ZhipuAI] = None


def get_zhipu_client() -> ZhipuAI:
    """Get or initialize singleton ZhipuAI client instance."""
    global _ZHIPU_CLIENT
    if _ZHIPU_CLIENT is not None:
        return _ZHIPU_CLIENT

    api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("ZHIPUAI_API_KEY (or OPENAI_API_KEY) is not set in environment")

    _ZHIPU_CLIENT = ZhipuAI(api_key=api_key)
    return _ZHIPU_CLIENT


def compress_image_to_base64(image_bytes: bytes, max_dim: int = 1024, quality: int = 85) -> str:
    """Resize image to thumbnail within max_dim x max_dim, convert to RGB, and return JPEG base64 string."""
    img = Image.open(BytesIO(image_bytes))
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    img.thumbnail((max_dim, max_dim))
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def build_multimodal_prompt(image_b64: Optional[str] = None, user_text: Optional[str] = None) -> str:
    """Construct structured lost-and-found prompt for feature extraction."""
    base = (
        "Please generate a lost-and-found description no longer than 256 characters, including item category, color, material, distinguishing features, and scene/location."
        " Do not include emotions or extended narrative — only factual attributes."
    )
    if user_text:
        combined = f"{base}\nUser description: {user_text}"
    else:
        combined = base

    if image_b64:
        combined = combined + "\n(Image attached; supplement features from the image.)"

    return combined


def extract_multimodal_summary(
    image_b64: Optional[str] = None,
    user_text: Optional[str] = None,
    client: Optional[ZhipuAI] = None
) -> str:
    """Call vision model (glm-4v-flash) to extract a concise <=256-character lost-and-found summary."""
    c = client or get_zhipu_client()
    fallback_prompt = build_multimodal_prompt(image_b64, user_text)

    msg_content = [
        {"type": "text", "text": "Summarize the visible lost-item information from the image or description into a single lost-and-found text no longer than 256 characters. Include category, color, material, key distinguishing features, and scene/location. Strictly no more than 256 characters."}
    ]
    if image_b64:
        msg_content.append({"type": "image_url", "image_url": {"url": image_b64}})
    if user_text:
        msg_content.insert(0, {"type": "text", "text": f"User text: {user_text}"})

    resp = c.chat.completions.create(
        model="glm-4v-flash",
        messages=[{"role": "user", "content": msg_content}],
        top_p=0.7,
        temperature=0.0,
    )

    text_out = resp.choices[0].message.content
    if isinstance(text_out, list):
        text_out = "\n".join([b.get('text', '') for b in text_out if isinstance(b, dict)])

    if not text_out or not str(text_out).strip():
        if user_text and str(user_text).strip():
            text_out = str(user_text).strip()[:256]
        else:
            text_out = fallback_prompt[:256]

    return str(text_out).strip()


def generate_embedding(
    text: str,
    dimensions: int = 2048,
    client: Optional[ZhipuAI] = None
) -> List[float]:
    """Convert input text to a normalized embedding vector using embedding-3."""
    c = client or get_zhipu_client()
    emb_resp = c.embeddings.create(
        model="embedding-3",
        input=text,
        dimensions=dimensions
    )
    return emb_resp.data[0].embedding


def extract_item_features(
    image_bytes: Optional[bytes] = None,
    user_text: Optional[str] = None,
    client: Optional[ZhipuAI] = None
) -> Tuple[str, List[float], Optional[str]]:
    """High-level facade method:
    1. Compress image to base64 thumbnail (if provided)
    2. Extract concise <=256 char summary via vision LLM
    3. Generate 2048-dim vector embedding
    Returns: (summary_text, embedding_vector, image_b64)
    """
    c = client or get_zhipu_client()

    image_b64 = None
    if image_bytes:
        image_b64 = compress_image_to_base64(image_bytes)

    summary_text = extract_multimodal_summary(image_b64=image_b64, user_text=user_text, client=c)
    vector = generate_embedding(text=summary_text, dimensions=2048, client=c)

    return summary_text, vector, image_b64


def parse_vector(raw_json: Optional[str]) -> Optional[List[float]]:
    """Safely deserialize JSON string to vector list."""
    if not raw_json:
        return None
    try:
        return json.loads(raw_json)
    except Exception:
        return None


def cosine_similarity(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    """Compute cosine similarity between two float vectors. Returns -1.0 if invalid."""
    if a is None or b is None:
        return -1.0
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    if va.size == 0 or vb.size == 0:
        return -1.0
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return -1.0
    return float(np.dot(va, vb) / denom)
