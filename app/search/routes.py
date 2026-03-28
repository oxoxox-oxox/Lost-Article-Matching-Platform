
from fastapi import APIRouter, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import os
import base64
import json
from io import BytesIO
from PIL import Image
from zhipuai import ZhipuAI
from service.database import get_db
from sqlalchemy.orm import Session
from service.models import Request as RequestModel

search_router = APIRouter(
    prefix="/request",
    tags=["Request"]
)
templates = Jinja2Templates(directory="templates")


@search_router.get("/", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse("search/search.html", {"request": request})


@search_router.post("/")
async def search_submit(
    request: Request,
    description: str = Form(None),
    image: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    # Logic to handle file uploads and search can be added here
    return templates.TemplateResponse("search/search.html", {"request": request, "success": True})


def _get_client():
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise RuntimeError("ZHIPUAI_API_KEY is not set in environment")
    return ZhipuAI(api_key=api_key)


def _image_to_base64(image_bytes: bytes) -> str:
    img = Image.open(BytesIO(image_bytes))
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.thumbnail((1024, 1024))
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _build_prompt_from_image_and_text(image_b64: str | None, user_text: str | None) -> str:
    base = (
        "Please generate a lost-and-found description no longer than 256 characters, including item category, color, material, distinguishing features, and scene/location."
        " Do not include emotions or extended narrative — only factual attributes."
    )
    if user_text:
        combined = f"{base}\nUser description: {user_text}"
    else:
        combined = base

    if image_b64:
        combined = combined + \
            "\n(Image attached; supplement features from the image.)"

    return combined


@search_router.post("/embed_image")
async def request_embed_image(
    request: Request,
    image: UploadFile | None = File(None),
    description: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """Process optional image + text and save a 2048-dim image-derived prompt embedding to requests.features_img."""
    try:
        client = _get_client()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    image_b64 = None
    if image:
        body = await image.read()
        image_b64 = _image_to_base64(body)

    prompt = _build_prompt_from_image_and_text(image_b64, description)

    try:
        msg_content = [
            {"type": "text", "text": "Summarize the visible lost-item information from the image or description into a single lost-and-found text no longer than 256 characters. Include category, color, material, key distinguishing features, and scene/location. Strictly no more than 256 characters."}
        ]
        if image_b64:
            msg_content.append(
                {"type": "image_url", "image_url": {"url": image_b64}})
        if description:
            msg_content.insert(
                0, {"type": "text", "text": f"User text: {description}"})

        resp = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[{"role": "user", "content": msg_content}],
            top_p=0.7,
            temperature=0.0,
        )
        text_out = resp.choices[0].message.content
        if isinstance(text_out, list):
            text_out = "\n".join([b.get('text', '')
                                 for b in text_out if isinstance(b, dict)])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vision model error: {e}")

    if not text_out:
        text_out = (prompt[:256])

    try:
        emb_resp = client.embeddings.create(
            model="embedding-3", input=text_out, dimensions=2048)
        vector = emb_resp.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding error: {e}")

    try:
        record = RequestModel(image=(image.filename if image else None),
                              description=text_out, features_img=json.dumps(vector))
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db error: {e}")

    return JSONResponse({"id": record.id, "description": text_out, "vector_len": len(vector)})


@search_router.post("/embed_text")
async def request_embed_text(text: str = Form(...), db: Session = Depends(get_db)):
    """Convert user-supplied text to a 2048-dim embedding and save to requests.features_dis."""
    try:
        client = _get_client()
        emb_resp = client.embeddings.create(
            model="embedding-3", input=text, dimensions=2048)
        vector = emb_resp.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding error: {e}")

    try:
        record = RequestModel(
            description=text, features_dis=json.dumps(vector))
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db error: {e}")

    return JSONResponse({"id": record.id, "vector_len": len(vector)})
