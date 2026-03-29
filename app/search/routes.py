
from fastapi import APIRouter, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import os
import base64
import json
from datetime import datetime, timedelta
from urllib.parse import urlencode, quote, unquote
from io import BytesIO
from PIL import Image
from zhipuai import ZhipuAI
from service.database import get_db
from sqlalchemy.orm import Session
from service.models import Request as RequestModel, User, Report
import service.security as security
from app.search.match import run_two_stage_screening
from app.search.diagnostic import analyze_item_features, fallback_diagnostic
from fastapi.responses import Response
import struct

search_router = APIRouter(
    prefix="/search",
    tags=["Search"]
)
templates = Jinja2Templates(directory="templates")

TERMINAL_MESSAGE = "Item is not currently in our database. We will notify you by email if a matching record appears."


def _encode_json_payload(payload: dict | None) -> str | None:
    if not payload:
        return None
    try:
        raw = json.dumps(payload, ensure_ascii=False)
        return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")
    except Exception:
        return None


def _decode_json_payload(payload: str | None) -> dict | None:
    if not payload:
        return None
    try:
        raw = base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8")
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _build_failed_url(
    text: str,
    assist_round: int,
    diagnostic_payload: dict | None = None,
    terminal_reached: bool = False,
    terminal_message: str | None = None,
) -> str:
    prefill_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
    params = {
        "prefill_description": prefill_text,
        "assist_round": str(max(0, int(assist_round))),
    }

    payload = _encode_json_payload(diagnostic_payload)
    if payload:
        params["diagnostic_payload"] = payload

    if terminal_reached:
        params["terminal_reached"] = "1"
        if terminal_message:
            params["terminal_message"] = quote(terminal_message)

    return f"/search/match/failed?{urlencode(params)}"


def _trim_and_schedule_terminal_cleanup(record: RequestModel) -> None:
    now = datetime.utcnow()
    retention_days = int(os.getenv("REQUEST_TERMINAL_RETENTION_DAYS", "14"))
    if retention_days < 1:
        retention_days = 1

    record.diagnostic_data = None
    record.identified_features = None
    record.missing_features = None
    record.terminal_reached = True
    record.terminal_reached_at = now
    record.cleanup_due_at = now + timedelta(days=retention_days)
    record.cleanup_status = "pending"


def _cleanup_due_terminal_requests(db: Session, page_size: int = 50) -> int:
    now = datetime.utcnow()
    rows = (
        db.query(RequestModel)
        .filter(RequestModel.terminal_reached == True)  # noqa: E712
        .filter(RequestModel.cleanup_status == "pending")
        .filter(RequestModel.cleanup_due_at.isnot(None))
        .filter(RequestModel.cleanup_due_at <= now)
        .order_by(RequestModel.id.asc())
        .limit(page_size)
        .all()
    )
    if not rows:
        return 0

    for row in rows:
        db.delete(row)

    db.commit()
    return len(rows)


def _require_auth_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.headers.get(
        "Authorization", "").replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="No access token provided")

    payload = security.decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid access token")

    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid access token")

    user = db.query(User).filter(User.email == email).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not authenticated")
    return user


def _optional_auth_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    token = request.headers.get(
        "Authorization", "").replace("Bearer ", "").strip()
    if not token:
        return None
    payload = security.decode_access_token(token)
    if not payload:
        return None
    email = payload.get("sub")
    if not email:
        return None
    user = db.query(User).filter(User.email == email).first()
    if not user or not user.is_active:
        return None
    return user


# Simple XOR token encoding/decoding using hex key
_XOR_KEY_HEX = "A387ED14"
_XOR_KEY = bytes.fromhex(_XOR_KEY_HEX)


def _encode_id_to_token(id_val: int) -> str:
    b = int(id_val).to_bytes(8, "big")
    x = bytes([b[i] ^ _XOR_KEY[i % len(_XOR_KEY)] for i in range(len(b))])
    return x.hex()


def _decode_token_to_id(token: str) -> int:
    try:
        raw = bytes.fromhex(token)
        b = bytes([raw[i] ^ _XOR_KEY[i % len(_XOR_KEY)]
                  for i in range(len(raw))])
        return int.from_bytes(b, "big")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid token")


@search_router.get("/", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse(request, "search/search.html", {"request": request})


@search_router.post("/")
async def search_submit(
    request: Request,
    description: str = Form(None),
    image: UploadFile = File(None),
    current_user: User = Depends(_require_auth_user),
    db: Session = Depends(get_db)
):
    # Logic to handle file uploads and search can be added here
    return templates.TemplateResponse(request, "search/search.html", {"request": request, "success": True})


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
    request_id: int | None = Form(None),
    current_user: User = Depends(_require_auth_user),
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
        if request_id is not None:
            record = db.query(RequestModel).filter(
                RequestModel.id == request_id).first()
            if not record:
                raise HTTPException(
                    status_code=404, detail=f"request_id={request_id} not found")
            if record.user_id and record.user_id != current_user.id:
                raise HTTPException(
                    status_code=403, detail="No permission to update this request")
            if not record.user_id:
                record.user_id = current_user.id
            record.image = image.filename if image else record.image
            record.features_img = json.dumps(vector)
            if not record.description:
                record.description = description or text_out
        else:
            record = RequestModel(
                user_id=current_user.id,
                image=(image.filename if image else None),
                description=(description or text_out),
                features_img=json.dumps(vector),
            )
            db.add(record)
        db.commit()
        db.refresh(record)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db error: {e}")

    return JSONResponse({"id": record.id, "description": text_out, "vector_len": len(vector)})


@search_router.post("/embed_text")
async def request_embed_text(
    request: Request,
    text: str = Form(...),
    request_id: int | None = Form(None),
    current_user: User = Depends(_require_auth_user),
    db: Session = Depends(get_db),
):
    """Convert user-supplied text to a 2048-dim embedding and save to requests.features_dis."""
    try:
        client = _get_client()
        emb_resp = client.embeddings.create(
            model="embedding-3", input=text, dimensions=2048)
        vector = emb_resp.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding error: {e}")

    try:
        if request_id is not None:
            record = db.query(RequestModel).filter(
                RequestModel.id == request_id).first()
            if not record:
                raise HTTPException(
                    status_code=404, detail=f"request_id={request_id} not found")
            if record.user_id and record.user_id != current_user.id:
                raise HTTPException(
                    status_code=403, detail="No permission to update this request")
            if not record.user_id:
                record.user_id = current_user.id
            record.description = text
            record.features_dis = json.dumps(vector)
        else:
            record = RequestModel(
                user_id=current_user.id,
                description=text,
                features_dis=json.dumps(vector),
            )
            db.add(record)
        db.commit()
        db.refresh(record)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db error: {e}")

    return JSONResponse({"id": record.id, "vector_len": len(vector)})


@search_router.post("/screen")
async def search_screen(
    request: Request,
    text: str = Form(...),
    image: UploadFile | None = File(None),
    assist_round: int = Form(0),
    top_n: int = Form(10),
    coarse_top_k: int = Form(30),
    min_coarse_score: float = Form(0.2),
    current_user: User = Depends(_require_auth_user),
    db: Session = Depends(get_db),
):
    """Run two-stage screening and return screening summary + candidate matches."""
    try:
        client = _get_client()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        text_emb_resp = client.embeddings.create(
            model="embedding-3", input=text, dimensions=2048)
        text_vector = text_emb_resp.data[0].embedding
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"text embedding error: {e}")

    image_vector = None
    image_prompt_text = None
    if image:
        try:
            body = await image.read()
            image_b64 = _image_to_base64(body)
            msg_content = [
                {"type": "text", "text": f"User text: {text}"},
                {"type": "text", "text": "Summarize the visible lost-item information from the image or description into a single lost-and-found text no longer than 256 characters. Include category, color, material, key distinguishing features, and scene/location. Strictly no more than 256 characters."},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ]
            vision_resp = client.chat.completions.create(
                model="glm-4v-flash",
                messages=[{"role": "user", "content": msg_content}],
                top_p=0.7,
                temperature=0.0,
            )
            image_prompt_text = vision_resp.choices[0].message.content
            if isinstance(image_prompt_text, list):
                image_prompt_text = "\n".join(
                    [b.get('text', '') for b in image_prompt_text if isinstance(b, dict)])
            if not image_prompt_text:
                image_prompt_text = text

            image_emb_resp = client.embeddings.create(
                model="embedding-3", input=image_prompt_text, dimensions=2048)
            image_vector = image_emb_resp.data[0].embedding
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"image embedding error: {e}")

    try:
        screening = run_two_stage_screening(
            db=db,
            input_text_vector=text_vector,
            input_image_vector=image_vector,
            query_description=text,
            top_n=top_n,
            coarse_top_k=coarse_top_k,
            min_coarse_score=min_coarse_score,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"screening error: {e}")

    # Attach tokenized success URLs for each candidate to support match confirmation flow.
    matches = screening.get("matches") if isinstance(screening, dict) else None
    if isinstance(matches, list):
        for item in matches:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            try:
                token = _encode_id_to_token(int(item_id))
                item["token"] = token
                item["success_url"] = f"/search/match/success/{token}"
            except Exception:
                item["token"] = None
                item["success_url"] = None

    safe_round = max(0, int(assist_round or 0))
    matched_count = len(matches) if isinstance(matches, list) else 0

    if matched_count == 0:
        if safe_round >= 2:
            failed_url = _build_failed_url(
                text=text,
                assist_round=safe_round,
                terminal_reached=True,
                terminal_message=TERMINAL_MESSAGE,
            )
            return JSONResponse({
                "query": {
                    "has_image": bool(image),
                    "text_len": len(text),
                    "image_prompt_text": image_prompt_text,
                },
                "screening": screening,
                "diagnostic": None,
                "terminal": {
                    "reached": True,
                    "message": TERMINAL_MESSAGE,
                },
                "outcome": {
                    "matched_count": 0,
                    "failed_url": failed_url,
                }
            })

        diagnostic = None
        try:
            diagnostic = await analyze_item_features(user_text=text)
        except Exception:
            diagnostic = None

        if not diagnostic:
            diagnostic = fallback_diagnostic(text)

        next_round = safe_round + 1
        failed_url = _build_failed_url(
            text=text,
            assist_round=next_round,
            diagnostic_payload=diagnostic,
            terminal_reached=False,
        )

        return JSONResponse({
            "query": {
                "has_image": bool(image),
                "text_len": len(text),
                "image_prompt_text": image_prompt_text,
            },
            "screening": screening,
            "diagnostic": diagnostic,
            "terminal": {"reached": False},
            "assist_round_next": next_round,
            "outcome": {
                "matched_count": 0,
                "failed_url": failed_url,
            }
        })

    return JSONResponse({
        "query": {
            "has_image": bool(image),
            "text_len": len(text),
            "image_prompt_text": image_prompt_text,
        },
        "screening": screening,
        "diagnostic": None,
        "terminal": {"reached": False},
        "outcome": {
            "matched_count": matched_count,
            "failed_url": _build_failed_url(
                text=text,
                assist_round=safe_round,
                terminal_reached=False,
            ),
        }
    })


@search_router.get('/match/success/{token}', response_class=HTMLResponse)
async def match_success_page(request: Request, token: str, current_user: User | None = Depends(_optional_auth_user), db: Session = Depends(get_db)):
    """Show a candidate found-item (Report) and ask the user to confirm whether it's their item."""
    report_id = _decode_token_to_id(token)
    record = db.query(Report).filter(Report.id == report_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"report not found")
    return templates.TemplateResponse(request, "search/match_success.html", {"request": request, "report": record, "token": token})


@search_router.get('/match/image/{token}')
async def match_image(token: str, db: Session = Depends(get_db)):
    """Return image bytes stored in DB for a given report token."""
    report_id = _decode_token_to_id(token)
    record = db.query(Report).filter(Report.id == report_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="report not found")
    if not getattr(record, 'image_data', None):
        raise HTTPException(status_code=404, detail="image not found")
    try:
        img_bytes = base64.b64decode(record.image_data)
    except Exception:
        raise HTTPException(status_code=500, detail="invalid image data")
    return Response(content=img_bytes, media_type='image/jpeg')


@search_router.post('/match/confirm')
async def match_confirm(
    request: Request,
    token: str = Form(...),
    confirm: str = Form(...),
    next_url: str | None = Form(None),
    current_user: User = Depends(_require_auth_user),
    db: Session = Depends(get_db),
):
    """Handle user's confirmation. If confirmed, delete the matching report from DB. If not, return a redirect URL to the failed page."""
    report_id = _decode_token_to_id(token)
    record = db.query(Report).filter(Report.id == report_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="report not found")

    if str(confirm).lower() in ("yes", "y", "true", "1"):
        try:
            if record.user_id:
                reporter = db.query(User).filter(
                    User.id == record.user_id).first()
                if reporter:
                    reporter.point = (reporter.point or 0) + 2
            db.delete(record)
            db.commit()
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"db delete error: {e}")
        return JSONResponse({"status": "deleted", "id": report_id})

    # not confirmed -> prefer safe next candidate URL if provided by frontend
    if isinstance(next_url, str):
        candidate = next_url.strip()
        if candidate.startswith('/search/match/success/') or candidate.startswith('/search/'):
            return JSONResponse({"status": "denied", "redirect": candidate})

    # fallback to failed page with prefilled description
    prefill = (record.description or "")
    redirect_url = f"/search/match/failed?prefill_description={base64.b64encode(prefill.encode('utf-8')).decode('ascii')}"
    # include the token so the client can't guess IDs easily
    redirect_url = redirect_url + f"&token={token}"
    return JSONResponse({"status": "denied", "redirect": redirect_url})


@search_router.get('/match/failed', response_class=HTMLResponse)
async def match_failed_page(
    request: Request,
    prefill_description: str | None = None,
    assist_round: int = 0,
    diagnostic_payload: str | None = None,
    terminal_reached: int = 0,
    terminal_message: str | None = None,
    current_user: User | None = Depends(_optional_auth_user),
):
    """Render the match-failed page where user can create a new `Request` (lost-item request)."""
    pre = None
    if prefill_description:
        try:
            pre = base64.b64decode(prefill_description).decode('utf-8')
        except Exception:
            pre = prefill_description
    decoded_diagnostic = _decode_json_payload(diagnostic_payload)
    msg = TERMINAL_MESSAGE
    if terminal_message:
        msg = unquote(terminal_message)

    return templates.TemplateResponse(request, "search/match_failed.html", {
        "request": request,
        "prefill_description": pre,
        "assist_round": max(0, int(assist_round or 0)),
        "diagnostic": decoded_diagnostic,
        "terminal_reached": bool(int(terminal_reached or 0)),
        "terminal_message": msg,
        "diagnostic_payload": diagnostic_payload,
    })


@search_router.post('/match/failed/submit')
async def match_failed_submit(
    request: Request,
    description: str = Form(...),
    location: str | None = Form(None),
    assist_round: int = Form(0),
    diagnostic_payload: str | None = Form(None),
    current_user: User = Depends(_require_auth_user),
    db: Session = Depends(get_db),
):
    """Create a new Request record for the user's lost-item report (from failed match flow)."""
    composed = description
    if location:
        composed = f"{composed}\nLocation: {location}"

    safe_round = max(0, int(assist_round or 0))
    diagnostic = _decode_json_payload(diagnostic_payload)

    try:
        record = RequestModel(
            user_id=current_user.id,
            description=composed,
            assist_round=safe_round,
            terminal_reached=False,
        )
        if diagnostic:
            record.diagnostic_data = json.dumps(diagnostic)
            record.identified_features = json.dumps(
                diagnostic.get("identified_features") or {}
            )
            merged_missing = []
            merged_missing.extend(diagnostic.get("missing_attributes") or [])
            merged_missing.extend(diagnostic.get("missing_features") or [])
            record.missing_features = json.dumps(merged_missing)

        if safe_round >= 2:
            _trim_and_schedule_terminal_cleanup(record)

        db.add(record)
        db.commit()
        db.refresh(record)

        # Run lightweight paged cleanup for due terminal records.
        _cleanup_due_terminal_requests(db=db, page_size=50)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db error: {e}")

    # return a tokenized id to avoid exposing raw IDs
    try:
        token = _encode_id_to_token(record.id)
    except Exception:
        token = str(record.id)

    return JSONResponse({
        "status": "created",
        "id": record.id,
        "token": token,
        "assist_round": safe_round,
        "terminal_reached": bool(record.terminal_reached),
        "terminal_message": TERMINAL_MESSAGE if bool(record.terminal_reached) else None,
    })
