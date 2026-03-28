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
from service.models import Report, User, Request as RequestModel
import service.security as security
from service.email import send_match_notification_email
import numpy as np

report_router = APIRouter(
    prefix="/report",
    tags=["Report"]
)
templates = Jinja2Templates(directory="templates")


def _parse_vector(v: str | None):
    if not v:
        return None
    try:
        return json.loads(v)
    except Exception:
        return None


def _cosine_similarity(a, b) -> float:
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


def _best_similarity(report_row: Report, request_row: RequestModel) -> float:
    r_text = _parse_vector(report_row.features_dis)
    r_img = _parse_vector(report_row.features_img)
    q_text = _parse_vector(request_row.features_dis)
    q_img = _parse_vector(request_row.features_img)

    scores = [
        _cosine_similarity(r_text, q_text),
        _cosine_similarity(r_img, q_img),
        _cosine_similarity(r_text, q_img),
        _cosine_similarity(r_img, q_text),
    ]
    return max(scores)


def _normalize_text_for_compare(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().lower().split())


def _get_notify_threshold() -> float:
    try:
        return float(os.getenv("MATCH_NOTIFY_THRESHOLD", "0.80"))
    except Exception:
        return 0.80


def _notify_best_request_match(db: Session, report_row: Report, threshold: float | None = None) -> dict:
    """Find the best pending request for a report and send notification email if score passes threshold."""
    threshold = _get_notify_threshold() if threshold is None else threshold

    if report_row.status:
        return {"sent": False, "reason": "report_already_processed", "best_score": None, "threshold": threshold}
    if not report_row.features_dis and not report_row.features_img:
        return {"sent": False, "reason": "report_has_no_vectors", "best_score": None, "threshold": threshold}

    candidates = db.query(RequestModel).filter(
        RequestModel.status == False,  # noqa: E712
        RequestModel.user_id.isnot(None)
    ).all()
    best_req = None
    best_score = -1.0

    report_text = _normalize_text_for_compare(report_row.description)

    for req in candidates:
        score = _best_similarity(report_row, req)
        req_text = _normalize_text_for_compare(req.description)
        # If descriptions are exactly the same after normalization, give a small confidence boost.
        if report_text and req_text and report_text == req_text:
            score = min(1.0, score + 0.05)
        if score > best_score:
            best_score = score
            best_req = req

    if best_req is None:
        return {"sent": False, "reason": "no_candidate_requests", "best_score": None, "threshold": threshold}
    if best_score < threshold:
        return {
            "sent": False,
            "reason": "score_below_threshold",
            "best_score": best_score,
            "threshold": threshold,
            "best_request_id": best_req.id,
        }
    if not best_req.user_id:
        return {"sent": False, "reason": "best_request_no_user", "best_score": best_score, "threshold": threshold}

    target_user = db.query(User).filter(User.id == best_req.user_id).first()
    if not target_user or not target_user.email:
        return {"sent": False, "reason": "target_user_email_missing", "best_score": best_score, "threshold": threshold}

    sent = send_match_notification_email(
        email=target_user.email,
        report_description=report_row.description,
        request_description=best_req.description,
        score=best_score,
    )

    # Mark as processed to avoid repeated notifications for the same report.
    if sent:
        report_row.status = True
        db.commit()
        return {
            "sent": True,
            "reason": "mail_sent",
            "best_score": best_score,
            "threshold": threshold,
            "best_request_id": best_req.id,
            "target_email": target_user.email,
        }

    return {
        "sent": False,
        "reason": "mail_send_failed",
        "best_score": best_score,
        "threshold": threshold,
        "best_request_id": best_req.id,
        "target_email": target_user.email,
    }


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


@report_router.get("/", response_class=HTMLResponse)
async def report_page(request: Request):
    return templates.TemplateResponse(request, "report/report.html", {"request": request})


@report_router.post("/")
async def report_submit(
    request: Request,
    image: UploadFile = File(None),
    time: str = Form(None),
    location: str = Form(None),
    description: str = Form(None),
    current_user: User = Depends(_require_auth_user),
    db: Session = Depends(get_db)
):
    # 这里可以添加处理上传文件和存储信息的逻辑
    return templates.TemplateResponse(request, "report/report.html", {"request": request, "success": True})


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
        # use user's text as additional context
        combined = f"{base}\nUser description: {user_text}"
    else:
        combined = base

    if image_b64:
        combined = combined + \
            "\n(Image attached; supplement features from the image.)"

    # ensure <= 256 chars when returned; we'll ask the model for that
    return combined


@report_router.post("/embed_image")
async def report_embed_image(
    request: Request,
    image: UploadFile | None = File(None),
    description: str | None = Form(None),
    report_id: int | None = Form(None),
    current_user: User = Depends(_require_auth_user),
    db: Session = Depends(get_db),
):
    """Accept optional image and description, produce a <=256-char lost-and-found prompt via vision model
    and convert it to a 2048-dim embedding saved to the `reports` table as `features_img`.
    """
    try:
        client = _get_client()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    image_b64 = None
    if image:
        body = await image.read()
        image_b64 = _image_to_base64(body)

    prompt = _build_prompt_from_image_and_text(image_b64, description)

    # call vision chat model to get concise description (<=256 chars)
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
        # best-effort extraction of text from response
        text_out = resp.choices[0].message.content
        if isinstance(text_out, list):
            # sometimes content returns list of blocks
            text_out = "\n".join([b.get('text', '')
                                 for b in text_out if isinstance(b, dict)])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vision model error: {e}")

    # fallback to prompt if model didn't return useful text
    if not text_out:
        # truncate to 256 chars
        text_out = (prompt[:256])

    # get embedding
    try:
        emb_resp = client.embeddings.create(
            model="embedding-3", input=text_out, dimensions=2048)
        vector = emb_resp.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding error: {e}")

    # persist to DB
    try:
        if report_id is not None:
            record = db.query(Report).filter(Report.id == report_id).first()
            if not record:
                raise HTTPException(
                    status_code=404, detail=f"report_id={report_id} not found")
            if record.user_id and record.user_id != current_user.id:
                raise HTTPException(
                    status_code=403, detail="No permission to update this report")
            if not record.user_id:
                record.user_id = current_user.id
            record.image = image.filename if image else record.image
            record.features_img = json.dumps(vector)
            if not record.description:
                record.description = description or text_out
        else:
            record = Report(
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

    notify_result = None
    try:
        notify_result = _notify_best_request_match(db, record)
    except Exception:
        # Do not block normal API response if mail sending fails.
        notify_result = {"sent": False, "reason": "notify_exception"}

    return JSONResponse({
        "id": record.id,
        "description": text_out,
        "vector_len": len(vector),
        "match_notification": notify_result,
    })


@report_router.post("/embed_text")
async def report_embed_text(
    request: Request,
    text: str = Form(...),
    report_id: int | None = Form(None),
    current_user: User = Depends(_require_auth_user),
    db: Session = Depends(get_db),
):
    """Convert supplied text to a 2048-dim embedding and save to reports.features_dis."""
    try:
        client = _get_client()
        emb_resp = client.embeddings.create(
            model="embedding-3", input=text, dimensions=2048)
        vector = emb_resp.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding error: {e}")

    try:
        if report_id is not None:
            record = db.query(Report).filter(Report.id == report_id).first()
            if not record:
                raise HTTPException(
                    status_code=404, detail=f"report_id={report_id} not found")
            if record.user_id and record.user_id != current_user.id:
                raise HTTPException(
                    status_code=403, detail="No permission to update this report")
            if not record.user_id:
                record.user_id = current_user.id
            record.description = text
            record.features_dis = json.dumps(vector)
        else:
            record = Report(user_id=current_user.id,
                            description=text, features_dis=json.dumps(vector))
            db.add(record)
        db.commit()
        db.refresh(record)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db error: {e}")

    notify_result = None
    try:
        notify_result = _notify_best_request_match(db, record)
    except Exception:
        # Do not block normal API response if mail sending fails.
        notify_result = {"sent": False, "reason": "notify_exception"}

    return JSONResponse({
        "id": record.id,
        "vector_len": len(vector),
        "match_notification": notify_result,
    })
