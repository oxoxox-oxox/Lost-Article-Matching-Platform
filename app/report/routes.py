from fastapi import APIRouter, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import os
import json
from service.database import get_db
from sqlalchemy.orm import Session
from service.models import Report, User, Request as RequestModel
import service.security as security
from service.email import send_match_notification_email
from service.vision import (
    parse_vector as _parse_vector,
    cosine_similarity as _cosine_similarity,
    extract_item_features,
    generate_embedding,
)

report_router = APIRouter(
    prefix="/report",
    tags=["Report"]
)
templates = Jinja2Templates(directory="templates")


def _best_similarity(report_row: Report, request_row: RequestModel) -> float:
    r_text = _parse_vector(report_row.features_dis)
    r_img = _parse_vector(report_row.features_img)
    q_text = _parse_vector(request_row.features_dis)
    q_img = _parse_vector(request_row.features_img)

    text_text = _cosine_similarity(r_text, q_text)
    image_image = _cosine_similarity(r_img, q_img)
    text_image = _cosine_similarity(r_text, q_img)
    image_text = _cosine_similarity(r_img, q_text)

    direct_scores = []
    cross_scores = []

    if text_text >= 0:
        direct_scores.append(("text_text", text_text))
    if image_image >= 0:
        direct_scores.append(("image_image", image_image))
    if text_image >= 0:
        cross_scores.append(text_image)
    if image_text >= 0:
        cross_scores.append(image_text)

    if not direct_scores and not cross_scores:
        return -1.0

    direct_map = {k: v for k, v in direct_scores}
    has_text_text = "text_text" in direct_map
    has_image_image = "image_image" in direct_map

    # Prefer direct-modality agreement: text-text is strong, image-image must also matter.
    if has_text_text and has_image_image:
        score = 0.6 * direct_map["text_text"] + 0.4 * direct_map["image_image"]

        # Penalize contradictions between text and image channels.
        gap = abs(direct_map["text_text"] - direct_map["image_image"])
        if gap > 0.15:
            score -= min(0.35, (gap - 0.15) * 1.2)
    elif has_text_text:
        score = direct_map["text_text"]
    elif has_image_image:
        score = direct_map["image_image"]
    else:
        # Fallback when only cross-modality comparisons are available.
        score = max(cross_scores)

    # Cross-modality is auxiliary evidence and should not dominate the final score.
    if cross_scores:
        score = 0.9 * score + 0.1 * max(cross_scores)

    return float(max(0.0, min(1.0, score)))


def _normalize_text_for_compare(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().lower().split())


def _get_notify_threshold() -> float:
    try:
        return float(os.getenv("MATCH_NOTIFY_THRESHOLD", "0.70"))
    except Exception:
        return 0.70


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
        report_id=report_row.id,
    )

    # Mark as processed to avoid repeated notifications for the same report.
    if sent:
        if report_row.user_id:
            reporter = db.query(User).filter(
                User.id == report_row.user_id).first()
            if reporter:
                reporter.point = (reporter.point or 0) + 2
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


# Centralized authentication dependency alias
_require_auth_user = security.get_current_user


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
    image_bytes = None
    if image:
        image_bytes = await image.read()

    try:
        text_out, vector, image_b64 = extract_item_features(
            image_bytes=image_bytes,
            user_text=description,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"multimodal feature extraction error: {e}"
        )

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
            record.image_data = image_b64 if image_b64 else record.image_data
            record.features_img = json.dumps(vector)
            if not record.description:
                record.description = description or text_out
        else:
            record = Report(
                user_id=current_user.id,
                image=(image.filename if image else None),
                image_data=image_b64,
                description=(description or text_out),
                features_img=json.dumps(vector),
            )
            db.add(record)
            current_user.point = (current_user.point or 0) + 1
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
        vector = generate_embedding(text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
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
            current_user.point = (current_user.point or 0) + 1
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
