
from fastapi import APIRouter, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import os
import base64
import json
import re
from io import BytesIO
from PIL import Image
from zhipuai import ZhipuAI
from service.database import get_db
from sqlalchemy.orm import Session
from service.models import Request as RequestModel, User
import service.security as security
from app.search.match import run_two_stage_screening, run_two_stage_screening_multimodal
from pydantic import BaseModel
from typing import Any, Optional
import importlib.util

# Import ItemEmbeddingEngine for multimodal ranking (dynamic import for hyphenated module name)
def _load_embedding_engine_for_routes() -> tuple[bool, Any, Any]:
    """Load ItemEmbeddingEngine and EngineConfig from Cross-Modal_Finder.py."""
    try:
        module_path = os.path.join(
            os.path.dirname(__file__),
            "../../model/Cross-Modal_Finder/Cross-Modal_Finder.py"
        )
        module_path = os.path.normpath(module_path)
        
        if not os.path.exists(module_path):
            return False, None, None
        
        spec = importlib.util.spec_from_file_location(
            "cross_modal_finder_routes",
            module_path
        )
        if spec is None or spec.loader is None:
            return False, None, None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, module.ItemEmbeddingEngine, module.EngineConfig
    except Exception:
        return False, None, None

EMBEDDING_ENGINE_AVAILABLE, ItemEmbeddingEngine, EngineConfig = _load_embedding_engine_for_routes()

# Global cache for ItemEmbeddingEngine (lazy initialization)
_embedding_engine: Optional[Any] = None

def _get_embedding_engine() -> Any:
    """Get or create the ItemEmbeddingEngine instance (lazy initialization)."""
    global _embedding_engine
    if _embedding_engine is None and EMBEDDING_ENGINE_AVAILABLE and EngineConfig is not None:
        try:
            config = EngineConfig(device="cpu")  # Use CPU for web service
            _embedding_engine = ItemEmbeddingEngine(config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ItemEmbeddingEngine: {e}")
    return _embedding_engine

search_router = APIRouter(
    prefix="/search",
    tags=["Search"]
)
templates = Jinja2Templates(directory="templates")


class SearchChatParseResult(BaseModel):
    normalized_query: str = ""
    should_search: bool = False
    follow_up_question: str = ""
    user_reply: str = ""


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


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort parse for JSON outputs wrapped in markdown or extra text."""
    if not text:
        return None
    content = text.strip()

    # Remove fenced code blocks if present.
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


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_markdown_text(text: str) -> str:
    """Convert common escaped sequences to real control chars for cleaner markdown rendering."""
    normalized = _to_str(text)
    normalized = normalized.replace("\\r\\n", "\n")
    normalized = normalized.replace("\\n", "\n")
    normalized = normalized.replace("\\t", "\t")
    return normalized


def _build_search_reply_from_screening(parsed_query: str, screening: dict[str, Any]) -> str:
    matches = screening.get("matches") or []
    summary = screening.get("summary") or {}
    kept = int(summary.get("refine_kept", len(matches)) or 0)
    coarse_pass = int(summary.get("coarse_pass", 0) or 0)

    if not matches:
        return (
            "## Search Result\n"
            f"I understood your request: **{parsed_query}**.\n\n"
            "No high-confidence match was found yet.\n\n"
            "### Next Step Suggestions\n"
            "- Add clearer identifiers: brand, color, scratches, stickers, serial pattern\n"
            "- Include exact time and area where the item was lost\n"
            "- Upload another clearer image if available"
        )

    top = matches[:3]
    lines = [
        "## Search Result",
        f"I found **{kept}** potential match(es) for: **{parsed_query}**",
        f"Initial screening passed **{coarse_pass}** candidate(s).",
        "",
        "### Top Candidates",
    ]
    for idx, item in enumerate(top, start=1):
        score = float(item.get("final_score", item.get("score", 0.0)) or 0.0)
        label = _to_str(item.get("refine_label")) or "maybe"
        desc = _to_str(item.get("description"))
        if len(desc) > 120:
            desc = desc[:120] + "..."
        lines.append(
            f"{idx}. **Candidate #{item.get('id')}** | confidence: `{score:.2f}` | label: `{label}`  \n{desc}"
        )
    lines.append(
        "\n### Continue Refining\n"
        "If you want, send more details or another photo, and I will narrow the result further."
    )
    return "\n".join(lines)


def _compose_markdown_reply_with_llm(
    client: ZhipuAI,
    parsed_query: str,
    screening: dict[str, Any] | None,
    should_search: bool,
    guidance_text: str,
) -> str | None:
    """Generate user-facing markdown reply with strict formatting constraints."""
    prompt = {
        "parsed_query": parsed_query,
        "should_search": should_search,
        "guidance_text": guidance_text,
        "screening": screening,
    }

    system_text = (
        "You are a lost-and-found search assistant. "
        "Reply in clean GitHub-flavored Markdown only. "
        "Use concise sections and short bullet lists when useful. "
        "Output requirements: "
        "1) Start with heading '## Search Assistant Reply'; "
        "2) If there are matches, include heading '### Top Matches' and a numbered list; "
        "3) If no matches, include heading '### Next Steps' with 3 actionable bullets; "
        "4) Keep tone supportive and practical; "
        "5) Do not output JSON or code fences."
    )

    try:
        resp = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False),
                },
            ],
            temperature=0.3,
            top_p=0.7,
        )
        text = resp.choices[0].message.content
        if isinstance(text, list):
            text = "\n".join(
                [block.get("text", "")
                 for block in text if isinstance(block, dict)]
            )
        text = _normalize_markdown_text(_to_str(text))
        return text or None
    except Exception:
        return None


def _normalize_history(history_raw: str | None) -> list[dict[str, str]]:
    if not history_raw:
        return []
    try:
        data = json.loads(history_raw)
    except Exception:
        return []
    if not isinstance(data, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in data[-8:]:
        if not isinstance(item, dict):
            continue
        role = _to_str(item.get("role")).lower()
        content = _to_str(item.get("content"))
        if role in {"user", "assistant"} and content:
            normalized.append({"role": role, "content": content})
    return normalized


@search_router.post("/chat")
async def search_chat(
    request: Request,
    message: str = Form(...),
    history: str | None = Form(None),
    image: UploadFile | None = File(None),
    top_n: int = Form(8),
    coarse_top_k: int = Form(30),
    min_coarse_score: float = Form(0.2),
    current_user: User = Depends(_require_auth_user),
    db: Session = Depends(get_db),
):
    """Chat-style search endpoint with optional image input and user-friendly responses."""
    user_message = (message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message is required")

    try:
        client = _get_client()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    image_b64 = None
    if image:
        try:
            body = await image.read()
            image_b64 = _image_to_base64(body)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid image: {e}")

    history_messages = _normalize_history(history)

    parse_system_prompt = (
        "You are an assistant for a lost-and-found search platform. "
        "Understand the user's latest request and output ONLY JSON with keys: "
        "normalized_query (string), should_search (boolean), follow_up_question (string), user_reply (string). "
        "Use should_search=true only when enough concrete item details exist. "
        "If details are insufficient, set should_search=false and ask one concise follow-up question in follow_up_question. "
        "Both user_reply and follow_up_question must be markdown-friendly plain text (short headings and bullets allowed)."
    )

    parse_user_payload = {
        "latest_message": user_message,
        "history": history_messages,
        "has_image": bool(image_b64),
    }

    parse_content: list[dict[str, Any]] = [
        {"type": "text", "text": json.dumps(
            parse_user_payload, ensure_ascii=False)}
    ]
    if image_b64:
        parse_content.append(
            {"type": "image_url", "image_url": {"url": image_b64}})

    try:
        parse_resp = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[
                {"role": "system", "content": parse_system_prompt},
                {"role": "user", "content": parse_content},
            ],
            temperature=0.1,
            top_p=0.7,
        )
        parse_raw = parse_resp.choices[0].message.content
        if isinstance(parse_raw, list):
            parse_raw = "\n".join(
                [block.get("text", "")
                 for block in parse_raw if isinstance(block, dict)]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"parse error: {e}")

    parsed_obj = _extract_json_object(_to_str(parse_raw)) or {}
    parsed = SearchChatParseResult(
        normalized_query=_to_str(parsed_obj.get(
            "normalized_query")) or user_message,
        should_search=bool(parsed_obj.get("should_search", False)),
        follow_up_question=_to_str(parsed_obj.get("follow_up_question")),
        user_reply=_to_str(parsed_obj.get("user_reply")),
    )

    if not parsed.should_search:
        friendly_reply = parsed.user_reply or "Thanks. I need one more detail before searching."
        if parsed.follow_up_question:
            friendly_reply = f"{friendly_reply}\n\n{parsed.follow_up_question}"
        friendly_reply = _normalize_markdown_text(friendly_reply)
        markdown_reply = _compose_markdown_reply_with_llm(
            client=client,
            parsed_query=parsed.normalized_query,
            screening=None,
            should_search=False,
            guidance_text=friendly_reply,
        )
        return JSONResponse(
            {
                "assistant_reply": markdown_reply or friendly_reply,
                "parsed": parsed.model_dump(),
                "screening": None,
            }
        )

    try:
        text_emb_resp = client.embeddings.create(
            model="embedding-3", input=parsed.normalized_query, dimensions=2048
        )
        text_vector = text_emb_resp.data[0].embedding
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"text embedding error: {e}")

    image_vector = None
    image_filename = image.filename if image else None
    if image_b64:
        try:
            vision_prompt = (
                "Summarize the visible lost-item information into one concise retrieval text "
                "with category, color, material, key features, and location clues."
            )
            vision_resp = client.chat.completions.create(
                model="glm-4v-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",
                                "text": f"User query: {parsed.normalized_query}"},
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": image_b64}},
                        ],
                    }
                ],
                temperature=0.0,
                top_p=0.7,
            )
            image_prompt_text = vision_resp.choices[0].message.content
            if isinstance(image_prompt_text, list):
                image_prompt_text = "\n".join(
                    [block.get("text", "")
                     for block in image_prompt_text if isinstance(block, dict)]
                )
            image_prompt_text = _to_str(
                image_prompt_text) or parsed.normalized_query

            image_emb_resp = client.embeddings.create(
                model="embedding-3", input=image_prompt_text, dimensions=2048
            )
            image_vector = image_emb_resp.data[0].embedding
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"image embedding error: {e}")

    try:
        request_record = RequestModel(
            user_id=current_user.id,
            image=image_filename,
            description=parsed.normalized_query,
            features_dis=json.dumps(text_vector),
            features_img=(json.dumps(image_vector)
                          if image_vector is not None else None),
        )
        db.add(request_record)
        db.commit()
        db.refresh(request_record)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"db error: {e}")

    try:
        screening = None
        
        # Try to use multimodal fine-ranking if engine is available
        if EMBEDDING_ENGINE_AVAILABLE:
            try:
                engine = _get_embedding_engine()
                
                # Prepare image for multimodal ranking
                user_image = None
                if image_b64:
                    try:
                        # Convert base64 to PIL.Image
                        image_data = base64.b64decode(image_b64.split(",")[-1])
                        user_image = Image.open(BytesIO(image_data))
                    except Exception:
                        user_image = None
                
                screening = run_two_stage_screening_multimodal(
                    db=db,
                    user_text=parsed.normalized_query,
                    user_image=user_image,
                    top_n=max(1, min(top_n, 10)),
                    coarse_top_k=max(10, min(coarse_top_k, 100)),
                    min_coarse_score=min_coarse_score,
                    input_text_vector=text_vector,
                    input_image_vector=image_vector,
                    engine=engine,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                # Fallback to old screening if multimodal fails
                screening = None
        
        # Fallback to old LLM-based screening
        if screening is None:
            screening = run_two_stage_screening(
                db=db,
                input_text_vector=text_vector,
                input_image_vector=image_vector,
                query_description=parsed.normalized_query,
                top_n=max(1, min(top_n, 10)),
                coarse_top_k=max(10, min(coarse_top_k, 100)),
                min_coarse_score=min_coarse_score,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"screening error: {e}")

    fallback_reply = _normalize_markdown_text(
        _build_search_reply_from_screening(parsed.normalized_query, screening)
    )
    assistant_reply = _compose_markdown_reply_with_llm(
        client=client,
        parsed_query=parsed.normalized_query,
        screening=screening,
        should_search=True,
        guidance_text=fallback_reply,
    ) or fallback_reply

    return JSONResponse(
        {
            "assistant_reply": assistant_reply,
            "parsed": parsed.model_dump(),
            "screening": screening,
            "request_id": request_record.id,
        }
    )


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
        screening = None
        
        # Try to use multimodal fine-ranking if engine is available
        if EMBEDDING_ENGINE_AVAILABLE:
            try:
                engine = _get_embedding_engine()
                
                # Prepare image for multimodal ranking
                user_image = None
                if image:
                    try:
                        # Convert uploaded image to PIL.Image
                        body = await image.read()
                        user_image = Image.open(BytesIO(body))
                    except Exception:
                        user_image = None
                
                screening = run_two_stage_screening_multimodal(
                    db=db,
                    user_text=text,
                    user_image=user_image,
                    top_n=top_n,
                    coarse_top_k=coarse_top_k,
                    min_coarse_score=min_coarse_score,
                    input_text_vector=text_vector,
                    input_image_vector=image_vector,
                    engine=engine,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                # Fallback to old screening if multimodal fails
                screening = None
        
        # Fallback to old LLM-based screening
        if screening is None:
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

    return JSONResponse({
        "query": {
            "has_image": bool(image),
            "text_len": len(text),
            "image_prompt_text": image_prompt_text,
        },
        "screening": screening,
    })
