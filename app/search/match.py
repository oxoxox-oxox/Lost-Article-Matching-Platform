import json
import os
import re
import numpy as np
from openai import OpenAI
from typing import List, Optional, Dict, Tuple
from urllib import request as urllib_request
from urllib.error import URLError, HTTPError
from sqlalchemy.orm import Session
from service.models import Report
from model.local_refiner import get_local_refine_confidence


REFINE_PROFILE_TEMPLATE = """
Objective: Reconstruct a high-precision physical profile.
Target Item: {text}
Instructions:
1. Identify the core category and specific material.
2. Isolate visual attributes: color, texture, and unique identifiers.
3. Fix the exact spatial coordinates or landmark.
4. Output format: [Category: ...][Color: ...][Material: ...][Features: ...][Location: ...].
Strictly ignore all narrative verbs, emotional tone, and grammar structure.
""".strip()


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "with",
    "my", "your", "item", "lost", "found", "this", "that", "is", "are", "be", "it",
    "location", "time", "db"
}


def _cosine_similarity(a: List[float], b: List[float]) -> float:
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


def match_search(
    db: Session,
    input_text_vector: Optional[List[float]] = None,
    input_image_vector: Optional[List[float]] = None,
    top_n: int = 10,
) -> List[Dict]:
    """Traverse Report rows and compute maximum cosine similarity across:
    - input_text vs report.features_dis
    - input_image vs report.features_img
    - input_text vs report.features_img
    - input_image vs report.features_dis

    Returns top_n entries sorted by similarity desc. Each entry is dict with id and score.
    """
    results = []
    rows = db.query(Report).all()
    for r in rows:
        best = -1.0
        reasons = []
        # load vectors if present
        try:
            db_text = json.loads(r.features_dis) if r.features_dis else None
        except Exception:
            db_text = None
        try:
            db_img = json.loads(r.features_img) if r.features_img else None
        except Exception:
            db_img = None

        # comparisons
        if input_text_vector is not None and db_text is not None:
            s = _cosine_similarity(input_text_vector, db_text)
            if s > best:
                best = s
                reasons = ["text->text"]
        if input_image_vector is not None and db_img is not None:
            s = _cosine_similarity(input_image_vector, db_img)
            if s > best:
                best = s
                reasons = ["image->image"]
        if input_text_vector is not None and db_img is not None:
            s = _cosine_similarity(input_text_vector, db_img)
            if s > best:
                best = s
                reasons = ["text->image"]
        if input_image_vector is not None and db_text is not None:
            s = _cosine_similarity(input_image_vector, db_text)
            if s > best:
                best = s
                reasons = ["image->text"]

        if best >= 0:
            results.append(
                {
                    "id": r.id,
                    "score": best,
                    "reasons": reasons,
                    "description": (r.description or ""),
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def _get_refine_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    resolved_key = api_key or os.getenv(
        "OPENAI_API_KEY") or os.getenv("ZHIPUAI_API_KEY") or os.getenv("zhipuAPI")
    if not resolved_key:
        raise RuntimeError(
            "Missing API key. Set OPENAI_API_KEY or ZHIPUAI_API_KEY.")

    resolved_base_url = base_url or os.getenv(
        "OPENAI_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4/"
    return OpenAI(api_key=resolved_key, base_url=resolved_base_url)


def _normalize_label(text: str) -> str:
    t = (text or "").strip().lower()
    if "yes" in t:
        return "yes"
    if "no" in t:
        return "no"
    return "maybe"


def _profile_from_text(text: str) -> str:
    return REFINE_PROFILE_TEMPLATE.format(text=(text or "").strip())


def _clamp_confidence(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def _confidence_to_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "yes"
    if confidence >= 0.65:
        return "maybe"
    return "no"


def _resolve_keep_min_confidence() -> float:
    try:
        return _clamp_confidence(float(os.getenv("REFINE_KEEP_MIN_CONF", "0.62")))
    except Exception:
        return 0.62


def _tokenize_keywords(text: str) -> set[str]:
    tokens = _TOKEN_RE.findall((text or "").lower())
    return {t for t in tokens if len(t) > 1 and t not in _STOPWORDS}


def _lexical_alignment_factor(query_text: str, candidate_text: str) -> float:
    q_tokens = _tokenize_keywords(query_text)
    c_tokens = _tokenize_keywords(candidate_text)
    if not q_tokens or not c_tokens:
        return 1.0

    overlap = len(q_tokens.intersection(c_tokens)) / max(1, len(q_tokens))
    if overlap >= 0.50:
        return 1.0
    if overlap >= 0.25:
        return 0.85
    if overlap >= 0.10:
        return 0.70
    return 0.45


def _resolve_refine_weights() -> Tuple[float, float]:
    try:
        external_weight = float(os.getenv("REFINE_EXTERNAL_WEIGHT", "0.6"))
    except Exception:
        external_weight = 0.6
    try:
        local_weight = float(os.getenv("REFINE_LOCAL_WEIGHT", "0.4"))
    except Exception:
        local_weight = 0.4

    external_weight = max(0.0, external_weight)
    local_weight = max(0.0, local_weight)
    total = external_weight + local_weight
    if total <= 0:
        return 0.6, 0.4
    return external_weight / total, local_weight / total


def _fuse_refine_confidence(
    external_confidence: Optional[float],
    local_confidence: Optional[float],
) -> Optional[float]:
    ext_w, loc_w = _resolve_refine_weights()

    if external_confidence is not None and local_confidence is not None:
        return _clamp_confidence(ext_w * external_confidence + loc_w * local_confidence)
    if external_confidence is not None:
        return _clamp_confidence(external_confidence)
    if local_confidence is not None:
        return _clamp_confidence(local_confidence)
    return None


def _external_profile_refine(
    query_text: str,
    candidate_text: str,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout_seconds: int = 20,
) -> Tuple[str, str, float]:
    resolved_url = api_url or os.getenv("CUSTOM_REFINE_API_URL")
    if not resolved_url:
        raise RuntimeError("CUSTOM_REFINE_API_URL is not configured")

    resolved_key = api_key or os.getenv("CUSTOM_REFINE_API_KEY") or os.getenv(
        "ZHIPUAI_API_KEY") or os.getenv("zhipuAPI")
    profile_query = _profile_from_text(query_text)
    profile_candidate = _profile_from_text(candidate_text)

    payload = {
        "template": REFINE_PROFILE_TEMPLATE,
        "query_text": query_text,
        "candidate_text": candidate_text,
        "query_profile": profile_query,
        "candidate_profile": profile_candidate,
    }
    if model:
        payload["model"] = model

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
    except (HTTPError, URLError, TimeoutError) as e:
        raise RuntimeError(f"custom refine API call failed: {e}")

    try:
        parsed = json.loads(body)
    except Exception as e:
        raise RuntimeError(f"custom refine API invalid JSON: {e}")

    if not isinstance(parsed, dict):
        raise RuntimeError("custom refine API returned non-object response")

    score_raw = parsed.get("confidence", parsed.get(
        "score", parsed.get("probability", 0.0)))
    try:
        score_val = float(score_raw)
    except Exception:
        score_val = 0.0

    # Allow percentage-like outputs from custom service.
    if score_val > 1.0 and score_val <= 100.0:
        score_val = score_val / 100.0

    confidence = _clamp_confidence(score_val)
    label = _normalize_label(str(parsed.get("label", "")))
    if label not in {"yes", "maybe", "no"}:
        label = _confidence_to_label(confidence)

    reason = str(parsed.get("reason", "")).strip()
    if not reason:
        reason = f"custom_refine_confidence={confidence:.3f}"

    return label, reason, confidence


def _llm_refine_same_item(
    lost_desc: str,
    found_desc: str,
    model: str = "glm-4-flash",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[str, str]:
    prompt = f"""
Task: Compare if these two items are likely the same object.

Rules:
- YES: Core category matches AND key features are consistent or identical
- NO: Irreconcilable conflict
- MAYBE: Category matches, but some features have minor discrepancies or are vague

Note: Allow for subjective descriptions.

Lost Item: "{lost_desc}"
Found Item: "{found_desc}"

Output ONLY "yes", "no", or "maybe".
"""

    client = _get_refine_client(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=16,
    )
    content = (resp.choices[0].message.content or "").strip()
    return _normalize_label(content), content


def match_search_refined(
    db: Session,
    input_text_vector: Optional[List[float]] = None,
    input_image_vector: Optional[List[float]] = None,
    query_description: Optional[str] = None,
    top_n: int = 10,
    coarse_top_k: int = 30,
    min_coarse_score: float = 0.2,
    keep_labels: Tuple[str, ...] = ("yes", "maybe"),
    model: str = "glm-4-flash",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    use_custom_refine: bool = True,
    use_local_refine: bool = True,
    custom_refine_api_url: Optional[str] = None,
    custom_refine_api_key: Optional[str] = None,
    custom_refine_timeout: int = 20,
) -> List[Dict]:
    """Two-stage matching:
    1) Coarse screening by embedding cosine similarity.
    2) Fine screening by LLM yes/no/maybe judgement.
    """
    coarse = match_search(
        db=db,
        input_text_vector=input_text_vector,
        input_image_vector=input_image_vector,
        top_n=coarse_top_k,
    )

    coarse = [c for c in coarse if c.get("score", -1.0) >= min_coarse_score]
    if not query_description:
        return coarse[:top_n]

    label_score = {"yes": 1.0, "maybe": 0.5, "no": 0.0}
    refined: List[Dict] = []

    for item in coarse:
        candidate_desc = (item.get("description") or "").strip()
        if not candidate_desc:
            continue

        custom_confidence: Optional[float] = None
        local_confidence: Optional[float] = None
        external_label: Optional[str] = None
        label: str
        refine_note_parts: List[str] = []
        query_profile = _profile_from_text(query_description)
        candidate_profile = _profile_from_text(candidate_desc)

        if use_custom_refine:
            try:
                label, raw_output, custom_confidence = _external_profile_refine(
                    query_text=query_description,
                    candidate_text=candidate_desc,
                    api_url=custom_refine_api_url,
                    api_key=custom_refine_api_key,
                    model=model,
                    timeout_seconds=custom_refine_timeout,
                )
                external_label = label
                refine_note_parts.append(f"external:{raw_output}")
            except Exception:
                label, raw_output = _llm_refine_same_item(
                    lost_desc=query_description,
                    found_desc=candidate_desc,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                )
                refine_note_parts.append(f"llm_fallback:{raw_output}")
        else:
            label, raw_output = _llm_refine_same_item(
                lost_desc=query_description,
                found_desc=candidate_desc,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
            refine_note_parts.append(f"llm_only:{raw_output}")

        if use_local_refine:
            local_confidence, local_note = get_local_refine_confidence(
                query_profile=query_profile,
                candidate_profile=candidate_profile,
            )
            refine_note_parts.append(local_note)

        fused_confidence = _fuse_refine_confidence(
            external_confidence=custom_confidence,
            local_confidence=local_confidence,
        )
        lexical_factor = _lexical_alignment_factor(query_description, candidate_desc)
        if fused_confidence is not None:
            fused_confidence = _clamp_confidence(fused_confidence * lexical_factor)
            label = _confidence_to_label(fused_confidence)

        min_keep_conf = _resolve_keep_min_confidence()
        if fused_confidence is not None and fused_confidence < min_keep_conf:
            continue

        if label not in keep_labels:
            continue

        coarse_score = float(item.get("score", 0.0))
        refine_score = fused_confidence if fused_confidence is not None else label_score.get(
            label, 0.5)
        final_score = 0.7 * coarse_score + 0.3 * refine_score

        enriched = dict(item)
        enriched["refine_label"] = label
        enriched["refine_output"] = " | ".join(refine_note_parts)
        enriched["lexical_factor"] = lexical_factor
        if custom_confidence is not None:
            enriched["external_refine_confidence"] = custom_confidence
        if local_confidence is not None:
            enriched["local_refine_confidence"] = local_confidence
        if fused_confidence is not None:
            enriched["refine_confidence"] = fused_confidence
            enriched["refine_method"] = "weighted_external_local"
        else:
            enriched["refine_method"] = "llm_label"
        enriched["final_score"] = final_score
        refined.append(enriched)

    refined.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return refined[:top_n]


def run_two_stage_screening(
    db: Session,
    input_text_vector: Optional[List[float]] = None,
    input_image_vector: Optional[List[float]] = None,
    query_description: Optional[str] = None,
    top_n: int = 10,
    coarse_top_k: int = 30,
    min_coarse_score: float = 0.2,
    keep_labels: Tuple[str, ...] = ("yes", "maybe"),
    model: str = "glm-4-flash",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    use_custom_refine: bool = True,
    use_local_refine: bool = True,
    custom_refine_api_url: Optional[str] = None,
    custom_refine_api_key: Optional[str] = None,
    custom_refine_timeout: int = 20,
) -> Dict:
    """Run coarse and fine screening, and return both matches and screening stats."""
    coarse_all = match_search(
        db=db,
        input_text_vector=input_text_vector,
        input_image_vector=input_image_vector,
        top_n=coarse_top_k,
    )
    coarse_pass = [
        item for item in coarse_all
        if float(item.get("score", -1.0)) >= min_coarse_score
    ]

    summary = {
        "coarse_total": len(coarse_all),
        "coarse_pass": len(coarse_pass),
        "min_coarse_score": min_coarse_score,
        "refine_attempted": 0,
        "refine_kept": 0,
        "refine_skipped_empty_desc": 0,
        "label_counts": {"yes": 0, "maybe": 0, "no": 0},
        "refine_method": "weighted_external_local" if (use_custom_refine or use_local_refine) else "llm_label",
    }

    if not query_description:
        final_matches = coarse_pass[:top_n]
        summary["refine_kept"] = len(final_matches)
        return {"summary": summary, "matches": final_matches}

    label_score = {"yes": 1.0, "maybe": 0.5, "no": 0.0}
    refined: List[Dict] = []

    for item in coarse_pass:
        candidate_desc = (item.get("description") or "").strip()
        if not candidate_desc:
            summary["refine_skipped_empty_desc"] += 1
            continue

        summary["refine_attempted"] += 1
        custom_confidence: Optional[float] = None
        local_confidence: Optional[float] = None
        external_label: Optional[str] = None
        label: str
        refine_note_parts: List[str] = []
        query_profile = _profile_from_text(query_description)
        candidate_profile = _profile_from_text(candidate_desc)

        if use_custom_refine:
            try:
                label, raw_output, custom_confidence = _external_profile_refine(
                    query_text=query_description,
                    candidate_text=candidate_desc,
                    api_url=custom_refine_api_url,
                    api_key=custom_refine_api_key,
                    model=model,
                    timeout_seconds=custom_refine_timeout,
                )
                external_label = label
                refine_note_parts.append(f"external:{raw_output}")
            except Exception:
                label, raw_output = _llm_refine_same_item(
                    lost_desc=query_description,
                    found_desc=candidate_desc,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                )
                summary["refine_method"] = "llm_label_fallback"
                refine_note_parts.append(f"llm_fallback:{raw_output}")
        else:
            label, raw_output = _llm_refine_same_item(
                lost_desc=query_description,
                found_desc=candidate_desc,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
            refine_note_parts.append(f"llm_only:{raw_output}")

        if use_local_refine:
            local_confidence, local_note = get_local_refine_confidence(
                query_profile=query_profile,
                candidate_profile=candidate_profile,
            )
            refine_note_parts.append(local_note)

        fused_confidence = _fuse_refine_confidence(
            external_confidence=custom_confidence,
            local_confidence=local_confidence,
        )
        lexical_factor = _lexical_alignment_factor(query_description, candidate_desc)
        if fused_confidence is not None:
            fused_confidence = _clamp_confidence(fused_confidence * lexical_factor)
            label = _confidence_to_label(fused_confidence)

        summary["label_counts"][label] = summary["label_counts"].get(
            label, 0) + 1

        min_keep_conf = _resolve_keep_min_confidence()
        if fused_confidence is not None and fused_confidence < min_keep_conf:
            continue

        if label not in keep_labels:
            continue

        coarse_score = float(item.get("score", 0.0))
        refine_score = fused_confidence if fused_confidence is not None else label_score.get(
            label, 0.5)
        final_score = 0.7 * coarse_score + 0.3 * refine_score

        enriched = dict(item)
        enriched["refine_label"] = label
        enriched["refine_output"] = " | ".join(refine_note_parts)
        enriched["lexical_factor"] = lexical_factor
        if custom_confidence is not None:
            enriched["external_refine_confidence"] = custom_confidence
        if local_confidence is not None:
            enriched["local_refine_confidence"] = local_confidence
        if fused_confidence is not None:
            enriched["refine_confidence"] = fused_confidence
            enriched["refine_method"] = "weighted_external_local"
        else:
            enriched["refine_method"] = "llm_label"
        enriched["final_score"] = final_score
        refined.append(enriched)

    refined.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    final_matches = refined[:top_n]
    summary["refine_kept"] = len(final_matches)
    return {"summary": summary, "matches": final_matches}
