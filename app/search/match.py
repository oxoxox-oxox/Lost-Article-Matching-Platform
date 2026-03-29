import json
import os
import numpy as np
from openai import OpenAI
from typing import List, Optional, Dict, Tuple, Union
from sqlalchemy.orm import Session
from service.models import Report
from PIL import Image

# Import new fine-ranking function
try:
    from app.search.ranking import calculate_fine_ranking_score, ItemEmbeddingEngine
except ImportError:
    calculate_fine_ranking_score = None
    ItemEmbeddingEngine = None


MIN_COARSE_SCORE_FOR_FINE = 0.6
_CMF_ENGINE = None
_CMF_ENGINE_LOAD_ERROR = None


def _get_cmf_engine():
    global _CMF_ENGINE_LOAD_ERROR
    global _CMF_ENGINE
    if _CMF_ENGINE is not None:
        return _CMF_ENGINE
    if ItemEmbeddingEngine is None:
        _CMF_ENGINE_LOAD_ERROR = "ItemEmbeddingEngine import failed"
        return None
    try:
        _CMF_ENGINE = ItemEmbeddingEngine()
        _CMF_ENGINE_LOAD_ERROR = None
        return _CMF_ENGINE
    except Exception as e:
        _CMF_ENGINE_LOAD_ERROR = str(e)
        return None


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


def _fuse_multimodal_score(
    text_text: float,
    image_image: float,
    text_image: float,
    image_text: float,
    query_has_image: bool,
) -> float:
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

    if query_has_image:
        if has_text_text and has_image_image:
            score = 0.6 * direct_map["text_text"] + \
                0.4 * direct_map["image_image"]
            gap = abs(direct_map["text_text"] - direct_map["image_image"])
            if gap > 0.15:
                score -= min(0.35, (gap - 0.15) * 1.2)
        elif has_text_text:
            score = direct_map["text_text"]
        elif has_image_image:
            score = direct_map["image_image"]
        else:
            score = max(cross_scores)

        if cross_scores:
            score = 0.9 * score + 0.1 * max(cross_scores)
    else:
        # Text-only query: rely much more on query-text vs candidate-image agreement.
        if has_text_text and text_image >= 0:
            score = 0.55 * direct_map["text_text"] + 0.45 * text_image
            gap = abs(direct_map["text_text"] - text_image)
            if gap > 0.12:
                score -= min(0.45, (gap - 0.12) * 1.5)
            if text_image < 0.18:
                score -= min(0.20, (0.18 - text_image) * 1.0)
        elif has_text_text:
            # Candidate has no image vector evidence.
            score = 0.9 * direct_map["text_text"]
        elif text_image >= 0:
            score = text_image
        elif has_image_image:
            score = direct_map["image_image"]
        else:
            score = max(cross_scores)

    return float(max(0.0, min(1.0, score)))


def _apply_self_consistency_penalty(
    score: float,
    query_has_image: bool,
    db_text: Optional[List[float]],
    db_img: Optional[List[float]],
) -> float:
    """Penalize candidates with inconsistent internal text-image features.

    This mainly targets text-only queries where a wrong image can otherwise hide
    behind a strong text-text match.
    """
    if score < 0:
        return score

    # When query has an image, cross-check already has enough visual evidence.
    if query_has_image:
        return score

    if db_text is None or db_img is None:
        return score

    self_consistency = _cosine_similarity(db_text, db_img)
    if self_consistency < 0:
        return score

    # Stronger penalty for clear text-image conflicts in the same candidate.
    if self_consistency < 0.28:
        penalty = min(0.55, (0.28 - self_consistency) * 1.8)
        score -= penalty

    return float(max(0.0, min(1.0, score)))


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

        text_text = _cosine_similarity(input_text_vector, db_text) if (
            input_text_vector is not None and db_text is not None) else -1.0
        image_image = _cosine_similarity(input_image_vector, db_img) if (
            input_image_vector is not None and db_img is not None) else -1.0
        text_image = _cosine_similarity(input_text_vector, db_img) if (
            input_text_vector is not None and db_img is not None) else -1.0
        image_text = _cosine_similarity(input_image_vector, db_text) if (
            input_image_vector is not None and db_text is not None) else -1.0

        best = _fuse_multimodal_score(
            text_text=text_text,
            image_image=image_image,
            text_image=text_image,
            image_text=image_text,
            query_has_image=(input_image_vector is not None),
        )

        best = _apply_self_consistency_penalty(
            score=best,
            query_has_image=(input_image_vector is not None),
            db_text=db_text,
            db_img=db_img,
        )

        if text_text >= 0:
            reasons.append("text->text")
        if image_image >= 0:
            reasons.append("image->image")
        if text_image >= 0:
            reasons.append("text->image")
        if image_text >= 0:
            reasons.append("image->text")

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
        "OPENAI_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
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
    min_coarse_score: float = MIN_COARSE_SCORE_FOR_FINE,
    keep_labels: Tuple[str, ...] = ("yes", "maybe"),
    model: str = "glm-4-flash",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
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

    effective_min_coarse_score = max(MIN_COARSE_SCORE_FOR_FINE, float(min_coarse_score))
    coarse = [
        c for c in coarse
        if float(c.get("score", -1.0)) >= effective_min_coarse_score
    ]
    if not query_description:
        return coarse[:top_n]

    label_score = {"yes": 1.0, "maybe": 0.5, "no": 0.0}
    refined: List[Dict] = []

    for item in coarse:
        candidate_desc = (item.get("description") or "").strip()
        if not candidate_desc:
            continue

        try:
            label, raw_output = _llm_refine_same_item(
                lost_desc=query_description,
                found_desc=candidate_desc,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
        except Exception:
            label, raw_output = "maybe", "maybe"

        if label not in keep_labels:
            continue

        coarse_score = float(item.get("score", 0.0))
        llm_score = label_score.get(label, 0.5)
        if input_image_vector is None:
            # Text-only query: increase LLM contribution.
            final_score = 0.7 * coarse_score + 0.3 * llm_score
        else:
            final_score = 0.85 * coarse_score + 0.15 * llm_score

        enriched = dict(item)
        enriched["refine_label"] = label
        enriched["refine_output"] = raw_output
        enriched["final_score"] = final_score
        refined.append(enriched)

    refined.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return refined[:top_n]


def run_two_stage_screening(
    db: Session,
    input_text_vector: Optional[List[float]] = None,
    input_image_vector: Optional[List[float]] = None,
    query_description: Optional[str] = None,
    user_image: Optional[Union[str, Image.Image]] = None,
    top_n: int = 10,
    coarse_top_k: int = 30,
    min_coarse_score: float = MIN_COARSE_SCORE_FOR_FINE,
    keep_labels: Tuple[str, ...] = ("yes", "maybe"),
    model: str = "glm-4-flash",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict:
    """Run coarse and fine screening, and return both matches and screening stats."""
    coarse_all = match_search(
        db=db,
        input_text_vector=input_text_vector,
        input_image_vector=input_image_vector,
        top_n=coarse_top_k,
    )
    effective_min_coarse_score = max(MIN_COARSE_SCORE_FOR_FINE, float(min_coarse_score))

    coarse_pass = [
        item for item in coarse_all
        if float(item.get("score", -1.0)) >= effective_min_coarse_score
    ]

    cmf_engine = _get_cmf_engine() if calculate_fine_ranking_score is not None else None

    summary = {
        "coarse_total": len(coarse_all),
        "coarse_pass": len(coarse_pass),
        "min_coarse_score": effective_min_coarse_score,
        "refine_attempted": 0,
        "refine_kept": 0,
        "refine_skipped_empty_desc": 0,
        "cmf_engine_available": bool(cmf_engine is not None),
        "cmf_scored_count": 0,
        "cmf_failed_count": 0,
        "label_counts": {"yes": 0, "maybe": 0, "no": 0},
    }

    if cmf_engine is None and _CMF_ENGINE_LOAD_ERROR:
        summary["cmf_engine_error"] = _CMF_ENGINE_LOAD_ERROR

    if cmf_engine is None:
        raise RuntimeError(
            "CMF engine unavailable for fine screening; refusing to fallback to coarse score. "
            "Check cmf_engine_error in summary or server logs."
        )

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
        try:
            label, raw_output = _llm_refine_same_item(
                lost_desc=query_description,
                found_desc=candidate_desc,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
        except Exception:
            label, raw_output = "maybe", "maybe"

        summary["label_counts"][label] = summary["label_counts"].get(
            label, 0) + 1

        if label not in keep_labels:
            continue

        # Fine-stage confidence now uses CMF score as the model confidence term.
        cmf_score: Optional[float] = None
        if cmf_engine is not None:
            try:
                item_id = item.get("id")
                report = db.query(Report).filter(Report.id == item_id).first()
                item_image = report.image if report else None
                if item_image:
                    candidate_image_path = os.path.join("static", item_image)
                    item_image = candidate_image_path if os.path.exists(candidate_image_path) else None

                cmf_score = float(calculate_fine_ranking_score(
                    user_text=query_description,
                    user_image=user_image,
                    item_text=candidate_desc,
                    item_image=item_image,
                    engine=cmf_engine,
                ))
            except Exception:
                cmf_score = None

        if cmf_score is None:
            summary["cmf_failed_count"] += 1
            continue

        summary["cmf_scored_count"] += 1

        llm_score = label_score.get(label, 0.5)
        if input_image_vector is None:
            final_score = 0.7 * cmf_score + 0.3 * llm_score
        else:
            final_score = 0.85 * cmf_score + 0.15 * llm_score

        enriched = dict(item)
        enriched["refine_label"] = label
        enriched["refine_output"] = raw_output
        enriched["coarse_score"] = float(item.get("score", 0.0))
        enriched["cmf_score"] = float(cmf_score)
        enriched["cmf_used"] = True
        enriched["final_score"] = final_score
        refined.append(enriched)

    refined.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    final_matches = refined[:top_n]
    summary["refine_kept"] = len(final_matches)
    return {"summary": summary, "matches": final_matches}


def run_two_stage_screening_multimodal(
    db: Session,
    user_text: str,
    user_image: Optional[Union[str, Image.Image]] = None,
    top_n: int = 10,
    coarse_top_k: int = 30,
    min_coarse_score: float = MIN_COARSE_SCORE_FOR_FINE,
    input_text_vector: Optional[List[float]] = None,
    input_image_vector: Optional[List[float]] = None,
    engine=None,
) -> Dict:
    """Run coarse and fine screening using multimodal embeddings.

    This is the new screening pipeline that:
    1) Performs coarse screening via vector similarity.
    2) Performs fine screening using calculate_fine_ranking_score (rejects LLM calls).

    Args:
        db: Database session.
        user_text: User's original query text.
        user_image: User's optional image (file path or PIL.Image).
        top_n: Number of final results to return.
        coarse_top_k: Number of candidates to keep after coarse screening.
        min_coarse_score: Minimum coarse score threshold.
        input_text_vector: Pre-computed user text embedding (optional).
        input_image_vector: Pre-computed user image embedding (optional).
        engine: ItemEmbeddingEngine instance for multimodal ranking.

    Returns:
        Dictionary with 'summary' and 'matches' keys.
    """

    if calculate_fine_ranking_score is None:
        raise RuntimeError(
            "calculate_fine_ranking_score is not available. "
            "Check ranking.py import."
        )

    if engine is None:
        raise ValueError("engine (ItemEmbeddingEngine instance) is required")

    # Step 1: Coarse screening via vector similarity
    coarse_all = match_search(
        db=db,
        input_text_vector=input_text_vector,
        input_image_vector=input_image_vector,
        top_n=coarse_top_k,
    )
    effective_min_coarse_score = max(MIN_COARSE_SCORE_FOR_FINE, float(min_coarse_score))

    coarse_pass = [
        item for item in coarse_all
        if float(item.get("score", -1.0)) >= effective_min_coarse_score
    ]

    summary = {
        "coarse_total": len(coarse_all),
        "coarse_pass": len(coarse_pass),
        "min_coarse_score": effective_min_coarse_score,
        "refine_attempted": 0,
        "refine_kept": 0,
        "fine_ranking_scores": {},
    }

    refined: List[Dict] = []

    # Step 2: Fine screening via multimodal score
    for item in coarse_pass:
        item_id = item.get("id")
        item_text = (item.get("description") or "").strip()

        if not item_text:
            continue

        summary["refine_attempted"] += 1

        try:
            # Fetch the Report record to get image path
            report = db.query(Report).filter(Report.id == item_id).first()
            item_image = report.image if report else None

            # Build full path to image if it exists
            if item_image:
                item_image_path = os.path.join("static", item_image)
                if not os.path.exists(item_image_path):
                    item_image = None
            else:
                item_image = None

            # Calculate fine-ranking score
            fine_score = calculate_fine_ranking_score(
                user_text=user_text,
                user_image=user_image,
                item_text=item_text,
                item_image=item_image,
                engine=engine,
            )

            # Determine label based on score threshold
            if fine_score >= 0.75:
                label = "yes"
            elif fine_score >= 0.45:
                label = "maybe"
            else:
                label = "no"

            summary["fine_ranking_scores"][str(
                item_id)] = round(float(fine_score), 4)

            # Always keep items (no early filtering by label)
            enriched = dict(item)
            enriched["refine_label"] = label
            enriched["refine_output"] = json.dumps({
                "method": "multimodal-fine-ranking",
                "fine_score": round(float(fine_score), 4),
            })
            enriched["final_score"] = float(fine_score)
            refined.append(enriched)

        except Exception as e:
            # Log error and skip this item
            summary["fine_ranking_scores"][str(item_id)] = None
            continue

    # Sort by fine-ranking score (final_score)
    refined.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    final_matches = refined[:top_n]
    summary["refine_kept"] = len(final_matches)

    return {"summary": summary, "matches": final_matches}
