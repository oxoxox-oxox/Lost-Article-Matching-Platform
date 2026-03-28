import json
import os
import numpy as np
from openai import OpenAI
from typing import List, Optional, Dict, Tuple
from sqlalchemy.orm import Session
from service.models import Report


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
    min_coarse_score: float = 0.2,
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

    coarse = [c for c in coarse if c.get("score", -1.0) >= min_coarse_score]
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
        final_score = 0.7 * coarse_score + 0.3 * llm_score

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
    top_n: int = 10,
    coarse_top_k: int = 30,
    min_coarse_score: float = 0.2,
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

        coarse_score = float(item.get("score", 0.0))
        llm_score = label_score.get(label, 0.5)
        final_score = 0.7 * coarse_score + 0.3 * llm_score

        enriched = dict(item)
        enriched["refine_label"] = label
        enriched["refine_output"] = raw_output
        enriched["final_score"] = final_score
        refined.append(enriched)

    refined.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    final_matches = refined[:top_n]
    summary["refine_kept"] = len(final_matches)
    return {"summary": summary, "matches": final_matches}
