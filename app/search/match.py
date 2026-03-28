import json
import math
import numpy as np
from typing import List, Optional, Dict
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
            results.append({"id": r.id, "score": best, "reasons": reasons})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]
