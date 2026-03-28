import os
from typing import Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class _LocalSentenceRefiner:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def confidence(self, query_profile: str, candidate_profile: str) -> float:
        vectors = self.model.encode(
            [query_profile, candidate_profile],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q = np.asarray(vectors[0], dtype=float)
        c = np.asarray(vectors[1], dtype=float)
        cosine = float(np.dot(q, c))
        # Map cosine [-1, 1] to confidence [0, 1].
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


_refiner: Optional[_LocalSentenceRefiner] = None


def get_local_refine_confidence(
    query_profile: str,
    candidate_profile: str,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Optional[float], str]:
    global _refiner

    resolved_model = model_name or os.getenv(
        "LOCAL_REFINE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    resolved_device = device or os.getenv("LOCAL_REFINE_DEVICE", "cpu")

    try:
        if (
            _refiner is None
            or _refiner.model_name != resolved_model
            or _refiner.device != resolved_device
        ):
            _refiner = _LocalSentenceRefiner(
                model_name=resolved_model,
                device=resolved_device,
            )

        score = _refiner.confidence(query_profile, candidate_profile)
        return score, "local_sentence_transformer"
    except Exception as e:
        return None, f"local_refine_error: {e}"
