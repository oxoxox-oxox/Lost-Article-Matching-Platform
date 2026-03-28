"""Fine-ranking (re-ranking) logic for multimodal search.

This module implements the new business logic for fine-ranking using text and image
embeddings from the ItemEmbeddingEngine. The ranking is purely based on multimodal
similarity scores without relying on previous coarse-stage scores.
"""

from __future__ import annotations

from typing import Union, Optional, Any
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import sys
import os
import importlib.util

# Dynamically import ItemEmbeddingEngine (handles hyphenated module name)
def _load_embedding_engine() -> Any:
    """Load ItemEmbeddingEngine from Cross-Modal_Finder.py using importlib."""
    try:
        module_path = os.path.join(
            os.path.dirname(__file__),
            "../../model/Cross-Modal_Finder/Cross-Modal_Finder.py"
        )
        module_path = os.path.normpath(module_path)
        
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Module not found at: {module_path}")
        
        spec = importlib.util.spec_from_file_location(
            "cross_modal_finder_module",
            module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load module spec")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ItemEmbeddingEngine
    except Exception as e:
        raise RuntimeError(
            f"Failed to import ItemEmbeddingEngine from Cross-Modal_Finder.py: {e}"
        )

try:
    ItemEmbeddingEngine = _load_embedding_engine()
except RuntimeError:
    # Allow graceful degradation - the function will fail at runtime if called
    ItemEmbeddingEngine = None


def calculate_fine_ranking_score(
    user_text: str,
    user_image: Union[str, Image.Image],
    item_text: str,
    item_image: Optional[Union[str, Image.Image]],
    engine: Any,
) -> float:
    """Calculate fine-ranking score using multimodal similarity.

    This function computes similarity scores between user input (text + optional image)
    and a database item (text + optional image) using normalized embeddings.
    The final score is the arithmetic mean of computed similarity scores.

    Args:
        user_text: The text input provided by the user (query description).
        user_image: The image input provided by the user (file path or PIL.Image).
        item_text: The text description of the target item in the database.
        item_image: The image of the target item (file path, PIL.Image, or None).
        engine: An initialized ItemEmbeddingEngine instance for encoding.

    Returns:
        A float value in the range [0.0, 1.0] representing the fine-ranking score.
        Ensures the return type is standard Python float for JSON serialization.

    Raises:
        ValueError: If user_text or item_text is empty, or if engine is invalid.
        RuntimeError: If encoding fails for any input.
    """
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string")
    if not isinstance(item_text, str) or not item_text.strip():
        raise ValueError("item_text must be a non-empty string")
    if engine is None:
        raise ValueError("engine must be an initialized ItemEmbeddingEngine instance")

    # ===== Feature Extraction =====
    # Extract user text embedding (2D array -> 1D vector via [0])
    user_text_vec: NDArray[np.float32] = engine.encode_text_batch([user_text])[0]

    # Extract user image embedding (2D array -> 1D vector via [0])
    user_image_vec: NDArray[np.float32] = engine.encode_image_batch([user_image])[0]

    # Extract item text embedding (2D array -> 1D vector via [0])
    item_text_vec: NDArray[np.float32] = engine.encode_text_batch([item_text])[0]

    # Conditionally extract item image embedding only if item_image is provided and valid
    item_image_vec: Optional[NDArray[np.float32]] = None
    has_item_image = item_image is not None and (
        isinstance(item_image, Image.Image) or
        (isinstance(item_image, str) and item_image.strip())
    )
    if has_item_image:
        item_image_vec = engine.encode_image_batch([item_image])[0]

    # ===== Similarity Calculation & Mapping =====
    scores: list[float] = []

    # Score 1: Text-to-Text similarity (NO linear mapping)
    text_text_score: float = float(np.dot(user_text_vec, item_text_vec))
    scores.append(text_text_score)

    # Score 2: Image-to-Text similarity (MUST apply linear mapping)
    image_text_score: float = float(np.dot(user_image_vec, item_text_vec))
    scaled_image_text_score: float = ItemEmbeddingEngine.scale_image_score(image_text_score)
    scores.append(scaled_image_text_score)

    # Score 3: Image-to-Image similarity (NO linear mapping) - ONLY if item_image exists
    if item_image_vec is not None:
        image_image_score: float = float(np.dot(user_image_vec, item_image_vec))
        scores.append(image_image_score)

    # ===== Score Fusion =====
    # Calculate arithmetic mean of all collected scores
    final_score: float = float(np.mean(scores))

    # Ensure return type is standard Python float (not numpy.float32) for JSON compatibility
    return float(final_score)
