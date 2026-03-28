"""Production-ready semantic embedding engine for Lost & Found descriptions.

This module provides a configurable `ItemEmbeddingEngine` built on
`sentence-transformers/all-MiniLM-L6-v2` and optimized for safe, batched
inference with normalized float32 vectors suitable for MySQL vector storage.
"""
# Copyright 2024 Google LLC
from __future__ import annotations

# Standard Library Imports
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

# Third-Party Imports
import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

# Configuration Data Class
@dataclass
class EngineConfig:
    """Configuration options for `ItemEmbeddingEngine`.

    Args:
        model_name: Name of the sentence-transformers model to load.
        batch_size: Number of texts to process per batch.
        max_seq_length: Maximum tokenized sequence length used by the model.
        device: Device preference for inference ("cpu", "cuda", or "mps").
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    max_seq_length: int = 256
    device: str = "cuda"  # Default to CUDA if available, otherwise fallback to CPU

# Embedding Engine Implementation
class ItemEmbeddingEngine:
    """Stateful embedding engine for Lost & Found item descriptions.

    The model is loaded once per engine instance and reused for all calls.
    Encoded vectors are normalized and returned as `np.float32` to match
    MySQL vector database compatibility requirements.
    """

    # Initialization and Model Loading
    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        """Initialize the embedding engine and warm up the model.

        Args:
            config: Optional engine configuration. Defaults to `EngineConfig()`.

        Raises:
            RuntimeError: If the model cannot be loaded or warm-up fails.
            ValueError: If config values are invalid.
        """

        # Set up logging
        self.config = config or EngineConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configure logging to output to console with a standard format
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Validate configuration values before proceeding
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.config.max_seq_length <= 0:
            raise ValueError("max_seq_length must be a positive integer")

        # Resolve and validate the requested device
        selected_device = self._resolve_device(self.config.device)
        self.config.device = selected_device

        # set up the model and warm it up with a dummy input to ensure it's ready for inference
        load_start = time.perf_counter()
        try:
            self.logger.info(
                "Loading model '%s' on device '%s'...",
                self.config.model_name,
                self.config.device,
            )
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
            self.model.max_seq_length = self.config.max_seq_length
            self.dimension = self.model.get_sentence_embedding_dimension()
            load_elapsed = time.perf_counter() - load_start
            self.logger.info(
                "Model loaded in %.2f seconds | dimension=%d | max_seq_length=%d",
                load_elapsed,
                self.dimension,
                self.model.max_seq_length,
            )

            warmup_start = time.perf_counter()
            _ = self.model.encode(
                ["warmup_text"],
                batch_size=1,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            warmup_elapsed = time.perf_counter() - warmup_start
            self.logger.info("Warm-up complete in %.2f seconds", warmup_elapsed)
        except Exception as exc:
            self.logger.exception("Failed to initialize embedding engine: %s", exc)
            raise RuntimeError(
                f"Unable to initialize model '{self.config.model_name}'."
            ) from exc
    
    # Device Resolution and Validation
    def _resolve_device(self, requested_device: str) -> str:
        """Resolve and validate an inference device.

        Args:
            requested_device: Desired device string.

        Returns:
            A valid, available device string.
        """

        # Normalize input and check availability
        normalized = requested_device.strip().lower()
        if normalized == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            self.logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"

        # Check for MPS (Apple Silicon) support if requested
        if normalized == "mps":
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if has_mps:
                return "mps"
            self.logger.warning("MPS requested but not available. Falling back to CPU.")
            return "cpu"

        # For any other device string, log a warning and default to CPU
        if normalized != "cpu":
            self.logger.warning(
                "Unsupported device '%s'. Falling back to CPU.", requested_device
            )
        return "cpu"
    
    # Text Preprocessing and Batch Encoding
    def _preprocess(self, text: str) -> str:
        """Normalize raw text before embedding.

        Args:
            text: Raw input string.

        Returns:
            Cleaned text with lowercasing and collapsed whitespace.
        """

        return " ".join(text.lower().strip().split())

    # Main Encoding Method with Batching and Error Handling
    def encode_batch(self, texts: List[str]) -> NDArray[np.float32]:
        """Encode a list of texts into normalized embedding vectors.

        Args:
            texts: Input descriptions to encode.

        Returns:
            A float32 numpy array with shape `(len(texts), self.dimension)`.

        Raises:
            ValueError: If any element in `texts` is not a string.
            RuntimeError: If an out-of-memory condition occurs during encoding.
        """

        if not texts:
            self.logger.warning("Received empty text list; returning empty embedding array.")
            return np.empty((0, self.dimension), dtype=np.float32)

        # Validate that all inputs are strings before processing
        for idx, item in enumerate(texts):
            if not isinstance(item, str):
                raise ValueError(
                    f"All elements must be strings. Invalid type at index {idx}: "
                    f"{type(item).__name__}"
                )

        # Preprocess texts and encode in batches, with robust error handling for OOM conditions
        cleaned_texts = [self._preprocess(text) for text in texts]
        total = len(cleaned_texts)
        self.logger.info(
            "Encoding %d texts with batch_size=%d on %s",
            total,
            self.config.batch_size,
            self.config.device,
        )

        # Use a list to collect batch embeddings and stack at the end to avoid large intermediate memory usage
        start_time = time.perf_counter()
        chunks: List[NDArray[np.float32]] = []

        try:
            for start in range(0, total, self.config.batch_size):
                end = min(start + self.config.batch_size, total)
                self.logger.info("Processing chunk %d:%d of %d", start, end, total)
                chunk_embeddings = self.model.encode(
                    cleaned_texts[start:end],
                    batch_size=self.config.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                chunks.append(np.asarray(chunk_embeddings, dtype=np.float32))
        except torch.OutOfMemoryError as exc:
            self.logger.exception("PyTorch CUDA out-of-memory during encoding: %s", exc)
            raise RuntimeError(
                "Out-of-memory while encoding texts. Reduce batch_size or use CPU."
            ) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self.logger.exception("Out-of-memory during encoding: %s", exc)
                raise RuntimeError(
                    "Out-of-memory while encoding texts. Reduce batch_size or input size."
                ) from exc
            raise

        embeddings = np.vstack(chunks).astype(np.float32, copy=False)
        elapsed = time.perf_counter() - start_time
        self.logger.info(
            "Encoding complete in %.2f seconds | output_shape=%s | dtype=%s",
            elapsed,
            embeddings.shape,
            embeddings.dtype,
        )
        return embeddings

# Basic Test Cases for the Embedding Engine
if __name__ == "__main__":
    # Test 1: Normal execution.
    default_engine = ItemEmbeddingEngine()
    normal_texts = [
        "Black wallet with student ID card",
        "Set of car keys with red keychain",
        "Blue backpack near train station",
    ]
    normal_embeddings = default_engine.encode_batch(normal_texts)
    assert normal_embeddings.shape == (len(normal_texts), default_engine.dimension)
    assert normal_embeddings.dtype == np.float32

    # Test 2: Empty list handling.
    empty_embeddings = default_engine.encode_batch([])
    assert empty_embeddings.shape == (0, default_engine.dimension)
    assert empty_embeddings.dtype == np.float32

    # Test 3: Invalid type validation.
    invalid_payload = ["text", 123, None]
    try:
        _ = default_engine.encode_batch(invalid_payload)  # type: ignore[arg-type]
        raise AssertionError("ValueError expected for invalid input types")
    except ValueError:
        pass

    # Test 4: Explicit batching with small batch size.
    small_batch_engine = ItemEmbeddingEngine(
        EngineConfig(batch_size=2, max_seq_length=256, device="cpu")
    )
    five_items = [
        "Black iPhone 12 with cracked screen",
        "Silver ring with engraving",
        "Green umbrella left in cafeteria",
        "Laptop charger adapter near library",
        "Brown leather purse with zipper",
    ]
    batched_embeddings = small_batch_engine.encode_batch(five_items)
    assert batched_embeddings.shape == (5, small_batch_engine.dimension)
    assert batched_embeddings.dtype == np.float32

    # Test 5: Long text to verify truncation and stable processing.
    massive_text = " ".join(["lost_item_description"] * 10000)
    long_embeddings = small_batch_engine.encode_batch([massive_text])
    assert long_embeddings.shape == (1, small_batch_engine.dimension)
    assert long_embeddings.dtype == np.float32

    print("All tests passed.")