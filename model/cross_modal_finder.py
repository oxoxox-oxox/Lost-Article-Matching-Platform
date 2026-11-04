"""Production-ready multimodal embedding engine for Lost & Found descriptions.

This module provides a configurable `ItemEmbeddingEngine` built on
`sentence-transformers/clip-ViT-B-32` and optimized for safe, batched
text/image inference with normalized float32 vectors suitable for MySQL vector
storage.
"""
# Copyright 2024 Google LLC
from __future__ import annotations

# Standard Library Imports
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional, Union

# Third-Party Imports
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError
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

    model_name: str = "sentence-transformers/clip-ViT-B-32"
    batch_size: int = 32
    max_seq_length: int = 256
    device: str = "cuda"

# Embedding Engine Implementation
class ItemEmbeddingEngine:
    """Stateful multimodal embedding engine for Lost & Found descriptions.

    The model is loaded once per engine instance and reused for all calls.
    Encoded vectors are normalized and returned as `np.float32` to match
    MySQL vector database compatibility requirements.
    """

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        """Initialize the embedding engine and warm up the model.

        Args:
            config: Optional engine configuration. Defaults to `EngineConfig()`.

        Raises:
            RuntimeError: If the model cannot be loaded or warm-up fails.
            ValueError: If config values are invalid.
        """

        self.config = config or EngineConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.config.max_seq_length <= 0:
            raise ValueError("max_seq_length must be a positive integer")

        selected_device = self._resolve_device(self.config.device)
        self.config.device = selected_device

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
            self.logger.warning(
                "max_seq_length=%d is ignored because CLIP uses a fixed context length.",
                self.config.max_seq_length,
            )
            warmup_start = time.perf_counter()
            text_warmup = self.model.encode(
                ["warmup_text"],
                batch_size=1,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            warmup_image = Image.new("RGB", (100, 100), color="black")
            try:
                _ = self.model.encode(
                    [warmup_image],
                    batch_size=1,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            finally:
                warmup_image.close()

            model_dim = self.model.get_sentence_embedding_dimension()
            if model_dim is None:
                model_dim = int(np.asarray(text_warmup).shape[1])
            self.dimension = int(model_dim)

            load_elapsed = time.perf_counter() - load_start
            self.logger.info(
                "Model loaded in %.2f seconds | dimension=%d",
                load_elapsed,
                self.dimension,
            )

            warmup_elapsed = time.perf_counter() - warmup_start
            self.logger.info("Text and image warm-up complete in %.2f seconds", warmup_elapsed)
        except Exception as exc:
            self.logger.exception("Failed to initialize embedding engine: %s", exc)
            raise RuntimeError(
                f"Unable to initialize model '{self.config.model_name}'."
            ) from exc

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve and validate an inference device.

        Args:
            requested_device: Desired device string.

        Returns:
            A valid, available device string.
        """

        normalized = requested_device.strip().lower()
        if normalized == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            self.logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"

        if normalized == "mps":
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if has_mps:
                return "mps"
            self.logger.warning("MPS requested but not available. Falling back to CPU.")
            return "cpu"

        if normalized != "cpu":
            self.logger.warning(
                "Unsupported device '%s'. Falling back to CPU.", requested_device
            )
        return "cpu"

    def _preprocess(self, text: str) -> str:
        """Normalize raw text before embedding.

        Args:
            text: Raw input string.

        Returns:
            Cleaned text with preserved casing and collapsed whitespace.
        """

        return " ".join(text.strip().split())

    def encode_text_batch(self, texts: List[str]) -> NDArray[np.float32]:
        """Encode a list of text descriptions into normalized embeddings.

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

        for idx, item in enumerate(texts):
            if not isinstance(item, str):
                raise ValueError(
                    f"All elements must be strings. Invalid type at index {idx}: "
                    f"{type(item).__name__}"
                )

        cleaned_texts = [self._preprocess(text) for text in texts]
        total = len(cleaned_texts)
        self.logger.info(
            "Encoding %d texts with batch_size=%d on %s",
            total,
            self.config.batch_size,
            self.config.device,
        )

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

    def encode_batch(self, texts: List[str]) -> NDArray[np.float32]:
        """Backward-compatible wrapper for text encoding.

        Args:
            texts: Input descriptions to encode.

        Returns:
            A float32 numpy array with text embeddings.
        """

        self.logger.warning(
            "encode_batch is deprecated and will be removed in a future release; "
            "use encode_text_batch instead."
        )
        return self.encode_text_batch(texts)

    def _preprocess_image(self, input_data: Union[str, Image.Image]) -> Image.Image:
        """Load and normalize an image input into an RGB PIL image.

        Args:
            input_data: A filesystem path or an in-memory PIL image.

        Returns:
            A converted RGB PIL image that can be safely closed by callers.

        Raises:
            ValueError: If the input cannot be read as a valid image.
        """

        try:
            if isinstance(input_data, Image.Image):
                input_data.load()
                return input_data.convert("RGB")

            if isinstance(input_data, str):
                img = Image.open(input_data)
                rgb_img = img.convert("RGB")
                img.close()
                return rgb_img

            raise ValueError(
                "Each image input must be a file path string or PIL.Image.Image instance."
            )
        except FileNotFoundError as exc:
            self.logger.exception("Image file not found: %s", input_data)
            raise ValueError(f"Image file not found: {input_data}") from exc
        except UnidentifiedImageError as exc:
            self.logger.exception("Invalid image input: %s", input_data)
            raise ValueError(f"Invalid image input: {input_data}") from exc
        except OSError as exc:
            self.logger.exception("Unable to process image input: %s", input_data)
            raise ValueError(f"Unable to process image input: {input_data}") from exc

    def encode_image_batch(
        self,
        image_inputs: List[Union[str, Image.Image]],
    ) -> NDArray[np.float32]:
        """Encode a list of image inputs into normalized embeddings.

        Args:
            image_inputs: A list of image paths and/or PIL images.

        Returns:
            A float32 numpy array with shape `(len(image_inputs), self.dimension)`.

        Raises:
            ValueError: If image loading fails or input types are invalid.
            RuntimeError: If an out-of-memory condition occurs during encoding.
        """

        if not image_inputs:
            self.logger.warning(
                "Received empty image list; returning empty embedding array."
            )
            return np.empty((0, self.dimension), dtype=np.float32)

        total = len(image_inputs)
        self.logger.info(
            "Encoding %d images with batch_size=%d on %s",
            total,
            self.config.batch_size,
            self.config.device,
        )

        start_time = time.perf_counter()
        chunks: List[NDArray[np.float32]] = []

        for start in range(0, total, self.config.batch_size):
            end = min(start + self.config.batch_size, total)
            self.logger.info("Processing image chunk %d:%d of %d", start, end, total)

            loaded_images: List[Image.Image] = []
            try:
                for item in image_inputs[start:end]:
                    loaded_images.append(self._preprocess_image(item))

                try:
                    chunk_embeddings = self.model.encode(
                        loaded_images,
                        batch_size=self.config.batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                    chunks.append(np.asarray(chunk_embeddings, dtype=np.float32))
                except torch.OutOfMemoryError as exc:
                    self.logger.exception(
                        "PyTorch CUDA out-of-memory during image encoding: %s", exc
                    )
                    raise RuntimeError(
                        "Out-of-memory while encoding images. Reduce batch_size or use CPU."
                    ) from exc
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        self.logger.exception("Out-of-memory during image encoding: %s", exc)
                        raise RuntimeError(
                            "Out-of-memory while encoding images. "
                            "Reduce batch_size or input size."
                        ) from exc
                    raise
            finally:
                for loaded in loaded_images:
                    loaded.close()
                loaded_images.clear()

        embeddings = np.vstack(chunks).astype(np.float32, copy=False)
        elapsed = time.perf_counter() - start_time
        self.logger.info(
            "Image encoding complete in %.2f seconds | output_shape=%s | dtype=%s",
            elapsed,
            embeddings.shape,
            embeddings.dtype,
        )
        return embeddings
    

    # scale the raw image-text similarity score into a 0.0 to 1.0 range with a single-sided piecewise linear mapping
    @staticmethod
    def scale_image_score(score: float) -> float:
        """Scale raw image-text similarity scores into a 0.0 to 1.0 range with a single-sided piecewise linear mapping."""
        if score < 0.15:
            return 0.0  # Noise floor: 0.0 for scores below 0.15
        elif score <= 0.32:
            # normalize 0.15-0.32 to 0.0-0.8, giving moderate confidence in this range
            return ((score - 0.15) / (0.3 - 0.15)) * 0.8
        else:
            # high confidence range: 0.8-1.0 for scores above 0.3, with a linear ramp that caps at 1.0
            scaled = 0.8 + ((score - 0.3) / (0.50 - 0.3)) * 0.2
            return min(1.0, scaled)