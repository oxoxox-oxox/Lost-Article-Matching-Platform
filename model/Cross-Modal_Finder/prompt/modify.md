# Role: Expert Machine Learning Engineer & Backend Architect

# Objective
Modify an existing production-ready Python Class, `ItemEmbeddingEngine`, to upgrade its capabilities from a text-only embedding system to a robust Multimodal (Text and Image) system using `sentence-transformers/clip-ViT-B-32`. The implementation must strictly adhere to enterprise-grade resource management, memory safety, and backward compatibility.

# Source Code
[ATTACH THE CONTENT OF `item_embedding_engine.py` HERE]

# Detailed Modification Instructions

Please refactor the provided Python script according to these strict engineering requirements:

## 1. Library Dependencies & Imports
- **Add Dependency:** Add `from PIL import Image, UnidentifiedImageError` to the imports. 
- Ensure standard library and third-party imports remain organized. Add `from typing import Union`.

## 2. Configuration Management & Backward Compatibility (`EngineConfig`)
- **Update Model:** Change `model_name` default to `'sentence-transformers/clip-ViT-B-32'`.
- **Backward Compatibility for `max_seq_length`:** DO NOT remove `max_seq_length` from the `EngineConfig` dataclass, as existing systems may instantiate `EngineConfig(max_seq_length=...)` and removing it will cause a `TypeError`. Keep it as `max_seq_length: int = 256`. 
- **Initialization Logic:** In `__init__`, completely ignore the `max_seq_length` setting (do not pass it to the model). Add a `self.logger.warning` stating that `max_seq_length` is ignored because CLIP uses a fixed context length.

## 3. Class Initialization & Future-Proof Warm-up (`__init__`)
- Maintain the stateful/singleton approach. `self.dimension` will dynamically update.
- **Dual Warm-up:** Warm up both the text and vision sub-models:
  - Text: `self.model.encode(["warmup_text"], ...)`
  - Image: Generate a generic dummy image: `Image.new('RGB', (100, 100), color='black')` and pass it to `encode`. Using 100x100 avoids hardcoding the model's exact expected resolution (like 224x224), allowing the `SentenceTransformer` internal preprocessor to handle the resizing dynamically.

## 4. Text Preprocessing & Encoding Update
- **Remove Lowercasing:** CLIP is case-sensitive. Update `_preprocess(self, text: str)` to strip whitespace but **DO NOT lowercase** the text to preserve semantic accuracy.
- **Rename Core Method:** Change the core logic method to `encode_text_batch`. 
- **Backward Compatibility:** Keep the original `encode_batch` method as a wrapper that calls `encode_text_batch`. Add a `logging.warning` indicating that `encode_batch` is deprecated.

## 5. NEW Method: Strict & Safe Image Preprocessing (`_preprocess_image`)
- **Signature:** `def _preprocess_image(self, input_data: Union[str, Image.Image]) -> Image.Image:`
- **Safe File Handling (No Context Managers):** - If `input_data` is a PIL `Image.Image`, verify it and proceed to channel conversion.
    - If `input_data` is a string (file path), use safe explicit handling to prevent file handle leaks: 
      `img = Image.open(input_data)` -> `rgb_img = img.convert('RGB')` -> `img.close()`. Return `rgb_img`. (The `.convert()` method creates a copy in memory, allowing the original file handle to be closed immediately).
    - Catch `FileNotFoundError` and `UnidentifiedImageError`, log them, and raise a `ValueError`.

## 6. NEW Method: Image Feature Extraction (`encode_image_batch`)
- **Signature:** `def encode_image_batch(self, image_inputs: List[Union[str, Image.Image]]) -> NDArray[np.float32]:`
- **Memory-Safe Batching & Resource Cleanup Logic:**
    - Iterate through `image_inputs` in chunks defined by `self.config.batch_size`.
    - **Crucial Inner Loop:** For each chunk, use a `try...finally` block to guarantee memory cleanup:
        1. Initialize an empty list `loaded_images = []`.
        2. Inside the `try` block: Loop through the chunk, call `_preprocess_image` on each item, and append to `loaded_images`. 
        3. Call `self.model.encode()` on `loaded_images` with EXACTLY: `batch_size=self.config.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False`.
        4. Append the resulting numpy array to a chunks list.
        5. Inside the `finally` block: Loop through `loaded_images` and explicitly call `.close()` on every image object, then clear the list. This prevents memory leaks even if an image fails to load or encoding crashes mid-chunk.
    - Stack the chunks (`np.vstack`) and ensure the final return type is `np.float32`.
- **Symmetric Exception Handling:** Implement `try-except` blocks identical to the text encoder around the encoding step, catching both `torch.OutOfMemoryError` and standard `RuntimeError` (checking for "out of memory" strings).

## 7. Deliverables & Testing Block (`if __name__ == "__main__":`)
Provide the complete, clean, production-ready Python code.
Update the unit test block with strict `assert` statements:
- **Test 1 (Text):** Verify text extraction shape is `(N, engine.dimension)`.
- **Test 2 (Backward Compatibility):** Test the deprecated `encode_batch`.
- **Test 3 (Image Execution & Dimensional Consistency):** Generate a dummy RGB image. Pass it to `encode_image_batch`. **Assert** the output shape is `(1, engine.dimension)`. **Explicitly assert** that the image output dimension matches the text output dimension.
- **Test 4 (Image Error Handling & Memory Safety):** Pass a valid image followed by a corrupted path (`["valid.jpg", "fake_image.jpg"]`). Use a mock or rely on the code to `assert` that a `ValueError` is raised. The `finally` block in the implementation ensures the valid image is cleaned up.
- **Test 5 (Cross-Modal Alignment - Rigorous):** - Text: "A photo of a solid red square." 
  - Image: `Image.new('RGB', (224, 224), color='red')`
  - Action: Calculate the dot product (cosine similarity) between them.
  - Assert: `assert similarity_score > 0.25`, demonstrating correct multi-modal alignment.