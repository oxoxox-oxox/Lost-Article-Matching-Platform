# Role: Expert Machine Learning Engineer & Backend Architect

## Objective

Develop a highly robust, production-ready Python Class using `sentence-transformers` to extract semantic feature vectors from "Lost and Found" item descriptions.
The output vectors must be strictly formatted for downstream insertion into a MySQL Vector Database for high-speed similarity calculations.
**Crucial:** Use the pre-trained `sentence-transformers/all-MiniLM-L6-v2` model directly for inference. Do not include any training, dataset loading, or fine-tuning logic.

## Context & Production Requirements

- **Domain:** Lost & Found matching system. The vocabulary includes domain-specific synonyms and varied descriptions.
- **Performance:** High throughput and low latency are critical. The system must handle large bursts of data without crashing (OOM).
- **Future-Proofing:** The architecture must allow for easy evaluation of embedding quality and seamless swapping to other pre-trained models in the future.

## Technical Specifications & Architecture

1. **Architecture:** Implement an Object-Oriented approach. Create a class named `ItemEmbeddingEngine`.
2. **Configuration Management:** Use `dataclasses` to create an `EngineConfig` class containing `model_name`, `batch_size`, `max_seq_length`, and `device` (default 'cpu', support 'cuda'/'mps'). Avoid hardcoded magic numbers.
3. **Model Management (Singleton/Stateful):** - The model must be loaded *only once* during the class initialization (`__init__`).
4. **Dependencies & Typing:** - Use `sentence-transformers`, `torch`, `numpy`, `logging`, `typing`.
   - Use `numpy.typing.NDArray` for precise return types.

## Core Class Methods Required

- `__init__(self, config: Optional[EngineConfig] = None)`:
  - Initialize logger and handle potential download failures gracefully.
  - Load the model and explicitly apply the device setting (e.g., `device=config.device`).
  - **Crucial:** Explicitly enforce truncation by setting `self.model.max_seq_length = config.max_seq_length`.
  - **Crucial:** Dynamically fetch and store the vector dimension using `self.dimension = self.model.get_sentence_embedding_dimension()`.
  - **Warm-up:** Force the engine to initialize fully by executing a dummy encoding (e.g., `self.model.encode(["warmup_text"])`) and log the completion.
- `_preprocess(self, text: str) -> str`: Clean input text (lowercase, strip, remove excessive whitespace).
- `encode_batch(self, texts: List[str]) -> NDArray[np.float32]`:
  - Must process data in chunks by explicitly passing `batch_size=self.config.batch_size` to the underlying encode method to prevent OOM errors.
  - Enable `normalize_embeddings=True` to output normalized vectors.
  - Explicitly cast or verify the return type is `np.float32` to guarantee MySQL compatibility.

## Error Handling & Logging Standards

- **Logging:** Implement standard Python `logging`. Log model loading time, device used, dimension size, batch processing progress, and elapsed time.
- **Strict Exception Handling:**
  - If `texts` is an empty list, log a warning and return `np.empty((0, self.dimension), dtype=np.float32)` instead of failing.
  - Raise a clear `ValueError` immediately if elements in the list are not strings.
  - Handle potential PyTorch CUDA out-of-memory exceptions during encoding and log them clearly.

## Code Quality Standards

- Strict type hinting across all methods.
- Use **Google Style Docstrings** for the class and all methods, explicitly documenting Args, Returns, and Raises.

## Deliverables

1. The complete, production-ready Python code.
2. A comprehensive `if __name__ == "__main__":` test block that uses strict `assert` statements to verify behaviors:
   - **Test 1 (Normal Execution):** Pass a normal list of items. `assert` the output shape is `(len(texts), engine.dimension)` and dtype is `float32`.
   - **Test 2 (Empty List):** Pass `[]`. `assert` it returns shape `(0, engine.dimension)` without crashing.
   - **Test 3 (Invalid Data Types):** Pass `["text", 123, None]`. Use a `try...except` block to `assert` that a `ValueError` is correctly raised.
   - **Test 4 (Batching Logic):** Set `config.batch_size = 2` and pass a list of 5 texts. `assert` the output shape is `(5, engine.dimension)` to prove chunking works.
   - **Test 5 (Truncation & OOM Protection):** Pass a massive text string (e.g., generated 10,000 words). Verify it processes without crashing, proving `max_seq_length` took effect.
