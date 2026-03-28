# Role Context
You are a Senior Python Backend Engineer specializing in Multimodal AI and Vector Search. Please help me refactor the "Fine-ranking" (re-ranking) logic for my multimodal search system.

# Background
We are using a custom embedding engine class named `ItemEmbeddingEngine` (defined in `Cross-Modal_Finder.py`). The model outputs L2-normalized `np.float32` embeddings. I need you to write a new standalone function (e.g., `calculate_fine_ranking_score`) that strictly follows my new business logic.

# Core Business Logic (Strict Adherence Required)

1. **Input Parameters**:
   The fine-ranking stage NO LONGER uses any scores from the previous "recall/coarse" stage. The new function must accept exactly these parameters:
   - `user_text` (str): The text inputted by the user.
   - `user_image` (str or PIL.Image): The image inputted by the user.
   - `item_text` (str): The text description of the target item in the database.
   - `item_image` (Optional[Union[str, PIL.Image]]): The image of the target item (can be `None` or empty).
   - `engine` (ItemEmbeddingEngine): An initialized instance of the embedding engine.

2. **Feature Extraction (Handle 2D Arrays)**:
   Extract features using the provided engine. Since the methods return 2D arrays `(batch_size, dimension)`, you MUST extract the 1D vector (e.g., `[0]` or `.flatten()`) for dot product calculations:
   - Extract `user_text_vec` and `item_text_vec` using `engine.encode_text_batch([text])[0]`
   - Extract `user_image_vec` using `engine.encode_image_batch([image])[0]`
   - Extract `item_image_vec` ONLY IF `item_image` is not None/empty.

3. **Similarity Calculation & Mapping Rules**:
   Since the vectors are already normalized, use **Numpy dot product (`np.dot`)** to calculate Cosine Similarity. You must calculate 2 or 3 specific scores and apply different mapping rules:
   - **Score 1 (Text-Text)**: Dot product of `user_text_vec` and `item_text_vec`. **Do NOT apply any linear mapping.**
   - **Score 2 (Image-Text)**: Dot product of `user_image_vec` and `item_text_vec`. **MUST be linearly scaled** by calling the static method: `ItemEmbeddingEngine.scale_image_score(score)`.
   - **Score 3 (Image-Image)**: Dot product of `user_image_vec` and `item_image_vec` (Only if `item_image` exists). **Do NOT apply any linear mapping.**

4. **Score Fusion**:
   - Collect the valid calculated scores (either 2 or 3 scores) into a list.
   - Calculate the final fine-ranking score as the **Arithmetic Mean** of this list.
   - Ensure the final returned score is cast to a standard Python `float` (not `numpy.float32`) for JSON serialization compatibility.
   - Do NOT blend or use any external weights or previous recall scores.

# Output Requirements
- Output ONLY the complete Python function code.
- Ensure production-ready type hints, docstrings, and comments explaining the logic.
- Ensure robust handling for the optional `item_image`.