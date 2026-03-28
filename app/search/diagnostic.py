import json
import os
import re
from typing import Any, Dict, Optional

from openai import OpenAI


CRITICAL_ATTRIBUTES = ["category", "color", "material", "brand", "size"]
OTHER_FEATURES = [
    "distinguishing_marks",
    "damage_condition",
    "accessories",
    "pattern",
    "location",
    "building_name",
    "floor_room",
    "approximate_date",
    "time_of_day",
    "time_context",
    "previous_owner_info",
    "special_circumstances",
]


def _get_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    resolved_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
    if not resolved_key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY or ZHIPUAI_API_KEY.")

    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4/"
    return OpenAI(api_key=resolved_key, base_url=resolved_base_url)


def _extract_json_block(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    return raw


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        s = str(item).strip().lower()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _normalize_result(parsed: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
    identified = parsed.get("identified_features")
    if not isinstance(identified, dict):
        identified = {}

    normalized_identified: Dict[str, str] = {}
    for k, v in identified.items():
        key = str(k).strip().lower()
        value = str(v).strip()
        if key and value:
            normalized_identified[key] = value

    missing_attributes = _clean_list(parsed.get("missing_attributes"))
    missing_features = _clean_list(parsed.get("missing_features"))

    # Enforce strict attribute set and priority order.
    attr_set = set(missing_attributes)
    ordered_missing_attributes = [a for a in CRITICAL_ATTRIBUTES if a in attr_set]

    feature_set = set(missing_features)
    feature_set = {f for f in feature_set if f not in CRITICAL_ATTRIBUTES}
    ordered_missing_features = [f for f in OTHER_FEATURES if f in feature_set]

    suggestions = parsed.get("suggestions")
    if not isinstance(suggestions, list):
        suggestions = []

    clean_suggestions: list[str] = []
    for item in suggestions:
        s = str(item).strip()
        if not s:
            continue
        clean_suggestions.append(s)

    # Ensure critical attributes are suggested first.
    if ordered_missing_attributes:
        attr_hint = "Please add these core item attributes first: " + ", ".join(ordered_missing_attributes) + "."
        if not clean_suggestions or attr_hint.lower() not in [x.lower() for x in clean_suggestions]:
            clean_suggestions.insert(0, attr_hint)

    if not clean_suggestions:
        clean_suggestions = _build_default_suggestions(ordered_missing_attributes, ordered_missing_features)

    return {
        "identified_features": normalized_identified,
        "missing_attributes": ordered_missing_attributes,
        "missing_features": ordered_missing_features,
        "suggestions": clean_suggestions[:6],
        "raw_llm_output": raw_output,
    }


def _build_default_suggestions(missing_attributes: list[str], missing_features: list[str]) -> list[str]:
    suggestions: list[str] = []
    if missing_attributes:
        suggestions.append(
            "Please add these core item attributes first: " + ", ".join(missing_attributes) + "."
        )
    if "distinguishing_marks" in missing_features or "damage_condition" in missing_features:
        suggestions.append("Please mention unique marks, scratches, labels, or any damage.")
    if "location" in missing_features or "building_name" in missing_features or "floor_room" in missing_features:
        suggestions.append("Please provide a more precise lost location, including building and floor if possible.")
    if "approximate_date" in missing_features or "time_of_day" in missing_features:
        suggestions.append("Please include when you likely lost it, such as date and time period.")

    if not suggestions:
        suggestions = [
            "Please add the item category, color, and material first.",
            "Please include exact location and approximate lost time.",
            "Please mention any unique marks or accessories.",
        ]
    return suggestions


def fallback_diagnostic(user_text: str) -> Dict[str, Any]:
    text = (user_text or "").strip().lower()
    identified: Dict[str, str] = {}

    if any(word in text for word in ["black", "white", "blue", "red", "green", "gray", "grey", "yellow", "brown"]):
        identified["color"] = "mentioned"
    if any(word in text for word in ["metal", "leather", "plastic", "cotton", "wood", "silicone", "fabric"]):
        identified["material"] = "mentioned"
    if any(word in text for word in ["bag", "phone", "wallet", "key", "earbud", "laptop", "watch", "bottle", "card"]):
        identified["category"] = "mentioned"

    missing_attributes = [a for a in CRITICAL_ATTRIBUTES if a not in identified]
    missing_features = [
        "distinguishing_marks",
        "location",
        "approximate_date",
        "time_of_day",
    ]

    return {
        "identified_features": identified,
        "missing_attributes": missing_attributes,
        "missing_features": missing_features,
        "suggestions": _build_default_suggestions(missing_attributes, missing_features),
        "raw_llm_output": "fallback",
    }


async def analyze_item_features(
    user_text: str,
    llm_model: str = "glm-4-flash",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = f"""
You are an expert Lost-and-Found analyzer.
The user input may be in any language. You must output English only.

Task:
1) Extract explicitly mentioned features.
2) Identify missing attributes from Product Attributes first:
   category, color, material, brand, size.
3) Identify missing features from other categories.
4) Return strict JSON only.

Output JSON schema:
{{
  "identified_features": {{
    "<feature_type>": "<extracted_value>"
  }},
  "missing_attributes": ["category|color|material|brand|size"],
  "missing_features": ["<other_feature_type>"],
  "suggestions": ["English suggestion 1", "English suggestion 2"]
}}

Allowed non-attribute feature types:
distinguishing_marks, damage_condition, accessories, pattern,
location, building_name, floor_room,
approximate_date, time_of_day, time_context,
previous_owner_info, special_circumstances.

Rules:
- Do not infer unsupported facts.
- Prioritize missing_attributes in suggestions.
- Suggestions must be short and actionable.
- Output JSON only, no markdown, no extra text.

User description: "{user_text}"
""".strip()

    client = _get_client(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )

    content = resp.choices[0].message.content
    if isinstance(content, list):
        content = "\n".join([b.get("text", "") for b in content if isinstance(b, dict)])
    raw_output = str(content or "").strip()

    parsed = json.loads(_extract_json_block(raw_output))
    if not isinstance(parsed, dict):
        raise ValueError("diagnostic output is not JSON object")

    return _normalize_result(parsed, raw_output)
