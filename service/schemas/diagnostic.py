from typing import Dict, List

from pydantic import BaseModel, Field


class DiagnosticResult(BaseModel):
    identified_features: Dict[str, str] = Field(default_factory=dict)
    missing_attributes: List[str] = Field(default_factory=list)
    missing_features: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    raw_llm_output: str | None = None
