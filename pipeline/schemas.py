"""
pipeline/schemas.py
-------------------
Pydantic models cho MCQ pipeline.
Dùng để validate output LLM ở _parse_response() và enforce schema với Gemini.
"""

from __future__ import annotations

try:
    from typing import Literal
except ImportError:  # Python 3.7 compatibility
    from typing_extensions import Literal
from pydantic import BaseModel, field_validator


class MCQItem(BaseModel):
    question:    str
    A:           str
    B:           str
    C:           str
    D:           str
    answer:      Literal["A", "B", "C", "D"]
    explanation: str
    bloom_level: Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]
    difficulty:  Literal["easy", "medium", "hard"]
    source_chunk: str


class KUItem(BaseModel):
    ku_id:             str
    type:              Literal["definition", "mechanism", "failure_mode",
                               "trade_off", "procedure", "application"]
    concept:           str
    content:           str
    verbatim_evidence: str
    related_concepts:  list[str]
    source_pages:      list[int]
    prominence:        Literal["primary", "supporting", "peripheral"]
    completeness:      Literal["complete", "incomplete"]

    @field_validator("source_pages", mode="before")
    @classmethod
    def coerce_pages(cls, v):
        if isinstance(v, int):
            return [v]
        if isinstance(v, str):
            import re
            return [int(x) for x in re.findall(r"\d+", v)] or [0]
        return v

    @field_validator("related_concepts", mode="before")
    @classmethod
    def coerce_related(cls, v):
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v
