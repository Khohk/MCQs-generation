"""
pipeline/schemas.py
-------------------
Pydantic models cho MCQ pipeline.
Dùng để validate output LLM ở _parse_response() và enforce schema với Gemini.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel


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
