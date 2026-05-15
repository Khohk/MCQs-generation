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
from pydantic import BaseModel, field_validator, model_validator

# Horizontal edges → distractor selection (confusable concepts)
HORIZONTAL_RELATIONS: frozenset[str] = frozenset({
    "CONTRASTS_WITH", "ALTERNATIVE_TO", "SIBLING_OF", "SIMILAR_TO",
})

# Structural edges → UI hierarchy / explanation only
STRUCTURAL_RELATIONS: frozenset[str] = frozenset({
    "PART_OF", "ENABLES", "EXTENDS", "APPLIES_TO",
})

ALL_RELATIONS = HORIZONTAL_RELATIONS | STRUCTURAL_RELATIONS


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


class RelatedKUEdge(BaseModel):
    target_concept: str
    relation:       Literal[
                        # Horizontal — for distractor selection
                        "CONTRASTS_WITH", "ALTERNATIVE_TO", "SIBLING_OF", "SIMILAR_TO",
                        # Structural — for UI / explanation
                        "PART_OF", "ENABLES", "EXTENDS", "APPLIES_TO",
                    ]
    evidence:       str

    @field_validator("target_concept", "evidence", mode="before")
    @classmethod
    def strip_str(cls, v):
        return v.strip() if isinstance(v, str) else v

    @property
    def is_horizontal(self) -> bool:
        return self.relation in HORIZONTAL_RELATIONS


class KUItem(BaseModel):
    ku_id:             str
    type:              Literal["definition", "mechanism", "failure_mode",
                               "trade_off", "procedure", "application"]
    concept:           str
    local_concept:     str = ""
    owner_level:       Literal["L1", "L2", "local"] = "local"
    owner_concept:     str = ""
    parent_l1:         str = ""
    parent_l2:         str = ""
    content:           str
    verbatim_evidence: str
    related_kus:       list[RelatedKUEdge] = []
    related_concepts:  list[str] = []
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
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v

    @model_validator(mode="after")
    def sync_related_concepts(self) -> "KUItem":
        if not self.related_concepts and self.related_kus:
            self.related_concepts = [e.target_concept for e in self.related_kus]
        return self
