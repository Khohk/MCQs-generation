"""
prompts/bloom_definitions.py
----------------------------
Bloom's Taxonomy constants for PS4 prompt engineering.
No logic — only data referenced by build_ps4_prompt.
"""

from __future__ import annotations

BLOOM_SPECS: dict[str, dict] = {
    "remember": {
        "description": "Retrieve factual knowledge from memory",
        "verbs": "define, list, identify, recall, recognize, name",
        "example": "Which of the following best defines a data pipeline?",
    },
    "understand": {
        "description": "Construct meaning, explain ideas in own words",
        "verbs": "explain, summarize, classify, describe, interpret, paraphrase",
        "example": "Which scenario best illustrates the concept of data lineage?",
    },
    "apply": {
        "description": "Use a procedure or concept in a given situation",
        "verbs": "use, execute, implement, solve, calculate, demonstrate",
        "example": "Given a sparse dataset with many missing values, which preprocessing step should you apply first?",
    },
    "analyze": {
        "description": "Break information into parts, find relationships and patterns",
        "verbs": "compare, differentiate, contrast, distinguish, break down, examine",
        "example": "What is the key difference between batch processing and stream processing that explains their latency trade-off?",
    },
    "evaluate": {
        "description": "Make judgments or recommendations based on criteria",
        "verbs": "justify, critique, assess, judge, recommend, defend",
        "example": "A team chooses accuracy as the sole metric on a highly imbalanced dataset. Which criticism of this choice is most valid?",
    },
    "create": {
        "description": "Put elements together to produce a novel solution or design",
        "verbs": "design, construct, plan, combine, propose, formulate",
        "example": "Which combination of techniques best addresses catastrophic forgetting in continual learning?",
    },
}

# Maps density label → ordered list of Bloom levels to generate
DENSITY_TO_LEVELS: dict[str, list[str]] = {
    # Vietnamese labels
    "Ít":    ["understand", "apply"],
    "Vừa":   ["remember", "understand", "apply", "analyze"],
    "Nhiều": ["remember", "understand", "apply", "analyze", "evaluate", "create"],
    # English aliases
    "low":    ["understand", "apply"],
    "Low":    ["understand", "apply"],
    "medium": ["remember", "understand", "apply", "analyze"],
    "Medium": ["remember", "understand", "apply", "analyze"],
    "high":   ["remember", "understand", "apply", "analyze", "evaluate", "create"],
    "High":   ["remember", "understand", "apply", "analyze", "evaluate", "create"],
}
