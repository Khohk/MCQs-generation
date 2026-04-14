"""
MCQ Quality Evaluation Framework
Uses Item-Writing Flaws (IWFs) + Semantic Quality assessment
Based on: Arif et al. (2024, L@S '24) + Elkins et al. (2024)
"""

import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum


class BloomLevel(Enum):
    """Bloom's Taxonomy levels"""
    REMEMBER = "Remember"
    UNDERSTAND = "Understand"
    APPLY = "Apply"
    ANALYZE = "Analyze"
    EVALUATE = "Evaluate"
    CREATE = "Create"


@dataclass
class IWFFlaws:
    """Item-Writing Flaws checklist (10 critical rules for LLM-generated MCQs)"""
    unfocused_stem: bool = False
    unclear_information: bool = False
    negative_wording_without_emphasis: bool = False
    implausible_distractors: bool = False
    heterogeneous_options: bool = False
    longest_answer_correct_bias: bool = False
    word_repeats: bool = False
    absolute_terms_in_distractors: bool = False
    multiple_correct_answers: bool = False
    factual_error: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)

    @property
    def pass_rate(self) -> float:
        """Calculate IWF pass rate (0-1): number of flaws NOT present / 10"""
        flaws_count = sum(self.to_dict().values())
        return (10 - flaws_count) / 10


@dataclass
class SemanticQuality:
    """Semantic Quality scores (Likert 1-5 scale)"""
    relevance: int  # 1-5: How well does question relate to source?
    answerability: int  # 1-5: Can question be answered using only source?
    bloom_alignment: int  # 1-5: Does question match labeled Bloom level?

    @property
    def average(self) -> float:
        return (self.relevance + self.answerability + self.bloom_alignment) / 3


@dataclass
class MCQEvaluation:
    """Complete evaluation result for single MCQ"""
    question_id: str
    question_text: str
    correct_answer: str
    bloom_level: str
    
    flaws: IWFFlaws
    semantic: SemanticQuality
    
    feedback: str = ""
    
    @property
    def flaw_pass_rate(self) -> float:
        return self.flaws.pass_rate
    
    @property
    def semantic_average(self) -> float:
        return self.semantic.average
    
    @property
    def overall_verdict(self) -> str:
        """Pass/Fail/Borderline based on both metrics"""
        # Criteria: IWF ≥ 80% AND Semantic avg ≥ 4.0 = PASS
        if self.flaw_pass_rate >= 0.8 and self.semantic_average >= 4.0:
            return "pass"
        elif self.flaw_pass_rate >= 0.7 and self.semantic_average >= 3.5:
            return "borderline"
        else:
            return "fail"

    def to_dict(self) -> Dict:
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "correct_answer": self.correct_answer,
            "bloom_level": self.bloom_level,
            "flaws": self.flaws.to_dict(),
            "flaw_pass_rate": round(self.flaw_pass_rate, 3),
            "semantic": {
                "relevance": self.semantic.relevance,
                "answerability": self.semantic.answerability,
                "bloom_alignment": self.semantic.bloom_alignment
            },
            "semantic_average": round(self.semantic_average, 2),
            "overall_verdict": self.overall_verdict,
            "feedback": self.feedback
        }


class MCQEvaluator:
    """Main evaluator class for batch MCQ evaluation"""
    
    def __init__(self, judge_model: str = "cross-model"):
        """
        Initialize evaluator
        judge_model: 'cross-model' (default) = Gemini generates, GPT-4/Claude judges
        """
        self.judge_model = judge_model
        self.evaluations: List[MCQEvaluation] = []
    
    def evaluate(self, evaluation: MCQEvaluation) -> MCQEvaluation:
        """Single MCQ evaluation"""
        self.evaluations.append(evaluation)
        return evaluation
    
    def batch_evaluate(self, evaluations: List[MCQEvaluation]) -> List[MCQEvaluation]:
        """Batch evaluate multiple MCQs"""
        self.evaluations.extend(evaluations)
        return evaluations
    
    def get_flaw_distribution(self) -> Dict[str, int]:
        """Get distribution of flaws across all evaluated MCQs"""
        flaw_counts = {}
        for eval in self.evaluations:
            for flaw_name, is_present in eval.flaws.to_dict().items():
                if is_present:
                    flaw_counts[flaw_name] = flaw_counts.get(flaw_name, 0) + 1
        
        # Sort by frequency
        return dict(sorted(flaw_counts.items(), key=lambda x: x[1], reverse=True))
    
    def get_top_n_flaws(self, n: int = 3) -> List[Tuple[str, int]]:
        """Get top N most common flaws"""
        dist = self.get_flaw_distribution()
        return list(dist.items())[:n]
    
    def get_semantic_average(self) -> Dict[str, float]:
        """Get average semantic scores across all MCQs"""
        if not self.evaluations:
            return {}
        
        relevance_avg = sum(e.semantic.relevance for e in self.evaluations) / len(self.evaluations)
        answerability_avg = sum(e.semantic.answerability for e in self.evaluations) / len(self.evaluations)
        bloom_avg = sum(e.semantic.bloom_alignment for e in self.evaluations) / len(self.evaluations)
        
        return {
            "relevance": round(relevance_avg, 2),
            "answerability": round(answerability_avg, 2),
            "bloom_alignment": round(bloom_avg, 2),
            "overall": round((relevance_avg + answerability_avg + bloom_avg) / 3, 2)
        }
    
    def get_pass_statistics(self) -> Dict[str, Any]:
        """Get production-ready statistics"""
        if not self.evaluations:
            return {}
        
        verdicts = [e.overall_verdict for e in self.evaluations]
        pass_count = verdicts.count("pass")
        fail_count = verdicts.count("fail")
        borderline_count = verdicts.count("borderline")
        
        pass_rate = pass_count / len(self.evaluations)
        
        return {
            "total_evaluated": len(self.evaluations),
            "pass": pass_count,
            "borderline": borderline_count,
            "fail": fail_count,
            "pass_rate": round(pass_rate, 3),
            "production_ready_rate": round(pass_rate, 3),  # % MCQ ≥ 80% IWF AND ≥ 4.0 semantic
        }
    
    def get_by_bloom_level(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics grouped by Bloom level"""
        by_bloom = {}
        
        for eval in self.evaluations:
            bloom = eval.bloom_level
            if bloom not in by_bloom:
                by_bloom[bloom] = {
                    "count": 0,
                    "pass_rate": 0,
                    "flaw_pass_rate": 0,
                    "semantic_avg": 0
                }
            
            by_bloom[bloom]["count"] += 1
            by_bloom[bloom]["flaw_pass_rate"] += eval.flaw_pass_rate
            by_bloom[bloom]["semantic_avg"] += eval.semantic_average
        
        # Calculate averages
        for bloom in by_bloom:
            count = by_bloom[bloom]["count"]
            by_bloom[bloom]["flaw_pass_rate"] = round(by_bloom[bloom]["flaw_pass_rate"] / count, 3)
            by_bloom[bloom]["semantic_avg"] = round(by_bloom[bloom]["semantic_avg"] / count, 2)
            
            # Pass rate: IWF ≥ 80% AND Semantic ≥ 4.0
            pass_count = sum(1 for e in self.evaluations 
                           if e.bloom_level == bloom and e.overall_verdict == "pass")
            by_bloom[bloom]["pass_rate"] = round(pass_count / count, 3)
        
        return by_bloom
    
    def export_results(self, filepath: str) -> None:
        """Export all evaluations to JSON"""
        results = {
            "metadata": {
                "judge_model": self.judge_model,
                "total_mcqs": len(self.evaluations),
                "evaluation_framework": "IWF (10 rules) + Semantic Quality (3 Likert dimensions)"
            },
            "summary_statistics": {
                "pass_statistics": self.get_pass_statistics(),
                "semantic_averages": self.get_semantic_average(),
                "top_3_flaws": dict(self.get_top_n_flaws(3)),
                "by_bloom_level": self.get_by_bloom_level()
            },
            "evaluations": [e.to_dict() for e in self.evaluations]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def export_detailed_report(self, filepath: str) -> None:
        """Export human-readable report"""
        report = self._generate_report()
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            f.write(report)
    
    def _generate_report(self) -> str:
        """Generate detailed report text"""
        stats = self.get_pass_statistics()
        semantic = self.get_semantic_average()
        flaws = self.get_top_n_flaws(3)
        by_bloom = self.get_by_bloom_level()
        
        report = f"""
MCQ QUALITY EVALUATION REPORT
{'='*80}

Framework: Item-Writing Flaws (IWFs) + Semantic Quality Assessment
Judge Model: Cross-model (LLM-as-a-judge via {self.judge_model})
Reference: Arif et al. (L@S '24), Elkins et al. (AAAI '24)

SUMMARY STATISTICS
{'-'*80}
Total MCQs Evaluated: {stats['total_evaluated']}
Production-Ready (Pass): {stats['pass']} ({stats['pass_rate']*100:.1f}%)
Borderline: {stats['borderline']}
Needs Revision (Fail): {stats['fail']}

SEMANTIC QUALITY AVERAGES (Likert 1-5)
{'-'*80}
Relevance: {semantic['relevance']}/5
Answerability: {semantic['answerability']}/5
Bloom Alignment: {semantic['bloom_alignment']}/5
Overall Semantic Average: {semantic['overall']}/5

TOP 3 MOST COMMON FLAWS
{'-'*80}
"""
        for i, (flaw_name, count) in enumerate(flaws, 1):
            percentage = round(count / stats['total_evaluated'] * 100, 1)
            report += f"{i}. {flaw_name}: {count} MCQs ({percentage}%)\n"
        
        report += f"""
PERFORMANCE BY BLOOM LEVEL
{'-'*80}
"""
        for bloom, stats_bloom in sorted(by_bloom.items()):
            report += f"""
{bloom}:
  Count: {stats_bloom['count']}
  IWF Pass Rate: {stats_bloom['flaw_pass_rate']*100:.1f}%
  Semantic Average: {stats_bloom['semantic_avg']}/5
  Production-Ready: {stats_bloom['pass_rate']*100:.1f}%
"""
        
        report += f"""
EVALUATION CRITERIA
{'-'*80}
PASS: IWF Pass Rate ≥ 80% AND Semantic Average ≥ 4.0
BORDERLINE: IWF Pass Rate ≥ 70% AND Semantic Average ≥ 3.5
FAIL: Below borderline thresholds

Item-Writing Flaws (10 rules):
1. Unfocused Stem
2. Unclear Information
3. Negative Wording Without Emphasis
4. Implausible Distractors
5. Heterogeneous Options
6. Longest Answer Correct Bias
7. Word Repeats
8. Absolute Terms in Distractors
9. Multiple Correct Answers
10. Factual Error

Semantic Quality (3 Likert 1-5 dimensions):
1. Relevance: Connection to source content
2. Answerability: Can be answered using only source
3. Bloom Alignment: Matches labeled Bloom level

{'='*80}
"""
        return report


# Prompt template for LLM judge
EVALUATION_PROMPT_TEMPLATE = """You are an expert in educational assessment specializing in 
multiple-choice question quality analysis.

Source content:
\"\"\"
{chunk_text}
\"\"\"

MCQ to evaluate:
Question: {question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
Correct Answer: {correct_answer}
Labeled Bloom Level: {bloom_level}

## PART 1: Item-Writing Flaws (answer YES if flaw is present, NO if absent)

For each item, determine if the MCQ contains this specific flaw:

1. **Unfocused Stem**: Does the question lack a clear, single focus?
   Answer: [YES/NO]

2. **Unclear Information**: Is any information in the stem ambiguous or difficult to understand?
   Answer: [YES/NO]

3. **Negative Wording Without Emphasis**: Does it use negation (NOT, EXCEPT) without highlighting that critical word?
   Answer: [YES/NO]

4. **Implausible Distractors**: Are any distractors too obviously wrong or unrealistic?
   Answer: [YES/NO]

5. **Heterogeneous Options**: Are the 4 options inconsistent in type, length, or grammatical structure?
   Answer: [YES/NO]

6. **Longest Answer Correct Bias**: Is the correct answer noticeably longer than the distractors?
   Answer: [YES/NO]

7. **Word Repeats**: Do any options share unique or distinctive words with the stem that giveaway the answer?
   Answer: [YES/NO]

8. **Absolute Terms in Distractors**: Do distractors use absolute language (always, never, all, none)?
   Answer: [YES/NO]

9. **Multiple Correct Answers**: Could more than one option reasonably be considered correct?
   Answer: [YES/NO]

10. **Factual Error**: Is any information in the question or options factually incorrect based on the source?
    Answer: [YES/NO]

## PART 2: Semantic Quality (score 1-5 for each dimension)

1. **Relevance** (1-5): How well does the question relate to and test the source content?
   - 5: Directly tests critical concept from source
   - 4: Tests important concept from source
   - 3: Tests minor concept or somewhat related
   - 2: Tangentially related
   - 1: Not related or hallucination
   Score: [1-5]

2. **Answerability** (1-5): Can the question be confidently answered using ONLY the source content?
   - 5: Clearly answerable from source, no missing info
   - 4: Mostly answerable from source
   - 3: Requires some inference beyond source
   - 2: Significant gaps in source information
   - 1: Cannot be answered from source alone
   Score: [1-5]

3. **Bloom Alignment** (1-5): Does the question match the labeled Bloom level?
   - 5: Perfect alignment with labeled level
   - 4: Mostly matches (±0 levels)
   - 3: Slightly misaligned (±1 level)
   - 2: Notably misaligned (±2 levels)
   - 1: Completely wrong level (±3+ levels)
   Score: [1-5]

## OUTPUT FORMAT

Return ONLY valid JSON (no markdown, no explanation):

{{
  "flaws": {{
    "unfocused_stem": false,
    "unclear_information": false,
    "negative_wording_without_emphasis": false,
    "implausible_distractors": false,
    "heterogeneous_options": false,
    "longest_answer_correct_bias": false,
    "word_repeats": false,
    "absolute_terms_in_distractors": false,
    "multiple_correct_answers": false,
    "factual_error": false
  }},
  "flaw_pass_rate": 0.9,
  "semantic": {{
    "relevance": 5,
    "answerability": 4,
    "bloom_alignment": 5
  }},
  "semantic_average": 4.67,
  "overall_verdict": "pass",
  "feedback": "Brief actionable feedback for improvement if verdict is fail/borderline"
}}
"""
