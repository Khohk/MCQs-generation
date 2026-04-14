"""
Example usage of MCQ Evaluator
Demonstrates how to evaluate MCQs using IWF + Semantic Quality framework
"""

from mcq_evaluator import (
    MCQEvaluator, MCQEvaluation, IWFFlaws, SemanticQuality,
    EVALUATION_PROMPT_TEMPLATE
)
import json


# Example 1: Evaluate a single MCQ
def example_single_evaluation():
    """Evaluate a single MCQ"""
    
    # Create IWF flaws (findings from LLM judge)
    flaws = IWFFlaws(
        unfocused_stem=False,
        unclear_information=False,
        negative_wording_without_emphasis=False,
        implausible_distractors=False,
        heterogeneous_options=False,
        longest_answer_correct_bias=False,
        word_repeats=False,
        absolute_terms_in_distractors=False,
        multiple_correct_answers=False,
        factual_error=False
    )
    
    # Create semantic quality scores
    semantic = SemanticQuality(
        relevance=5,
        answerability=4,
        bloom_alignment=5
    )
    
    # Create evaluation
    evaluation = MCQEvaluation(
        question_id="Q001_Database_Complex_Queries",
        question_text="What is the primary purpose of database indexing?",
        correct_answer="To improve query performance",
        bloom_level="Understand",
        flaws=flaws,
        semantic=semantic,
        feedback="Excellent MCQ. Clear question, plausible distractors, perfect alignment with Understand level."
    )
    
    print("Single MCQ Evaluation:")
    print(json.dumps(evaluation.to_dict(), indent=2, ensure_ascii=False))
    print(f"\nOverall Verdict: {evaluation.overall_verdict}")
    print(f"IWF Pass Rate: {evaluation.flaw_pass_rate*100:.1f}%")
    print(f"Semantic Average: {evaluation.semantic_average:.2f}/5")


# Example 2: Batch evaluate multiple MCQs
def example_batch_evaluation():
    """Batch evaluate multiple MCQs and generate report"""
    
    evaluator = MCQEvaluator(judge_model="gpt-4-turbo")
    
    # Create sample evaluations
    sample_mcqs = [
        MCQEvaluation(
            question_id="Q001",
            question_text="What is database normalization?",
            correct_answer="A process of organizing data to minimize redundancy",
            bloom_level="Remember",
            flaws=IWFFlaws(
                longest_answer_correct_bias=True,  # 1 flaw
            ),
            semantic=SemanticQuality(relevance=5, answerability=5, bloom_alignment=4),
            feedback="Good question but correct answer is too long compared to distractors."
        ),
        MCQEvaluation(
            question_id="Q002",
            question_text="Which of the following is NOT a type of database?",
            correct_answer="File system",
            bloom_level="Recall",
            flaws=IWFFlaws(
                negative_wording_without_emphasis=True,  # 1 flaw
                implausible_distractors=True,  # another flaw
            ),
            semantic=SemanticQuality(relevance=4, answerability=4, bloom_alignment=3),
            feedback="Negative wording not highlighted. Some distractors are too obvious."
        ),
        MCQEvaluation(
            question_id="Q003",
            question_text="Analyze how ACID properties ensure data integrity in transaction processing.",
            correct_answer="ACID ensures atomicity, consistency, isolation, durability",
            bloom_level="Analyze",
            flaws=IWFFlaws(),  # No flaws
            semantic=SemanticQuality(relevance=5, answerability=5, bloom_alignment=5),
            feedback="Excellent MCQ. Production-ready."
        ),
        MCQEvaluation(
            question_id="Q004",
            question_text="Database refers to all computer systems.",
            correct_answer="False",
            bloom_level="Remember",
            flaws=IWFFlaws(
                unfocused_stem=True,
                factual_error=True,
            ),
            semantic=SemanticQuality(relevance=2, answerability=2, bloom_alignment=4),
            feedback="Question is ambiguous and factually incorrect."
        ),
        MCQEvaluation(
            question_id="Q005",
            question_text="Which property guarantees that successful transactions will persist?",
            correct_answer="Durability",
            bloom_level="Understand",
            flaws=IWFFlaws(),  # No flaws
            semantic=SemanticQuality(relevance=5, answerability=5, bloom_alignment=5),
            feedback="Clear, concise, and well-aligned with Understand level."
        ),
    ]
    
    # Batch evaluate
    evaluator.batch_evaluate(sample_mcqs)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("BATCH EVALUATION SUMMARY")
    print("="*80)
    
    stats = evaluator.get_pass_statistics()
    print(f"\nPass Statistics:")
    print(f"  Total MCQs: {stats['total_evaluated']}")
    print(f"  Pass (Production-Ready): {stats['pass']} ({stats['pass_rate']*100:.1f}%)")
    print(f"  Borderline: {stats['borderline']}")
    print(f"  Fail: {stats['fail']}")
    
    semantic_avg = evaluator.get_semantic_average()
    print(f"\nSemantic Quality Averages:")
    print(f"  Relevance: {semantic_avg['relevance']}/5")
    print(f"  Answerability: {semantic_avg['answerability']}/5")
    print(f"  Bloom Alignment: {semantic_avg['bloom_alignment']}/5")
    print(f"  Overall: {semantic_avg['overall']}/5")
    
    top_flaws = evaluator.get_top_n_flaws(3)
    print(f"\nTop 3 Most Common Flaws:")
    for i, (flaw, count) in enumerate(top_flaws, 1):
        pct = count / stats['total_evaluated'] * 100
        print(f"  {i}. {flaw}: {count} MCQs ({pct:.0f}%)")
    
    by_bloom = evaluator.get_by_bloom_level()
    print(f"\nPerformance by Bloom Level:")
    for bloom, stats_bloom in sorted(by_bloom.items()):
        print(f"\n  {bloom}:")
        print(f"    Count: {stats_bloom['count']}")
        print(f"    IWF Pass Rate: {stats_bloom['flaw_pass_rate']*100:.1f}%")
        print(f"    Semantic Avg: {stats_bloom['semantic_avg']}/5")
        print(f"    Production-Ready: {stats_bloom['pass_rate']*100:.1f}%")
    
    # Export results
    print("\n" + "="*80)
    evaluator.export_results("mcq_evaluation_results.json")
    evaluator.export_detailed_report("mcq_evaluation_report.txt")
    print("[OK] Results exported to:")
    print("  - mcq_evaluation_results.json")
    print("  - mcq_evaluation_report.txt")


# Example 3: Show the LLM judge prompt
def example_llm_prompt():
    """Show the evaluation prompt template for LLM judge"""
    
    sample_chunk = """
    Database normalization is a process of organizing data in a database to minimize redundancy 
    and improve data integrity. There are several normal forms (1NF, 2NF, 3NF, BCNF) that represent 
    increasing levels of normalization. The goal is to decompose relations with anomalies into 
    smaller, well-structured relations.
    """
    
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        chunk_text=sample_chunk,
        question="What is the primary goal of database normalization?",
        option_a="To increase storage space",
        option_b="To minimize data redundancy and improve integrity",
        option_c="To make queries faster",
        option_d="To allow multiple users to access the database",
        correct_answer="To minimize data redundancy and improve integrity",
        bloom_level="Understand"
    )
    
    print("\n" + "="*80)
    print("SAMPLE LLM EVALUATION PROMPT")
    print("="*80)
    print(prompt)


if __name__ == "__main__":
    print("MCQ EVALUATOR - EXAMPLE USAGE\n")
    
    # Run examples
    example_single_evaluation()
    print("\n" + "-"*80 + "\n")
    
    example_batch_evaluation()
    print("\n" + "-"*80 + "\n")
    
    example_llm_prompt()
