"""
Test MCQ Evaluation from Downloaded/External Files
Load MCQs from .xlsx, .json, .csv files and evaluate them
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.mcq_evaluator import (
    MCQEvaluator, MCQEvaluation, IWFFlaws, SemanticQuality
)
from evaluation.load_from_files import MCQFileLoader
from evaluation.test_with_real_data import evaluate_with_openai, create_evaluation_from_result


async def evaluate_from_file(
    filepath: str,
    num_mcqs: int = 10,
    judge_model: str = "gpt-4-turbo",
    output_prefix: str = "evaluation_from_file"
):
    """
    Evaluate MCQs loaded from external file (.xlsx, .json, .csv)
    
    Args:
        filepath: Path to file (supports .xlsx, .json, .csv)
        num_mcqs: Number of MCQs to evaluate
        judge_model: GPT-4 model for evaluation
        output_prefix: Prefix for output files
    """
    
    print("\n" + "="*80)
    print("MCQ EVALUATION FROM EXTERNAL FILE")
    print("="*80)
    
    # Step 1: Load MCQs from file
    print(f"\n[STEP 1] Loading MCQs from file...")
    print(f"  File: {filepath}")
    
    mcqs = MCQFileLoader.load_file(filepath)
    
    if not mcqs:
        print(f"  [ERROR] No MCQs loaded from file")
        return
    
    print(f"  [OK] Loaded {len(mcqs)} MCQs")
    
    # Step 2: Validate and standardize
    print(f"\n[STEP 2] Validating and standardizing...")
    valid_mcqs = MCQFileLoader.filter_valid(mcqs)
    standard_mcqs = MCQFileLoader.standardize_mcqs(valid_mcqs)
    
    print(f"  Valid MCQs: {len(valid_mcqs)}/{len(mcqs)}")
    print(f"  Standardized: {len(standard_mcqs)}")
    
    # Sample MCQs
    sample_mcqs = standard_mcqs[:num_mcqs]
    print(f"  Sample size: {len(sample_mcqs)}")
    
    # Step 3: Show sample MCQs
    print(f"\n[STEP 3] Sample MCQs to evaluate:")
    for i, mcq in enumerate(sample_mcqs[:3], 1):
        print(f"  {i}. Q: {mcq['question'][:60]}...")
        print(f"     Bloom: {mcq.get('bloom_level', 'unknown')}")
    
    # Step 4: Evaluate with OpenAI judge
    print(f"\n[STEP 4] Evaluating with OpenAI judge ({judge_model})...")
    
    evaluator = MCQEvaluator(judge_model=judge_model)
    
    for i, mcq in enumerate(sample_mcqs, 1):
        print(f"  [{i}/{len(sample_mcqs)}] Evaluating: {mcq.get('id', f'ext_{i}')}")
        
        # Call OpenAI judge
        judge_result = await evaluate_with_openai(mcq, judge_model)
        
        if judge_result:
            # Create evaluation from judge result
            evaluation = create_evaluation_from_result(mcq, judge_result)
            evaluator.evaluate(evaluation)
            print(f"    ✓ Verdict: {evaluation.overall_verdict} (IWF: {evaluation.flaw_pass_rate:.1f}, Semantic: {evaluation.semantic_average:.1f})")
        else:
            # Fallback if judge failed
            print(f"    ✗ Judge failed, using placeholder")
            evaluation = MCQEvaluation(
                question_id=mcq.get('id', f'ext_{i}'),
                question_text=mcq.get('question', ''),
                correct_answer=mcq.get('answer', ''),
                bloom_level=mcq.get('bloom_level', 'understand'),
                flaws=IWFFlaws(),
                semantic=SemanticQuality(relevance=3, answerability=3, bloom_alignment=3),
                feedback="Judge API failed - placeholder evaluation"
            )
            evaluator.evaluate(evaluation)
    
    # Step 5: Generate reports
    print(f"\n[STEP 5] Generating reports...")
    
    # Get summary
    stats = evaluator.get_pass_statistics()
    semantic = evaluator.get_semantic_average()
    
    print(f"\nSummary:")
    print(f"  Total: {stats['total_evaluated']}")
    print(f"  Pass Rate: {stats['pass_rate']*100:.1f}%")
    print(f"  Semantic Avg: {semantic['overall']}/5")
    
    # Export
    filename_safe = Path(filepath).stem.replace(' ', '_')
    output_json = f"{output_prefix}_{filename_safe}.json"
    output_txt = f"{output_prefix}_{filename_safe}.txt"
    
    evaluator.export_results(output_json)
    evaluator.export_detailed_report(output_txt)
    
    print(f"\n[OK] Results exported:")
    print(f"  - {output_json}")
    print(f"  - {output_txt}")


def create_sample_files():
    """Create sample files in data/test_results/ for testing"""
    
    test_results_dir = Path("data/test_results")
    test_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample JSON
    sample_json = [
        {
            "id": "sample_1",
            "question": "What is database normalization?",
            "A": "Increasing data storage",
            "B": "Reducing data redundancy",
            "C": "Making queries faster",
            "D": "Adding more servers",
            "answer": "B",
            "bloom_level": "understand",
            "difficulty": "medium",
            "source_chunk": "Chapter 2"
        },
        {
            "id": "sample_2",
            "question": "Which normal form eliminates non-key dependencies?",
            "A": "1NF",
            "B": "2NF",
            "C": "3NF",
            "D": "BCNF",
            "answer": "C",
            "bloom_level": "apply",
            "difficulty": "hard",
            "source_chunk": "Chapter 2"
        }
    ]
    
    json_path = test_results_dir / "sample_mcqs.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sample_json, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Created sample file: {json_path}")
    
    # Sample CSV (if pandas available)
    try:
        import pandas as pd
        
        df = pd.DataFrame(sample_json)
        csv_path = test_results_dir / "sample_mcqs.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"[OK] Created sample file: {csv_path}")
    except ImportError:
        print("[SKIP] pandas not available, skipping CSV creation")


if __name__ == "__main__":
    import asyncio
    
    print("MCQ Evaluation from External Files")
    print("="*60)
    
    # Evaluate from real file
    asyncio.run(evaluate_from_file(
        filepath="data/test_results/mcqs_NLP_Week_3.json",
        num_mcqs=10,
        output_prefix="real_results"
    ))
