"""
Test MCQ Evaluation with Real Data from Checkpoints
Load MCQs from data/checkpoints/*.json and evaluate with OpenAI judge
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Force UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.mcq_evaluator import (
    MCQEvaluator, MCQEvaluation, IWFFlaws, SemanticQuality,
    EVALUATION_PROMPT_TEMPLATE
)


def load_checkpoints(data_dir: str = "./data/checkpoints") -> Dict[str, List[Dict]]:
    """Load all MCQs from checkpoint JSON files"""
    
    checkpoints = {}
    checkpoint_dir = Path(data_dir)
    
    if not checkpoint_dir.exists():
        print(f"[ERROR] Directory not found: {checkpoint_dir}")
        return {}
    
    for filepath in sorted(checkpoint_dir.glob("checkpoint_*.json")):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                checkpoint_name = filepath.stem.replace("checkpoint_", "")
                checkpoints[checkpoint_name] = data
                print(f"[OK] Loaded {checkpoint_name}")
        except Exception as e:
            print(f"[ERROR] Failed to load {filepath}: {e}")
    
    return checkpoints


def flatten_mcqs(checkpoints: Dict[str, List[Dict]]) -> List[Dict]:
    """Flatten checkpoint structure to list of MCQs with metadata"""
    
    mcqs = []
    for checkpoint_name, chunks in checkpoints.items():
        for chunk_id, mcq_list in chunks.items():
            for mcq in mcq_list:
                mcq_with_meta = {
                    **mcq,
                    'checkpoint': checkpoint_name,
                    'chunk_id': chunk_id,
                    'id': f"{checkpoint_name}_{chunk_id}_{len(mcqs)}"
                }
                mcqs.append(mcq_with_meta)
    
    return mcqs


async def evaluate_with_openai(
    mcq: Dict[str, Any],
    judge_model: str = "gpt-4-turbo"
) -> Dict[str, Any]:
    """
    Evaluate single MCQ using OpenAI as judge
    
    Args:
        mcq: MCQ dict from checkpoint with Q, A, B, C, D, answer
        judge_model: OpenAI model (default gpt-4-turbo)
    
    Returns:
        Evaluation result dict
    """
    
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        
        # Load OpenAI key from .env
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("[ERROR] OPENAI_API_KEY not found in .env")
            return None
        
        client = OpenAI(api_key=api_key)
        
        # Format prompt
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            chunk_text=mcq.get("source_chunk", ""),  # Optional context
            question=mcq.get("question", ""),
            option_a=mcq.get("A", ""),
            option_b=mcq.get("B", ""),
            option_c=mcq.get("C", ""),
            option_d=mcq.get("D", ""),
            correct_answer=mcq.get("answer", ""),
            bloom_level=mcq.get("bloom_level", "Understand").capitalize()
        )
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Deterministic
            max_tokens=1000
        )
        
        result_text = response.choices[0].message.content
        
        # Parse JSON from response
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown if wrapped
            if "```json" in result_text:
                json_start = result_text.find("```json") + 7
                json_end = result_text.find("```", json_start)
                result = json.loads(result_text[json_start:json_end].strip())
            else:
                print(f"[ERROR] Failed to parse JSON response: {result_text[:100]}")
                return None
        
        return result
    
    except ImportError:
        print("[ERROR] openai package not installed. Run: pip install openai")
        return None
    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        return None


def create_evaluation_from_result(
    mcq: Dict,
    judge_result: Dict
) -> MCQEvaluation:
    """Convert judge result to MCQEvaluation object"""
    
    if not judge_result:
        # Fallback if judge failed
        return MCQEvaluation(
            question_id=mcq.get("id", "unknown"),
            question_text=mcq.get("question", ""),
            correct_answer=mcq.get("answer", ""),
            bloom_level=mcq.get("bloom_level", "Understand"),
            flaws=IWFFlaws(),
            semantic=SemanticQuality(relevance=3, answerability=3, bloom_alignment=3),
            feedback="Judge evaluation failed"
        )
    
    # Create flaws from judge result
    flaws = IWFFlaws(
        unfocused_stem=judge_result.get("flaws", {}).get("unfocused_stem", False),
        unclear_information=judge_result.get("flaws", {}).get("unclear_information", False),
        negative_wording_without_emphasis=judge_result.get("flaws", {}).get("negative_wording_without_emphasis", False),
        implausible_distractors=judge_result.get("flaws", {}).get("implausible_distractors", False),
        heterogeneous_options=judge_result.get("flaws", {}).get("heterogeneous_options", False),
        longest_answer_correct_bias=judge_result.get("flaws", {}).get("longest_answer_correct_bias", False),
        word_repeats=judge_result.get("flaws", {}).get("word_repeats", False),
        absolute_terms_in_distractors=judge_result.get("flaws", {}).get("absolute_terms_in_distractors", False),
        multiple_correct_answers=judge_result.get("flaws", {}).get("multiple_correct_answers", False),
        factual_error=judge_result.get("flaws", {}).get("factual_error", False),
    )
    
    # Create semantic scores
    semantic_data = judge_result.get("semantic", {})
    semantic = SemanticQuality(
        relevance=semantic_data.get("relevance", 3),
        answerability=semantic_data.get("answerability", 3),
        bloom_alignment=semantic_data.get("bloom_alignment", 3),
    )
    
    return MCQEvaluation(
        question_id=mcq.get("id", "unknown"),
        question_text=mcq.get("question", ""),
        correct_answer=mcq.get("answer", ""),
        bloom_level=mcq.get("bloom_level", "Understand"),
        flaws=flaws,
        semantic=semantic,
        feedback=judge_result.get("feedback", "")
    )


async def test_with_real_data(
    num_mcqs: int = 5,
    checkpoint_pattern: str = None
):
    """Test evaluation with real MCQs from checkpoints"""
    
    print("\n" + "="*80)
    print("MCQ EVALUATION TEST WITH REAL DATA")
    print("="*80)
    
    # Step 1: Load checkpoints
    print("\n[STEP 1] Loading checkpoints...")
    checkpoints = load_checkpoints()
    
    if not checkpoints:
        print("[ERROR] No checkpoints found")
        return
    
    # Step 2: Flatten and sample MCQs
    print(f"\n[STEP 2] Extracting MCQs (limiting to {num_mcqs})...")
    all_mcqs = flatten_mcqs(checkpoints)
    print(f"  Total MCQs found: {len(all_mcqs)}")
    
    # Filter by pattern if specified
    if checkpoint_pattern:
        sample_mcqs = [m for m in all_mcqs if checkpoint_pattern.lower() in m['checkpoint'].lower()][:num_mcqs]
    else:
        sample_mcqs = all_mcqs[:num_mcqs]
    
    print(f"  Sample MCQs to evaluate: {len(sample_mcqs)}")
    for i, mcq in enumerate(sample_mcqs, 1):
        print(f"    {i}. {mcq['checkpoint']} - Q: {mcq['question'][:60]}...")
    
    # Step 3: Evaluate each MCQ
    print(f"\n[STEP 3] Evaluating with OpenAI judge (gpt-4-turbo)...")
    evaluator = MCQEvaluator(judge_model="gpt-4-turbo")
    
    for i, mcq in enumerate(sample_mcqs, 1):
        print(f"\n  [{i}/{len(sample_mcqs)}] Evaluating: {mcq['checkpoint']}")
        print(f"      Q: {mcq['question'][:70]}...")
        
        # Call judge
        judge_result = await evaluate_with_openai(mcq)
        
        if judge_result:
            # Convert to MCQEvaluation
            evaluation = create_evaluation_from_result(mcq, judge_result)
            evaluator.evaluate(evaluation)
            
            print(f"      Verdict: {evaluation.overall_verdict.upper()}")
            print(f"      IWF Pass Rate: {evaluation.flaw_pass_rate*100:.0f}%")
            print(f"      Semantic Avg: {evaluation.semantic_average:.2f}/5")
        else:
            print(f"      [SKIP] Judge evaluation failed")
    
    # Step 4: Generate report
    print(f"\n[STEP 4] Generating reports...")
    
    stats = evaluator.get_pass_statistics()
    semantic = evaluator.get_semantic_average()
    top_flaws = evaluator.get_top_n_flaws(3)
    by_bloom = evaluator.get_by_bloom_level()
    
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nPass Statistics:")
    print(f"  Total: {stats['total_evaluated']}")
    print(f"  [PASS] Production-Ready: {stats['pass']} ({stats['pass_rate']*100:.1f}%)")
    print(f"  [WARN] Borderline: {stats['borderline']}")
    print(f"  [FAIL] Fail: {stats['fail']}")
    
    print(f"\nSemantic Quality (Likert 1-5):")
    print(f"  Relevance: {semantic['relevance']}/5")
    print(f"  Answerability: {semantic['answerability']}/5")
    print(f"  Bloom Alignment: {semantic['bloom_alignment']}/5")
    print(f"  Overall: {semantic['overall']}/5")
    
    if top_flaws:
        print(f"\nTop Flaws:")
        for flaw, count in top_flaws:
            pct = count / stats['total_evaluated'] * 100
            print(f"  - {flaw}: {count} ({pct:.0f}%)")
    
    print(f"\nPerformance by Bloom Level:")
    for bloom, bloom_stats in sorted(by_bloom.items()):
        print(f"  {bloom}:")
        print(f"    Count: {bloom_stats['count']}")
        print(f"    Pass Rate: {bloom_stats['pass_rate']*100:.1f}%")
        print(f"    Semantic Avg: {bloom_stats['semantic_avg']}/5")
    
    # Step 5: Export results
    print(f"\n[STEP 5] Exporting results...")
    evaluator.export_results("evaluation_real_data_results.json")
    evaluator.export_detailed_report("evaluation_real_data_report.txt")
    print(f"  [OK] Results saved")
    print(f"       - evaluation_real_data_results.json")
    print(f"       - evaluation_real_data_report.txt")


if __name__ == "__main__":
    import asyncio
    
    # Test chapter_2_DataModels checkpoint with 10 MCQs
    asyncio.run(test_with_real_data(
        num_mcqs=10,
        checkpoint_pattern="chapter_2"
    ))
