"""
Integration module: Connect MCQ Generator → MCQ Evaluator
Orchestrates the full pipeline: Generate → Evaluate → Report
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from pipeline.generator import MCQGenerator  # Your existing generator
from evaluation.mcq_evaluator import (
    MCQEvaluator, MCQEvaluation, IWFFlaws, SemanticQuality,
    EVALUATION_PROMPT_TEMPLATE
)


class EvaluationOrchestrator:
    """
    Orchestrates MCQ generation and evaluation pipeline
    
    Workflow:
    1. Generate MCQs from source documents (using pipeline.generator)
    2. Evaluate MCQs using IWF + Semantic Quality framework
    3. Export results and detailed reports
    4. Provide filtering/sorting capabilities
    """
    
    def __init__(self, 
                 generator: MCQGenerator,
                 judge_model: str = "gpt-4-turbo",
                 output_dir: str = "./evaluation/results"):
        """
        Args:
            generator: MCQGenerator instance for creating MCQs
            judge_model: LLM model to use for evaluation (judge)
            output_dir: Directory for saving results
        """
        self.generator = generator
        self.judge_model = judge_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluator = MCQEvaluator(judge_model=judge_model)
        self.generated_mcqs: List[Dict[str, Any]] = []
        self.evaluations: List[MCQEvaluation] = []
    
    async def generate_and_evaluate(self,
                                    documents: List[Dict[str, str]],
                                    batch_size: int = 10) -> Dict[str, Any]:
        """
        Full pipeline: Generate MCQs then evaluate them
        
        Args:
            documents: List of {"content": text, "title": name, "level": "Remember|Understand|..."}
            batch_size: Number of MCQs to evaluate in parallel
        
        Returns:
            Summary results dict
        """
        
        print(f"[PIPELINE] Starting generation and evaluation...")
        print(f"  Documents to process: {len(documents)}")
        print(f"  Judge model: {self.judge_model}")
        
        # Step 1: Generate MCQs
        print(f"\n[STEP 1/3] Generating MCQs...")
        self.generated_mcqs = await self.generator.generate_batch(documents)
        print(f"  [OK] Generated {len(self.generated_mcqs)} MCQs")
        
        # Step 2: Evaluate MCQs
        print(f"\n[STEP 2/3] Evaluating MCQs...")
        self.evaluations = await self._evaluate_batch(
            self.generated_mcqs, 
            batch_size=batch_size
        )
        print(f"  [OK] Evaluated {len(self.evaluations)} MCQs")
        
        # Step 3: Export results
        print(f"\n[STEP 3/3] Generating reports...")
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_documents": len(documents),
                "total_mcqs": len(self.generated_mcqs),
                "total_evaluated": len(self.evaluations),
                "judge_model": self.judge_model
            },
            "summary": self._generate_summary(),
            "evaluations": [e.to_dict() for e in self.evaluations]
        }
        
        # Save results
        self._export_all_results(results)
        print(f"  [OK] Reports saved to {self.output_dir}")
        
        return results
    
    async def _evaluate_batch(self, 
                             mcqs: List[Dict[str, Any]],
                             batch_size: int = 10) -> List[MCQEvaluation]:
        """
        Evaluate MCQs in batches (parallel API calls to judge)
        
        Args:
            mcqs: List of generated MCQ dicts
            batch_size: Number to evaluate in parallel
        
        Returns:
            List of MCQEvaluation objects
        """
        evaluations = []
        
        # For now, return mock evaluations (in production, call judge LLM)
        for i, mcq in enumerate(mcqs):
            # TODO: In production, send prompt to judge LLM here
            # For now, create placeholder evaluation
            evaluation = await self._evaluate_single(mcq, index=i)
            evaluations.append(evaluation)
            
            if (i + 1) % batch_size == 0:
                print(f"  Evaluated {i + 1}/{len(mcqs)}")
        
        self.evaluator.batch_evaluate(evaluations)
        return evaluations
    
    async def _evaluate_single(self, 
                              mcq: Dict[str, Any],
                              index: int) -> MCQEvaluation:
        """
        Evaluate single MCQ (integrate with judge LLM)
        
        Args:
            mcq: Generated MCQ dict with keys:
                 source_content, question, options, correct_answer, 
                 bloom_level, id
        
        Returns:
            MCQEvaluation object
        """
        
        # TODO: Call judge LLM with EVALUATION_PROMPT_TEMPLATE
        # For now, return placeholder
        
        # Example of what integration would look like:
        """
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            chunk_text=mcq["source_content"],
            question=mcq["question"],
            option_a=mcq["options"][0],
            option_b=mcq["options"][1],
            option_c=mcq["options"][2],
            option_d=mcq["options"][3],
            correct_answer=mcq["correct_answer"],
            bloom_level=mcq["bloom_level"]
        )
        
        response = await call_judge_llm(self.judge_model, prompt)
        result_json = json.loads(response)
        """
        
        # Placeholder evaluation
        evaluation = MCQEvaluation(
            question_id=mcq.get("id", f"Q_{index:03d}"),
            question_text=mcq.get("question", ""),
            correct_answer=mcq.get("correct_answer", ""),
            bloom_level=mcq.get("bloom_level", "Understand"),
            flaws=IWFFlaws(),  # Would be set by judge
            semantic=SemanticQuality(
                relevance=4,
                answerability=4,
                bloom_alignment=4
            ),
            feedback="Placeholder evaluation - integrate with judge LLM"
        )
        
        return evaluation
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from evaluations"""
        return {
            "pass_statistics": self.evaluator.get_pass_statistics(),
            "semantic_averages": self.evaluator.get_semantic_average(),
            "top_3_flaws": dict(self.evaluator.get_top_n_flaws(3)),
            "by_bloom_level": self.evaluator.get_by_bloom_level()
        }
    
    def _export_all_results(self, results_dict: Dict[str, Any]) -> None:
        """Export full results in multiple formats"""
        
        # JSON results
        json_path = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        # Text report
        text_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(self._format_text_report(results_dict))
        
        # CSV for spreadsheet analysis
        csv_path = self.output_dir / f"evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._export_csv(csv_path, results_dict['evaluations'])
    
    def _format_text_report(self, results_dict: Dict[str, Any]) -> str:
        """Format results as human-readable text report"""
        
        summary = results_dict['summary']
        stats = summary['pass_statistics']
        semantic = summary['semantic_averages']
        flaws = summary['top_3_flaws']
        by_bloom = summary['by_bloom_level']
        
        report = f"""
MCQ QUALITY EVALUATION REPORT
{'='*80}

Timestamp: {results_dict['metadata']['timestamp']}
Judge Model: {results_dict['metadata']['judge_model']}
Documents Processed: {results_dict['metadata']['total_documents']}
MCQs Generated: {results_dict['metadata']['total_mcqs']}
MCQs Evaluated: {results_dict['metadata']['total_evaluated']}

PASS/FAIL STATISTICS
{'-'*80}
Total: {stats['total_evaluated']}
[PASS] Pass (Production-Ready): {stats['pass']} ({stats['pass_rate']*100:.1f}%)
[WARN] Borderline: {stats['borderline']}
[FAIL] Fail: {stats['fail']}

SEMANTIC QUALITY (Likert 1-5)
{'-'*80}
Relevance: {semantic['relevance']}/5
Answerability: {semantic['answerability']}/5
Bloom Alignment: {semantic['bloom_alignment']}/5
Overall Average: {semantic['overall']}/5

TOP 3 COMMON FLAWS
{'-'*80}
"""
        
        for i, (flaw_name, count) in enumerate(flaws.items(), 1):
            pct = count / stats['total_evaluated'] * 100
            report += f"{i}. {flaw_name}: {count} ({pct:.1f}%)\n"
        
        report += f"""
PERFORMANCE BY BLOOM LEVEL
{'-'*80}
"""
        for bloom, bloom_stats in sorted(by_bloom.items()):
            report += f"""
{bloom}:
  Count: {bloom_stats['count']}
  Pass Rate: {bloom_stats['pass_rate']*100:.1f}%
  IWF Pass Rate: {bloom_stats['flaw_pass_rate']*100:.1f}%
  Semantic Average: {bloom_stats['semantic_avg']}/5
"""
        
        report += f"""
CRITERIA
{'-'*80}
PASS: IWF ≥ 80% AND Semantic ≥ 4.0
BORDERLINE: IWF ≥ 70% AND Semantic ≥ 3.5
FAIL: Below borderline

Item-Writing Flaws (10 rules): Objective checklist
Semantic Quality (3 dimensions): Subjective Likert scale

Reference:
- Arif et al. (2024) "Generation and Assessment of MCQs from LLMs" L@S '24
- Elkins et al. (2024) AAAI '24
- Zheng et al. (2023) "Judging LLM-as-a-Judge"

{'='*80}
"""
        
        return report
    
    def _export_csv(self, path: Path, evaluations: List[Dict]) -> None:
        """Export evaluations to CSV for spreadsheet analysis"""
        import csv
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'question_id', 'bloom_level', 'verdict',
                'flaw_pass_rate', 'semantic_avg',
                'relevance', 'answerability', 'bloom_alignment',
                'num_flaws', 'feedback'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for eval in evaluations:
                flaw_count = sum(1 for v in eval['flaws'].values() if v)
                
                writer.writerow({
                    'question_id': eval['question_id'],
                    'bloom_level': eval['bloom_level'],
                    'verdict': eval['overall_verdict'],
                    'flaw_pass_rate': eval['flaw_pass_rate'],
                    'semantic_avg': eval['semantic_average'],
                    'relevance': eval['semantic']['relevance'],
                    'answerability': eval['semantic']['answerability'],
                    'bloom_alignment': eval['semantic']['bloom_alignment'],
                    'num_flaws': flaw_count,
                    'feedback': eval['feedback']
                })
    
    def filter_by_verdict(self, verdict: str) -> List[MCQEvaluation]:
        """Get all MCQs with specific verdict (pass/borderline/fail)"""
        return [e for e in self.evaluations if e.overall_verdict == verdict]
    
    def filter_by_bloom(self, bloom_level: str) -> List[MCQEvaluation]:
        """Get all MCQs for specific Bloom level"""
        return [e for e in self.evaluations if e.bloom_level == bloom_level]
    
    def get_failing_mcqs(self) -> List[MCQEvaluation]:
        """Get all MCQs that failed (need revision)"""
        return self.filter_by_verdict("fail")
    
    def get_production_ready(self) -> List[MCQEvaluation]:
        """Get all production-ready MCQs (pass verdict)"""
        return self.filter_by_verdict("pass")


# Example usage
if __name__ == "__main__":
    
    # This is example template code
    # In actual implementation, you would:
    
    """
    from pipeline.generator import MCQGenerator
    
    # 1. Initialize generator
    generator = MCQGenerator(
        llm_model="gemini-pro",
        api_key="YOUR_KEY"
    )
    
    # 2. Load documents
    documents = [
        {
            "title": "Database Systems",
            "content": "... chapter content ...",
            "level": "Understand"
        },
        # ... more documents
    ]
    
    # 3. Create orchestrator
    orchestrator = EvaluationOrchestrator(
        generator=generator,
        judge_model="gpt-4-turbo",
        output_dir="./evaluation/results"
    )
    
    # 4. Run full pipeline
    results = asyncio.run(orchestrator.generate_and_evaluate(documents))
    
    # 5. Analyze results
    failing_mcqs = orchestrator.get_failing_mcqs()
    pass_rate = results['summary']['pass_statistics']['pass_rate']
    
    print(f"Pass Rate: {pass_rate:.1%}")
    print(f"Production-Ready MCQs: {len(orchestrator.get_production_ready())}")
    """
