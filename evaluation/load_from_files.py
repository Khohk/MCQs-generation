"""
Utility to load MCQ data from downloaded files (.xlsx, .json, .csv)
Flexible format converter for testing with external data
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class MCQFileLoader:
    """Load MCQs from various file formats"""
    
    @staticmethod
    def load_json(filepath: str) -> List[Dict[str, Any]]:
        """
        Load MCQs from JSON file
        
        Expected format:
        [
            {
                "question": "...",
                "A": "...", "B": "...", "C": "...", "D": "...",
                "answer": "A|B|C|D or A|B|C|D",
                "bloom_level": "remember|understand|apply|analyze|evaluate|create",
                "difficulty": "easy|medium|hard",
                "source_chunk": "..."  (optional)
            },
            ...
        ]
        
        Or checkpoint format:
        {
            "chunk_001": [mcq_object, ...],
            "chunk_002": [...]
        }
        """
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # If it's dict of chunks, flatten it
            if isinstance(data, dict) and not any(k in data for k in ["question", "A", "B"]):
                mcqs = []
                for chunk_id, chunk_data in data.items():
                    if isinstance(chunk_data, list):
                        mcqs.extend(chunk_data)
                return mcqs
            
            # If it's already list of MCQs
            if isinstance(data, list):
                return data
            
            return []
        
        except Exception as e:
            print(f"[ERROR] Failed to load JSON: {e}")
            return []
    
    @staticmethod
    def load_xlsx(filepath: str, sheet_name: str = 0) -> List[Dict[str, Any]]:
        """
        Load MCQs from Excel file
        
        Expected columns:
        question, A, B, C, D, answer, bloom_level, difficulty, source_chunk (optional)
        """
        
        if not HAS_PANDAS:
            print("[ERROR] pandas required. Run: pip install pandas openpyxl")
            return []
        
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            
            # Convert DataFrame to list of dicts
            mcqs = []
            for _, row in df.iterrows():
                mcq = row.to_dict()
                # Clean NaN values
                mcq = {k: v for k, v in mcq.items() if pd.notna(v)}
                mcqs.append(mcq)
            
            return mcqs
        
        except Exception as e:
            print(f"[ERROR] Failed to load Excel: {e}")
            return []
    
    @staticmethod
    def load_csv(filepath: str) -> List[Dict[str, Any]]:
        """
        Load MCQs from CSV file
        
        Expected columns:
        question, A, B, C, D, answer, bloom_level, difficulty, source_chunk (optional)
        """
        
        if not HAS_PANDAS:
            print("[ERROR] pandas required. Run: pip install pandas")
            return []
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # Convert DataFrame to list of dicts
            mcqs = []
            for _, row in df.iterrows():
                mcq = row.to_dict()
                # Clean NaN values
                mcq = {k: v for k, v in mcq.items() if pd.notna(v)}
                mcqs.append(mcq)
            
            return mcqs
        
        except Exception as e:
            print(f"[ERROR] Failed to load CSV: {e}")
            return []
    
    @staticmethod
    def load_file(filepath: str, sheet_name: str = 0) -> List[Dict[str, Any]]:
        """
        Auto-detect file type and load MCQs
        
        Supports: .json, .xlsx, .csv, .xls
        """
        
        path = Path(filepath)
        
        if not path.exists():
            print(f"[ERROR] File not found: {filepath}")
            return []
        
        suffix = path.suffix.lower()
        
        if suffix == '.json':
            return MCQFileLoader.load_json(filepath)
        elif suffix in ['.xlsx', '.xls']:
            return MCQFileLoader.load_xlsx(filepath, sheet_name=sheet_name)
        elif suffix == '.csv':
            return MCQFileLoader.load_csv(filepath)
        else:
            print(f"[ERROR] Unsupported file type: {suffix}")
            return []
    
    @staticmethod
    def validate_mcq(mcq: Dict[str, Any]) -> bool:
        """
        Validate MCQ has required fields
        
        Required: question, A, B, C, D, answer
        Optional: bloom_level, difficulty, source_chunk
        """
        
        required = {'question', 'A', 'B', 'C', 'D', 'answer'}
        return all(field in mcq for field in required)
    
    @staticmethod
    def filter_valid(mcqs: List[Dict]) -> List[Dict]:
        """Filter only valid MCQs"""
        return [mcq for mcq in mcqs if MCQFileLoader.validate_mcq(mcq)]
    
    @staticmethod
    def standardize_mcqs(mcqs: List[Dict]) -> List[Dict]:
        """
        Standardize MCQ format to match checkpoint structure
        
        Ensures all MCQs have: question, A, B, C, D, answer, bloom_level
        """
        
        standardized = []
        for i, mcq in enumerate(mcqs):
            if not MCQFileLoader.validate_mcq(mcq):
                continue
            
            std_mcq = {
                'question': mcq.get('question', ''),
                'A': mcq.get('A', ''),
                'B': mcq.get('B', ''),
                'C': mcq.get('C', ''),
                'D': mcq.get('D', ''),
                'answer': mcq.get('answer', ''),
                'bloom_level': mcq.get('bloom_level', 'understand').lower(),
                'difficulty': mcq.get('difficulty', 'medium').lower(),
                'source_chunk': mcq.get('source_chunk', ''),
                'id': mcq.get('id', f'external_{i}')
            }
            standardized.append(std_mcq)
        
        return standardized


# Example usage
if __name__ == "__main__":
    # Test loading from different formats
    
    print("MCQ File Loader Examples")
    print("=" * 60)
    
    # Example 1: Load from JSON
    print("\n[Example 1] Loading from JSON:")
    print("  mcqs = MCQFileLoader.load_file('data/test_results/mcqs.json')")
    
    # Example 2: Load from Excel
    print("\n[Example 2] Loading from Excel:")
    print("  mcqs = MCQFileLoader.load_file('data/test_results/mcqs.xlsx')")
    
    # Example 3: Load from CSV
    print("\n[Example 3] Loading from CSV:")
    print("  mcqs = MCQFileLoader.load_file('data/test_results/mcqs.csv')")
    
    # Example 4: Validate and standardize
    print("\n[Example 4] Validate and standardize:")
    print("  mcqs = MCQFileLoader.load_file('path/to/file')")
    print("  valid_mcqs = MCQFileLoader.filter_valid(mcqs)")
    print("  standard_mcqs = MCQFileLoader.standardize_mcqs(valid_mcqs)")
    
    print("\n" + "=" * 60)
    print("See test_from_results.py for full integration example")
