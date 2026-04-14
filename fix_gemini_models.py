"""
Fix Gemini Model Configuration
Auto-detect available models and update generator.py
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    print("[ERROR] google-genai not installed. Run: pip install google-genai")
    sys.exit(1)


def check_available_models():
    """
    Check available models từ Gemini API
    Returns list of model names (with models/ prefix)
    """
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not found in .env")
        return []
    
    try:
        client = genai.Client(api_key=api_key)
        
        print("\n[Checking available models...]")
        print("="*60)
        
        available = []
        
        for model in client.models.list():
            model_name = model.name
            print(f"[OK] {model_name}")
            available.append(model_name)
        
        print("="*60)
        return available
    
    except Exception as e:
        print(f"[ERROR] Failed to list models: {e}")
        return []


def update_generator_config(available_models):
    """
    Update generator.py with available models
    Prioritize by quota (flash-lite > flash > 1.5-flash)
    """
    
    if not available_models:
        print("[ERROR] No models available. Check API key and quota.")
        return False
    
    # Sort by preference
    preferred_order = [
        "models/gemini-2.5-flash-lite",  # Best: 15 RPM, 1000 RPD
        "models/gemini-2.5-flash",       # Good: 10 RPM, 250 RPD
        "models/gemini-1.5-flash",       # Legacy: 15 RPM
    ]
    
    # Filter to only available models
    prioritized = [m for m in preferred_order if m in available_models]
    
    if not prioritized:
        # Fallback: use any available flash model
        flash_models = [m for m in available_models if 'flash' in m.lower()]
        if flash_models:
            prioritized = flash_models
        else:
            prioritized = available_models[:3]  # Use first 3
    
    print(f"\n[Recommended Model Order]")
    for i, model in enumerate(prioritized, 1):
        print(f"  {i}. {model}")
    
    # Read current generator.py
    generator_path = Path("pipeline/generator.py")
    
    if not generator_path.exists():
        print(f"[ERROR] pipeline/generator.py not found")
        return False
    
    content = generator_path.read_text(encoding='utf-8')
    
    # Build new MODEL_FALLBACKS
    new_fallbacks = "MODEL_FALLBACKS = [\n"
    
    comments = {
        0: "# Primary (highest RPM/RPD)",
        1: "# Secondary (fallback if primary rate limited)",
        2: "# Tertiary (last resort)",
    }
    
    for i, model in enumerate(prioritized[:3]):
        comment = comments.get(i, "")
        new_fallbacks += f'    "{model}",  {comment}\n'
    
    new_fallbacks += "]"
    
    # Replace in content
    import re
    pattern = r'MODEL_FALLBACKS\s*=\s*\[[\s\S]*?\]'
    new_content = re.sub(pattern, new_fallbacks, content)
    
    # Write back
    generator_path.write_text(new_content, encoding='utf-8')
    
    print(f"\n[OK] Updated pipeline/generator.py")
    return True


def main():
    print("Gemini Model Configuration Fixer")
    print("="*60)
    
    # Step 1: Check available models
    available = check_available_models()
    
    if not available:
        print("\n[ERROR] Could not detect available models.")
        print("\nPossible reasons:")
        print("  1. API key invalid or missing")
        print("  2. API key exhausted (check quota in Google Console)")
        print("  3. Network issue")
        print("\nFix:")
        print("  - Check .env has valid GEMINI_API_KEY")
        print("  - Visit: https://aistudio.google.com/app/apikey")
        print("  - Check quota usage: https://console.cloud.google.com/")
        return False
    
    # Step 2: Update config
    success = update_generator_config(available)
    
    if success:
        print("\n[SUCCESS] Configuration updated!")
        print("\nNext steps:")
        print("  1. Run: python check_models.py (verify setup)")
        print("  2. Run: python app.py (start generation)")
    
    return success


if __name__ == "__main__":
    main()
