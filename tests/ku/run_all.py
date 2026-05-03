"""
tests/ku/run_all.py
--------------------
Chạy toàn bộ unit tests (không cần API).

Chạy:
    python tests/ku/run_all.py
"""

from __future__ import annotations

import sys
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

UNIT_TESTS = [
    "tests.ku.test_ku_schema",
    "tests.ku.test_ku_verify",
    "tests.ku.test_ku_graph",
    "tests.ku.test_ku_prompt",
    "tests.ku.test_ku_extract_mock",
]


def main():
    print(f"\n{'='*55}")
    print("  KU Pipeline — Unit Test Suite")
    print(f"{'='*55}\n")

    all_pass = True
    for module_name in UNIT_TESTS:
        mod = importlib.import_module(module_name)
        ok  = mod.run()
        if not ok:
            all_pass = False
        print()

    print(f"{'='*55}")
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED — see above")
    print(f"{'='*55}\n")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
