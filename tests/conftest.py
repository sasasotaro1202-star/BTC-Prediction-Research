"""Shared test import path configuration.

Keep repository source and research scripts importable under plain pytest
invocation, matching the GitHub Actions test environment without requiring
shell-specific PYTHONPATH configuration.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
