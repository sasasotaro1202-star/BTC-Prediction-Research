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

# Keep legacy top-level imports and package imports pointed at one module object.
# This prevents duplicate loading of src/predict.py under both "predict" and
# "src.predict", which can create collection-time import-order failures.
import importlib
_predict_module = importlib.import_module("src.predict")
sys.modules.setdefault("predict", _predict_module)
