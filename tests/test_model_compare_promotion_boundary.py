import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestModelComparePromotionBoundary(unittest.TestCase):
    def test_compare_h_cannot_mutate_production(self):
        tree = ast.parse((ROOT / "src" / "model_compare.py").read_text(encoding="utf-8"))
        target = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "compare_h"
        )
        calls = {
            node.func.id
            for node in ast.walk(target)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("adopt_candidate", calls)
        self.assertNotIn("set_prod", calls)

    def test_research_result_declares_no_production_change(self):
        text = (ROOT / "src" / "model_compare.py").read_text(encoding="utf-8")
        self.assertIn("'production_changed':False", text)
        self.assertIn("'eligible_pending_explicit_promotion'", text)


if __name__ == "__main__":
    unittest.main()
