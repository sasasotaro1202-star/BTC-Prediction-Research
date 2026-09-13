import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from model_compare import CLASSES, EMBARGO_BARS, PURGE_BARS, load_rows, metrics, normalize  # noqa: E402


def test_research_class_order_matches_db_storage_conversion():
    assert CLASSES == ['DOWN', 'FLAT', 'UP']
    stored_up_down_flat = [0.70, 0.10, 0.20]
    research_down_flat_up = [stored_up_down_flat[1], stored_up_down_flat[2], stored_up_down_flat[0]]
    assert research_down_flat_up == [0.10, 0.20, 0.70]
    assert max(range(3), key=lambda i: research_down_flat_up[i]) == CLASSES.index('UP')


def test_probability_normalization_is_finite_and_sums_to_one():
    p = normalize([[0.7, 0.2, 0.1], [10.0, 0.0, 0.0]])
    assert p.shape == (2, 3)
    assert all(math.isfinite(float(x)) for x in p.ravel())
    assert all(abs(float(row.sum()) - 1.0) < 1e-9 for row in p)


def test_metrics_respect_research_class_order():
    ys = ['UP', 'DOWN', 'FLAT']
    probs = [[0.05, 0.05, 0.90], [0.90, 0.05, 0.05], [0.05, 0.90, 0.05]]
    m = metrics(ys, probs)
    assert m['accuracy'] == 1.0
    assert m['logloss'] < 0.2
    assert m['brier'] < 0.1


def test_horizon_purge_and_embargo_are_conservative():
    assert PURGE_BARS['5m'] == 5
    assert PURGE_BARS['10m'] == 10
    assert EMBARGO_BARS['5m'] >= 60
    assert EMBARGO_BARS['10m'] >= 60
