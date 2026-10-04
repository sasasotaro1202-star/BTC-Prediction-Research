from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path:sys.path.insert(0,str(ROOT/"src"))
import pattern_matrix_research as pm

def test_matrix_size_and_axes():
    assert len(pm.FEATURE_SETS)==10
    assert len(pm.MODELS)==8
    assert set(pm.WINDOWS)=={"expanding","recent_1500","recent_3000"}
    assert len(pm.candidates())==240

def test_feature_sets_are_unique_canonical_subsets():
    c=set(pm.FEATURE_ORDER)
    for f in pm.FEATURE_SETS.values():
        assert len(f)==len(set(f))
        assert set(f)<=c

def test_fingerprints_change_with_pattern_axis():
    assert pm.cfg("all_15","rf","expanding")["fingerprint"] != pm.cfg("trend","rf","expanding")["fingerprint"]
    assert pm.cfg("all_15","rf","expanding")["fingerprint"] != pm.cfg("all_15","rf","recent_1500")["fingerprint"]

def test_fold_geometry_is_chronological():
    e=pm.fold_ends(10000,2500,500,6)
    assert e==sorted(set(e))
    assert all(2500<=x<=9500 and x+500<=10000 for x in e)

def test_no_random_temporal_split():
    s=Path("src/pattern_matrix_research.py").read_text(encoding="utf-8")
    assert "train_test_split" not in s
    assert "shuffle=True" not in s
    assert ".sample(" not in s

def test_holdout_is_descriptive_only():
    s=Path("src/pattern_matrix_research.py").read_text(encoding="utf-8")
    assert '"used_for_selection":False' in s
    assert '"used_for_gate":False' in s
    assert '"status":"DESCRIPTIVE_ONLY"' in s
    assert "rows[:-HOLDOUT]" in s and "rows[-HOLDOUT:]" in s

def test_prequential_calibration_uses_prior_training_rows():
    s=Path("src/pattern_matrix_research.py").read_text(encoding="utf-8")
    assert "inner,cal=train[:split],train[split:]" in s
    assert "m.fit(Xall,yall)" in s

def test_lineage_and_promotion_firewall():
    s=Path("src/pattern_matrix_research.py").read_text(encoding="utf-8")
    assert 'sha=os.getenv("GITHUB_SHA") or "LOCAL_UNPINNED"' in s
    assert '"research_only":True' in s
    assert '"production_changed":False' in s
    assert '"promotion_allowed":False' in s

def test_screen_failures_are_persisted_not_silently_discarded():
    s=Path("src/pattern_matrix_research.py").read_text(encoding="utf-8")
    assert "screen_failures=[]" in s
    assert 'screen_failures.append({"config":c,"error_type":type(e).__name__' in s
    assert '"failures":screen_failures+failures' in s

def test_workflow_handoff_and_stale_main_guard():
    s=Path(".github/workflows/btc_pattern_matrix_research.yml").read_text(encoding="utf-8")
    assert 'echo "TESTS_PASSED=true" >> "$GITHUB_ENV"' in s
    assert 'echo "AUDIT_PASSED=true" >> "$GITHUB_ENV"' in s
    assert 'test "$TESTS_PASSED" = "true"' in s
    assert 'test "$AUDIT_PASSED" = "true"' in s
    assert 'if [ "$main_sha" != "$GITHUB_SHA" ]; then' in s

def test_workflow_expressions_are_not_backslash_escaped():
    s=Path(".github/workflows/btc_pattern_matrix_research.yml").read_text(encoding="utf-8")
    assert "\\${{" not in s
    assert "ref: ${{ github.sha }}" in s
    assert "group: btc-pattern-matrix-research" in s
