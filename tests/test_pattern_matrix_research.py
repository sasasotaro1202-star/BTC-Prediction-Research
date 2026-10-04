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

def test_window_minimum_training_geometry_matches_window_size():
    assert pm.min_train_for_window("expanding") == pm.MIN_TRAIN
    assert pm.min_train_for_window("recent_3000") == pm.MIN_TRAIN
    assert pm.min_train_for_window("recent_1500") == 1500

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
    assert "screen_failures=list(checkpoint.get" in s
    assert "screen_failures" in s
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
    assert "cancel-in-progress: false" in s

def test_screen_selection_is_multi_objective_and_diverse():
    rows=[]
    for i in range(18):
        rows.append({
            "feature_set":f"f{i%6}",
            "model":f"m{i%4}",
            "window":["expanding","recent_1500","recent_3000"][i%3],
            "fingerprint":str(i),
            "aggregate":{
                "relative_logloss_improvement":0.01+i/1000,
                "relative_brier_improvement":0.005+(17-i)/2000
            },
            "stability":{
                "accuracy_non_worse_ratio":0.5+(i%5)/10,
                "logloss_improved_ratio":0.5+(i%4)/10,
                "brier_improved_ratio":0.5+(i%3)/10,
                "worst_logloss_delta":0.2-i/1000,
                "worst_accuracy_delta":-0.1+i/1000
            }
        })
    selected=pm.select_finalists(rows,limit=12)
    assert len(selected)==12
    assert len({x["fingerprint"] for x in selected})==12
    assert len({x["model"] for x in selected})>=3
    assert all("screen_selection_score" in x for x in selected)
def test_canonical_docs_match_matrix_contract():
    instructions=Path("PROJECT_INSTRUCTIONS.md").read_text(encoding="utf-8")
    source=Path("docs/PROJECT_SOURCE.md").read_text(encoding="utf-8")
    expected_matrix_markers = (
        "10 feature sets × 8 deterministic model variants × 3 training-window policies",
        "240 configurations",
    )
    for text in (instructions,source):
        assert all(marker in text for marker in expected_matrix_markers)
        assert "recent_1500" in text
        assert "logreg_c0.03" in text
        assert "mean_reversion" in text


def test_checkpoint_lineage_contract():
    s=Path("src/pattern_matrix_research.py").read_text(encoding="utf-8")
    assert "CHECKPOINT_DIR" in s
    assert "rows_fingerprint(rows)" in s
    assert "candidate_manifest_hash(cs)" in s
    assert "load_checkpoint(h,rows_fp,candidate_hash)" in s
    assert "save_checkpoint(h," in s
    assert '"status":"RUNNING" if stage!="COMPLETED" else "COMPLETED"' in s
    assert '"COMPLETED"' in s

def test_checkpoint_rejects_cross_sha_resume():
    s=Path("src/pattern_matrix_research.py").read_text(encoding="utf-8")
    assert 'obj.get("analysis_git_sha")!=_analysis_sha()' in s
    assert 'obj.get("rows_fingerprint")!=rows_fp' in s


def test_workflow_restores_same_sha_checkpoints():
    s=Path(".github/workflows/btc_pattern_matrix_research.yml").read_text(encoding="utf-8")
    assert "actions: read" in s
    assert "Restore same-SHA pattern checkpoints" in s
    assert "head_sha == $sha" in s
    assert 'conclusion == "failure" or .conclusion == "timed_out"' in s
    assert "actions/artifacts/$" in s
    assert "btc-pattern-matrix-checkpoints-" in s
    assert "if: always()" in s

def test_checkpoint_reuses_only_successful_candidates():
    s=Path("src/pattern_matrix_research.py").read_text(encoding="utf-8")
    assert "screen_ok=False" in s
    assert 'if screen_ok:\n            completed_screen.add(c["fingerprint"])' in s
    assert "final_ok=False" in s
    assert 'if final_ok:\n            completed_final.add(c["fingerprint"])' in s
