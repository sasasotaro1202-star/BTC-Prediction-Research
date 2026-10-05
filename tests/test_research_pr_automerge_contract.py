from pathlib import Path


WORKFLOW = Path(".github/workflows/btc_research_pr_automerge.yml")


def test_automerge_requires_current_main_head():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert 'gh api "repos/$REPO/compare/main...$head_sha"' in workflow
    assert 'behind_by="$(jq -r' in workflow
    assert 'if [ "$behind_by" -ne 0 ]; then' in workflow
    assert 'HOLD branch-behind-main=' in workflow


def test_automerge_keeps_sensitive_path_firewall():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert ".github/workflows/*" in workflow
    assert "data/*|models/*|registry/*|*holdout*" in workflow


def test_automerge_requires_explicit_research_marker():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert '<!-- btc-automerge:research-only -->' in workflow
