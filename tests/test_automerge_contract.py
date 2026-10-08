from pathlib import Path


WORKFLOW = Path(".github/workflows/btc_research_pr_automerge.yml")


def test_automerge_can_dispatch_post_merge_unit_tests():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions: write" in text
    assert "gh workflow run btc_unit_tests.yml --repo \"$REPO\" --ref main" in text
    assert "main_sha_after_merge=\"$(gh api \"repos/$REPO/git/ref/heads/main\" --jq '.object.sha')\"" in text
    assert "POST_MERGE_UNIT_TEST_DISPATCH_VERIFIED" in text


def test_automerge_binds_post_merge_verification_to_merge_sha():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'if [ "$main_sha_after_merge" != "$merge_sha" ]; then' in text
    assert 'select(.event == "workflow_dispatch")' in text
    assert 'select(.head_sha ==' in text
    assert '$merge_sha' in text
    assert "POST_MERGE_UNIT_TEST_DISPATCH_NOT_VERIFIED" in text


def test_automerge_fails_closed_if_post_merge_validation_cannot_be_started():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'if [ "$dispatch_ok" -ne 1 ]; then' in text
    assert 'exit 1' in text
