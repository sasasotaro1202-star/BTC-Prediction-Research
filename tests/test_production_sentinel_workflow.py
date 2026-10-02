from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_sentinel_uses_shared_bounded_retry_per_step():
    workflow = (ROOT / ".github" / "workflows" / "btc_production_sentinel.yml").read_text(encoding="utf-8")

    assert "retry_api()" not in workflow
    assert "30s gh api" not in workflow
    assert "ci_gh_api_get" in workflow
    assert "ci_gh run cancel" in workflow

    recovery_start = workflow.index("      - name: Check and recover production heartbeat")
    recovery_end = workflow.index("      - name: Spawn next sentinel generation")
    recovery = workflow[recovery_start:recovery_end]
    assert ". scripts/ci_network_retry.sh" in recovery

    spawn_start = workflow.index("      - name: Spawn next sentinel generation")
    spawn = workflow[spawn_start:]
    assert ". scripts/ci_network_retry.sh" in spawn
    assert "ci_gh_api_get \"/repos/$REPO/actions/workflows/btc_production_sentinel.yml/dispatches\"" in spawn
