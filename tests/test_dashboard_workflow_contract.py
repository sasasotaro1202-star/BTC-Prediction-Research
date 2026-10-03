from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "btc_pages_dashboard.yml"


def test_dashboard_packaging_is_independent_from_pages_enablement():
    text = WORKFLOW.read_text(encoding="utf-8")
    package_at = text.index("  package:\n")
    prepare_at = text.index("      - name: Prepare dashboard site", package_at)
    upload_at = text.index("      - name: Upload GitHub Pages artifact", prepare_at)
    deploy_at = text.index("  deploy:\n", package_at)
    configure_at = text.index("      - name: Configure GitHub Pages", deploy_at)

    assert package_at < prepare_at < upload_at < deploy_at < configure_at
    assert "    needs: package\n" in text[deploy_at:]
    assert "    permissions:\n      contents: read\n    steps:" in text[package_at:deploy_at]
    assert "      pages: write" in text[deploy_at:configure_at]


def test_dashboard_package_preserves_current_data_contract():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "forecast_trajectory.json" in text
    assert "pit_oos_audit.json" in text
    assert "live_cycle_status.json" in text
    assert '["5m","10m","15m","30m","1h","3h","6h","12h","24h"]' in text
    assert "BTC dashboard data: VERIFIED" in text


if __name__ == "__main__":
    raise SystemExit("pytest-only contract test")
