from pathlib import Path
import re


DASHBOARD = Path(__file__).resolve().parents[1] / "docs" / "index.html"


def test_dashboard_is_mobile_first_and_covers_all_horizons():
    text = DASHBOARD.read_text(encoding="utf-8")
    assert 'let payload=null,currentH="5m"' in text
    assert 'const HORIZONS=["5m","10m","15m","30m","1h","3h","6h","12h","24h"]' in text
    assert '["7d",604800000]' in text
    assert 'setInterval(()=>load(),60000);' in text


def test_dashboard_refresh_rebuilds_controls_without_duplication():
    text = DASHBOARD.read_text(encoding="utf-8")
    assert 'h.innerHTML="";' in text
    assert 'rg.innerHTML="";' in text
    assert 'currentRange="24h"' in text
    assert 'RANGES=[["1h",3600000],["6h",21600000],["24h",86400000]' in text
    assert 'z.classList.toggle("active",z.dataset.r===currentRange)' in text


def test_dashboard_never_bridges_missing_series_values():
    text = DASHBOARD.read_text(encoding="utf-8")
    assert 'if(y==null||!Number.isFinite(y)){valid=0;return;}' in text


def test_dashboard_fetches_dynamic_state_without_browser_cache():
    text = DASHBOARD.read_text(encoding="utf-8")
    assert re.search(r'forecast_trajectory\.json\?ts="\+Date\.now\(\)', text)
    assert re.search(r'live_cycle_status\.json\?ts="\+Date\.now\(\)', text)
    assert re.search(r'pit_oos_audit\.json\?ts="\+Date\.now\(\)', text)



def test_dashboard_renders_all_horizon_overview_in_same_range_and_metric():
    text = DASHBOARD.read_text(encoding="utf-8")
    assert 'id="overviewGrid" class="overviewGrid"' in text
    assert 'function overviewRangeRows(h)' in text
    assert 'function overviewMetricSpec(points)' in text
    assert 'function drawOverview()' in text
    assert 'drawChart(rows);drawOverview();updateLegend();updateSummary()' in text
    for horizon in ("5m", "10m", "15m", "30m", "1h", "3h", "6h", "12h", "24h"):
        assert horizon in text


def test_dashboard_overview_never_invents_missing_points():
    text = DASHBOARD.read_text(encoding="utf-8")
    assert 'if(!last){' in text
    assert '最初の実運用サンプル待ち' in text
    assert '未取得・未決済は線を補完しません' in text
