from pathlib import Path

from src import historical_research as hr


def test_oi_history_window_stays_inside_recent_public_availability():
    assert hr.OI_HISTORY_DAYS == 26
    assert hr.ARCHIVE_SAFETY_DAYS == 3
    source = Path("src/historical_research.py").read_text(encoding="utf-8")
    assert "start=max(start,end-timedelta(days=OI_HISTORY_DAYS))" in source
    assert "end-timedelta(days=30)" not in source
