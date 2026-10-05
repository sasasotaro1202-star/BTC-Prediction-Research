import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'ci_failure_streak.sh'


def _run(payload):
    proc = subprocess.run(
        ['bash', str(SCRIPT)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    return int(proc.stdout.strip())


def test_failure_streak_counts_only_failures_since_latest_success():
    payload = {'workflow_runs': [
        {'status': 'completed', 'conclusion': 'failure', 'updated_at': '2026-10-05T10:00:00Z'},
        {'status': 'completed', 'conclusion': 'cancelled', 'updated_at': '2026-10-05T09:55:00Z'},
        {'status': 'completed', 'conclusion': 'success', 'updated_at': '2026-10-05T09:50:00Z'},
        {'status': 'completed', 'conclusion': 'failure', 'updated_at': '2026-10-05T09:45:00Z'},
    ]}
    assert _run(payload) == 2


def test_neutral_terminal_result_stops_failure_streak():
    payload = {'workflow_runs': [
        {'status': 'completed', 'conclusion': 'failure', 'updated_at': '2026-10-05T10:00:00Z'},
        {'status': 'completed', 'conclusion': 'skipped', 'updated_at': '2026-10-05T09:55:00Z'},
        {'status': 'completed', 'conclusion': 'failure', 'updated_at': '2026-10-05T09:50:00Z'},
    ]}
    assert _run(payload) == 1
