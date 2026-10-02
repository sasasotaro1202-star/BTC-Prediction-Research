import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "ci_network_retry.sh"


def _run_bash(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
        env=env,
        timeout=20,
    )


def test_ci_gh_api_get_retries_and_forwards_arguments(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    state = tmp_path / "gh_count"
    log = tmp_path / "gh_args.log"

    fake_gh = bindir / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        "n=0; [ -f \"$STATE\" ] && n=$(cat \"$STATE\")\n"
        "n=$((n+1)); printf '%s' \"$n\" > \"$STATE\"\n"
        "printf '%s\\n' \"$*\" >> \"$LOG\"\n"
        "if [ \"$n\" -lt 3 ]; then exit 1; fi\n"
        "printf '%s\\n' '{\"object\":{\"sha\":\"ok\"}}'\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "PATH": f"{bindir}:{env['PATH']}",
        "STATE": str(state),
        "LOG": str(log),
        "CI_GH_API_ATTEMPTS": "3",
        "CI_GH_API_TIMEOUT_SECONDS": "2",
        "CI_GH_API_BACKOFF_SECONDS": "0",
    })
    result = _run_bash(
        f'. "{HELPER}" && ci_gh_api_get "repos/example/git/ref/heads/main" --jq ".object.sha"',
        env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith('{"object":{"sha":"ok"}}')
    assert state.read_text() == "3"
    lines = log.read_text().splitlines()
    assert len(lines) == 3
    assert all("--jq .object.sha" in line for line in lines)
    assert all("repos/example/git/ref/heads/main" in line for line in lines)


def test_ci_git_fetch_retries_and_forwards_arguments(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    state = tmp_path / "git_count"
    log = tmp_path / "git_args.log"

    fake_git = bindir / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" != \"fetch\" ]; then exit 2; fi\n"
        "n=0; [ -f \"$STATE\" ] && n=$(cat \"$STATE\")\n"
        "n=$((n+1)); printf '%s' \"$n\" > \"$STATE\"\n"
        "printf '%s\\n' \"$*\" >> \"$LOG\"\n"
        "if [ \"$n\" -lt 2 ]; then exit 1; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "PATH": f"{bindir}:{env['PATH']}",
        "STATE": str(state),
        "LOG": str(log),
        "CI_GIT_FETCH_ATTEMPTS": "3",
        "CI_GIT_FETCH_TIMEOUT_SECONDS": "2",
        "CI_GIT_FETCH_BACKOFF_SECONDS": "0",
    })
    result = _run_bash(
        f'. "{HELPER}" && ci_git_fetch origin main --depth=1',
        env,
    )

    assert result.returncode == 0, result.stderr
    assert state.read_text() == "2"
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    assert all("fetch origin main --depth=1" in line for line in lines)
