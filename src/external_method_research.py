"""Research-only controller for the external OSS method queue.

This module deliberately stops at source verification / method readiness.
It does not install arbitrary external code, train external models, alter
Production, or treat external benchmark results as BTC evidence.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

QUEUE_PATH = Path("data/external_research_method_queue.json")
RUNTIME_PATH = Path("data/external_research_runtime.json")
RESULTS_DIR = Path("data/external_research_results")
TERMINAL_STATUSES = {"SOURCE_VERIFIED", "HOLD", "SKIPPED"}

_PRIORITY_GROUPS = (
    "immediate_local_reproduction",
    "predictive_method_second_wave",
    "research_infrastructure_next",
    "external_information_next",
    "frontier_only",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid_json:{path}:{exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        api_base: str = "https://api.github.com",
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.api_base = api_base.rstrip("/")
        self._opener = opener or urllib.request.urlopen

    def get(self, path: str) -> dict[str, Any] | None:
        url = self.api_base + path
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "BTC-Prediction-Research/external-method-research",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        last_error: Exception | None = None
        for attempt in range(1, 5):
            request = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with self._opener(request, timeout=30) as response:
                    body = response.read()
                    return json.loads(body.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"github_http_{exc.code}:{path}") from exc
                last_error = exc
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = exc

            if attempt < 4:
                time.sleep(min(2 ** (attempt - 1), 8))

        raise RuntimeError(f"github_request_failed:{path}:{last_error}")


def candidate_order(queue: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    priority_gate = queue.get("priority_gate", {})
    if isinstance(priority_gate, dict):
        for group in _PRIORITY_GROUPS:
            items = priority_gate.get(group, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    repo = str(item.get("repository", "")).strip()
                    _append_unique(ordered, repo)

    candidates = queue.get("candidates", [])
    if isinstance(candidates, list):
        sortable = [
            item
            for item in candidates
            if isinstance(item, dict) and str(item.get("repository", "")).strip()
        ]
        sortable.sort(key=lambda item: int(item.get("queue_rank", 10**9)))
        for item in sortable:
            _append_unique(ordered, str(item["repository"]).strip())

    return ordered


def _candidate_map(queue: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = queue.get("candidates", [])
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, dict):
            continue
        repo = str(item.get("repository", "")).strip()
        if repo:
            result[repo] = item
    return result


def _runtime_results(runtime: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = runtime.get("results", {})
    if isinstance(raw, dict):
        return {
            str(repo): value
            for repo, value in raw.items()
            if isinstance(value, dict)
        }
    return {}


def choose_next_candidate(
    queue: dict[str, Any],
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    runtime = runtime or {}
    processed = {
        repo
        for repo, record in _runtime_results(runtime).items()
        if str(record.get("status", "")) in TERMINAL_STATUSES
    }
    candidates = _candidate_map(queue)
    for repo in candidate_order(queue):
        if repo in processed:
            continue
        candidate = candidates.get(repo)
        if candidate is None:
            continue
        return candidate
    return None


def _license_state(license_obj: Any) -> tuple[str, str]:
    if not isinstance(license_obj, dict):
        return "UNKNOWN", ""
    spdx = str(license_obj.get("spdx_id", "")).strip()
    if not spdx or spdx.upper() in {"NOASSERTION", "OTHER"}:
        return "UNKNOWN", spdx
    return "KNOWN", spdx


def verify_source(repository: str, client: GitHubClient) -> dict[str, Any]:
    repo_obj = client.get(f"/repos/{repository}")
    if repo_obj is None:
        return {
            "status": "HOLD",
            "hold_reason": "REPOSITORY_NOT_FOUND",
            "source_verified": False,
            "source_repo_archived": None,
            "license_state": "UNKNOWN",
            "license_spdx": "",
        }

    archived = bool(repo_obj.get("archived", False))
    license_state, license_spdx = _license_state(repo_obj.get("license"))
    default_branch = str(repo_obj.get("default_branch", "")).strip()
    base: dict[str, Any] = {
        "source_verified": False,
        "source_repo_archived": archived,
        "repository_html_url": repo_obj.get("html_url"),
        "default_branch": default_branch,
        "license_state": license_state,
        "license_spdx": license_spdx,
        "source_updated_at": repo_obj.get("updated_at"),
        "source_pushed_at": repo_obj.get("pushed_at"),
        "source_stars": repo_obj.get("stargazers_count"),
    }

    if archived:
        return {**base, "status": "HOLD", "hold_reason": "REPOSITORY_ARCHIVED"}

    if license_state != "KNOWN":
        return {**base, "status": "HOLD", "hold_reason": "LICENSE_UNVERIFIED"}

    if not default_branch:
        return {
            **base,
            "status": "HOLD",
            "hold_reason": "DEFAULT_BRANCH_UNVERIFIED",
        }

    encoded_branch = urllib.parse.quote(default_branch, safe="")
    commit = client.get(f"/repos/{repository}/commits/{encoded_branch}")
    if not isinstance(commit, dict) or not commit.get("sha"):
        return {
            **base,
            "status": "HOLD",
            "hold_reason": "DEFAULT_BRANCH_COMMIT_UNVERIFIED",
        }

    readme = client.get(f"/repos/{repository}/readme")
    readme_sha = readme.get("sha") if isinstance(readme, dict) else None
    return {
        **base,
        "status": "SOURCE_VERIFIED",
        "source_verified": True,
        "source_commit_sha": commit.get("sha"),
        "source_commit_url": commit.get("html_url"),
        "readme_available": isinstance(readme, dict),
        "readme_sha": readme_sha,
        "hold_reason": "",
    }


def _result_path(repository: str) -> Path:
    safe = repository.replace("/", "__").replace("\\", "__")
    return RESULTS_DIR / f"{safe}.json"


def process_one(
    root: Path,
    client: GitHubClient,
    analysis_sha: str,
    run_id: str,
) -> dict[str, Any]:
    queue = load_json(root / QUEUE_PATH)
    if not isinstance(queue, dict):
        raise RuntimeError("external_method_queue_missing_or_invalid")

    runtime = load_json(root / RUNTIME_PATH, default={})
    if not isinstance(runtime, dict):
        raise RuntimeError("external_method_runtime_invalid")

    candidate = choose_next_candidate(queue, runtime)
    if candidate is None:
        runtime.update(
            {
                "schema_version": 1,
                "research_only": True,
                "production_changed": False,
                "updated_at_utc": _utc_now(),
                "queue_exhausted": True,
                "queue_head_at_execution": analysis_sha,
            }
        )
        write_json(root / RUNTIME_PATH, runtime)
        return {
            "status": "EXHAUSTED",
            "research_only": True,
            "production_changed": False,
            "queue_exhausted": True,
        }

    repository = str(candidate["repository"]).strip()
    verification = verify_source(repository, client)
    status = str(verification["status"])
    next_gate = str(candidate.get("next_gate", "UNSPECIFIED"))
    processed_at = _utc_now()

    result = {
        "schema_version": 1,
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "external_performance_transfer_allowed": False,
        "repository": repository,
        "queue_rank": candidate.get("queue_rank"),
        "mechanism": candidate.get("mechanism"),
        "method_abstract": candidate.get("method_abstract"),
        "btc_relevance": candidate.get("btc_relevance"),
        "queue_next_gate": next_gate,
        "status": status,
        "processed_at_utc": processed_at,
        "analysis_git_sha": analysis_sha,
        "workflow_run_id": str(run_id),
        "source_verification": verification,
        "next_action": (
            "PROCEED_TO_LOCAL_GATE_RESEARCH_ONLY"
            if status == "SOURCE_VERIFIED"
            else "HOLD_UNTIL_SOURCE_RISK_RESOLVED"
        ),
    }

    results = _runtime_results(runtime)
    results[repository] = {
        "repository": repository,
        "status": status,
        "queue_rank": candidate.get("queue_rank"),
        "queue_next_gate": next_gate,
        "processed_at_utc": processed_at,
        "analysis_git_sha": analysis_sha,
        "workflow_run_id": str(run_id),
        "source_commit_sha": verification.get("source_commit_sha"),
        "license_spdx": verification.get("license_spdx", ""),
        "hold_reason": verification.get("hold_reason", ""),
    }
    runtime.update(
        {
            "schema_version": 1,
            "research_only": True,
            "production_changed": False,
            "updated_at_utc": processed_at,
            "queue_head_at_execution": analysis_sha,
            "queue_exhausted": False,
            "results": dict(sorted(results.items())),
        }
    )

    write_json(root / _result_path(repository), result)
    write_json(root / RUNTIME_PATH, runtime)
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    result = process_one(
        root=root,
        client=GitHubClient(),
        analysis_sha=os.environ.get("GITHUB_SHA", "LOCAL_UNPINNED"),
        run_id=os.environ.get("GITHUB_RUN_ID", "local"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
