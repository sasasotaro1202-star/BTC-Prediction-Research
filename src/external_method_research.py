"""Research-only controller for external OSS methods."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

QUEUE_PATH = Path("data/external_research_method_queue.json")
RUNTIME_PATH = Path("data/external_research_runtime.json")
RESULTS_DIR = Path("data/external_research_results")
TERMINAL_STATUSES = {"LOCAL_GATE_READY", "HOLD", "SKIPPED", "SHADOW_MATURED", "REJECTED", "FAILED"}
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


class JsonClient:
    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN", "")

    def get(self, url: str) -> dict[str, Any] | None:
        headers = {
            "Accept": "application/json",
            "User-Agent": "BTC-Prediction-Research/external-method-research",
        }
        if self.token and url.startswith("https://api.github.com/"):
            headers["Authorization"] = f"Bearer {self.token}"
            headers["X-GitHub-Api-Version"] = "2022-11-28"
        last: Exception | None = None
        for attempt in range(1, 5):
            req = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"http_{exc.code}:{url}") from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last = exc
            if attempt < 4:
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"request_failed:{url}:{last}")


def candidate_order(queue: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    gate = queue.get("priority_gate", {})
    if isinstance(gate, dict):
        for group in _PRIORITY_GROUPS:
            items = gate.get(group, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        repo = str(item.get("repository", "")).strip()
                        if repo and repo not in ordered:
                            ordered.append(repo)
    items = queue.get("candidates", [])
    if isinstance(items, list):
        ranked = [
            x for x in items
            if isinstance(x, dict) and str(x.get("repository", "")).strip()
        ]
        ranked.sort(key=lambda x: int(x.get("queue_rank", 10**9)))
        for item in ranked:
            repo = str(item["repository"]).strip()
            if repo not in ordered:
                ordered.append(repo)
    return ordered


def _candidate_map(queue: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in queue.get("candidates", []):
        if isinstance(item, dict):
            repo = str(item.get("repository", "")).strip()
            if repo:
                out[repo] = item
    return out


def _runtime_results(runtime: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = runtime.get("results", {})
    return {
        str(repo): record
        for repo, record in raw.items()
        if isinstance(record, dict)
    } if isinstance(raw, dict) else {}


def choose_next_candidate(queue: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, Any] | None:
    runtime = runtime or {}
    processed = {
        repo
        for repo, record in _runtime_results(runtime).items()
        if record.get("status") in TERMINAL_STATUSES
    }
    candidates = _candidate_map(queue)
    for repo in candidate_order(queue):
        if repo not in processed and repo in candidates:
            return candidates[repo]
    return None


def _license_state(obj: Any) -> tuple[str, str]:
    if not isinstance(obj, dict):
        return "UNKNOWN", ""
    spdx = str(obj.get("spdx_id", "")).strip()
    if not spdx or spdx.upper() in {"NOASSERTION", "OTHER"}:
        return "UNKNOWN", spdx
    return "KNOWN", spdx


def verify_github_repo(repository: str, client: JsonClient) -> dict[str, Any]:
    obj = client.get(f"https://api.github.com/repos/{repository}")
    if obj is None:
        return {"status": "HOLD", "hold_reason": "REPOSITORY_NOT_FOUND", "source_verified": False}
    archived = bool(obj.get("archived", False))
    license_state, license_spdx = _license_state(obj.get("license"))
    default_branch = str(obj.get("default_branch", "")).strip()
    base = {
        "repository": repository,
        "repository_html_url": obj.get("html_url"),
        "source_repo_archived": archived,
        "default_branch": default_branch,
        "license_state": license_state,
        "license_spdx": license_spdx,
        "source_updated_at": obj.get("updated_at"),
        "source_pushed_at": obj.get("pushed_at"),
        "source_stars": obj.get("stargazers_count"),
        "source_verified": False,
    }
    if archived:
        return {**base, "status": "HOLD", "hold_reason": "REPOSITORY_ARCHIVED"}
    if license_state != "KNOWN":
        return {**base, "status": "HOLD", "hold_reason": "LICENSE_UNVERIFIED"}
    if not default_branch:
        return {**base, "status": "HOLD", "hold_reason": "DEFAULT_BRANCH_UNVERIFIED"}
    commit = client.get(
        f"https://api.github.com/repos/{repository}/commits/{urllib.parse.quote(default_branch, safe='')}"
    )
    if not isinstance(commit, dict) or not commit.get("sha"):
        return {**base, "status": "HOLD", "hold_reason": "DEFAULT_BRANCH_COMMIT_UNVERIFIED"}
    return {
        **base,
        "status": "SOURCE_VERIFIED",
        "source_verified": True,
        "source_commit_sha": commit.get("sha"),
        "source_commit_url": commit.get("html_url"),
    }


def verify_hf_model(model_id: str, client: JsonClient) -> dict[str, Any]:
    obj = client.get(
        "https://huggingface.co/api/models/"
        + urllib.parse.quote(model_id, safe="/")
    )
    if obj is None:
        return {
            "model_id": model_id,
            "status": "HOLD",
            "hold_reason": "MODEL_REPOSITORY_NOT_FOUND",
            "source_verified": False,
        }
    card = obj.get("cardData") if isinstance(obj.get("cardData"), dict) else {}
    license_name = str(card.get("license") or obj.get("license") or "").strip()
    if not license_name:
        return {
            "model_id": model_id,
            "status": "HOLD",
            "hold_reason": "MODEL_LICENSE_UNVERIFIED",
            "source_verified": False,
        }
    return {
        "model_id": model_id,
        "status": "SOURCE_VERIFIED",
        "source_verified": True,
        "license": license_name,
        "sha": obj.get("sha"),
        "last_modified": obj.get("lastModified"),
    }


def verify_source_contracts(candidate: dict[str, Any], client: JsonClient) -> dict[str, Any]:
    contracts = candidate.get("source_contracts", [])
    if not isinstance(contracts, list):
        contracts = []
    checked: list[dict[str, Any]] = []
    failures: list[str] = []
    for contract in contracts:
        if not isinstance(contract, dict):
            failures.append("INVALID_SOURCE_CONTRACT")
            continue
        kind = str(contract.get("kind", "")).strip()
        ident = str(contract.get("id", "")).strip()
        if kind == "github":
            item = verify_github_repo(ident, client)
        elif kind == "huggingface_model":
            item = verify_hf_model(ident, client)
        else:
            item = {
                "id": ident,
                "status": "HOLD",
                "source_verified": False,
                "hold_reason": "UNKNOWN_SOURCE_CONTRACT_KIND",
            }
        checked.append(item)
        if item.get("status") != "SOURCE_VERIFIED":
            failures.append(f"{kind}:{ident}:{item.get('hold_reason', 'SOURCE_UNVERIFIED')}")
    return {"checked": checked, "all_verified": not failures, "failures": failures}


def process_one(root: Path, client: JsonClient, analysis_sha: str, run_id: str) -> dict[str, Any]:
    queue = load_json(root / QUEUE_PATH)
    runtime = load_json(root / RUNTIME_PATH, default={})
    if not isinstance(queue, dict) or not isinstance(runtime, dict):
        raise RuntimeError("external_method_queue_or_runtime_invalid")

    candidate = choose_next_candidate(queue, runtime)
    if candidate is None:
        runtime.update({
            "schema_version": 3,
            "research_only": True,
            "production_changed": False,
            "updated_at_utc": _utc_now(),
            "queue_exhausted": True,
            "queue_head_at_execution": analysis_sha,
        })
        write_json(root / RUNTIME_PATH, runtime)
        return {"status": "EXHAUSTED", "research_only": True, "production_changed": False, "queue_exhausted": True}

    repository = str(candidate["repository"]).strip()
    primary = verify_github_repo(repository, client)
    if primary.get("status") != "SOURCE_VERIFIED":
        status = "HOLD"
        contracts = {"checked": [], "all_verified": False, "failures": [str(primary.get("hold_reason", "SOURCE_UNVERIFIED"))]}
    else:
        contracts = verify_source_contracts(candidate, client)
        status = "LOCAL_GATE_READY" if contracts["all_verified"] else "HOLD"

    processed_at = _utc_now()
    next_action = (
        "DISPATCH_CANDIDATE_SPECIFIC_NEXT_GATE"
        if status == "LOCAL_GATE_READY"
        else "HOLD_UNTIL_SOURCE_RISK_RESOLVED"
    )
    result = {
        "schema_version": 3,
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "external_performance_transfer_allowed": False,
        "repository": repository,
        "queue_rank": candidate.get("queue_rank"),
        "mechanism": candidate.get("mechanism"),
        "method_abstract": candidate.get("method_abstract"),
        "btc_relevance": candidate.get("btc_relevance"),
        "queue_next_gate": candidate.get("next_gate"),
        "status": status,
        "processed_at_utc": processed_at,
        "analysis_git_sha": analysis_sha,
        "workflow_run_id": str(run_id),
        "primary_source_verification": primary,
        "source_contract_verification": contracts,
        "next_action": next_action,
    }

    results = _runtime_results(runtime)
    results[repository] = {
        "repository": repository,
        "status": status,
        "queue_rank": candidate.get("queue_rank"),
        "queue_next_gate": candidate.get("next_gate"),
        "processed_at_utc": processed_at,
        "analysis_git_sha": analysis_sha,
        "workflow_run_id": str(run_id),
        "source_commit_sha": primary.get("source_commit_sha"),
        "license_spdx": primary.get("license_spdx", ""),
        "source_contracts": contracts,
        "next_action": next_action,
    }
    runtime.update({
        "schema_version": 3,
        "research_only": True,
        "production_changed": False,
        "updated_at_utc": processed_at,
        "queue_head_at_execution": analysis_sha,
        "queue_exhausted": False,
        "results": dict(sorted(results.items())),
    })
    safe = repository.replace("/", "__").replace("\\", "__")
    write_json(root / (RESULTS_DIR / f"{safe}.json"), result)
    write_json(root / RUNTIME_PATH, runtime)
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    result = process_one(
        root,
        JsonClient(),
        os.environ.get("GITHUB_SHA", "LOCAL_UNPINNED"),
        os.environ.get("GITHUB_RUN_ID", "local"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
