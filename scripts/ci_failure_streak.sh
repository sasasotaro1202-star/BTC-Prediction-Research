#!/usr/bin/env bash
set -euo pipefail

ci_failure_streak_from_stdin() {
  jq -r '[.workflow_runs[]? | select(.status=="completed") | {conclusion,updated_at}]
    | sort_by(.updated_at)
    | reverse
    | reduce .[] as $r ({count:0,done:false};
        if .done then .
        elif $r.conclusion == "success" then .done=true
        elif ($r.conclusion == "failure" or $r.conclusion == "cancelled" or $r.conclusion == "timed_out" or $r.conclusion == "action_required" or $r.conclusion == "stale") then .count += 1
        else .done=true end)
    | .count'
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  ci_failure_streak_from_stdin
fi
