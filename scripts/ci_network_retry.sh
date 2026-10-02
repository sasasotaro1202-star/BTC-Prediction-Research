#!/usr/bin/env bash
# Shared bounded retry helpers for GitHub Actions network operations.
# Never waits forever: each attempt has a finite timeout and the retry count is bounded.
set -euo pipefail

ci_gh_api_get() {
  local endpoint="$1"
  shift || true
  local attempts="${CI_GH_API_ATTEMPTS:-5}"
  local timeout_s="${CI_GH_API_TIMEOUT_SECONDS:-30}"
  local backoff_s="${CI_GH_API_BACKOFF_SECONDS:-3}"
  local attempt out errfile
  errfile="$(mktemp)"
  trap 'rm -f "$errfile"' RETURN

  for attempt in $(seq 1 "$attempts"); do
    if out="$(timeout --signal=TERM --kill-after=5s "${timeout_s}s" gh api "$endpoint" "$@" 2>"$errfile")"; then
      printf '%s\n' "$out"
      return 0
    fi
    echo "WARN: gh api failed (attempt ${attempt}/${attempts}): $endpoint" >&2
    if [ "$attempt" -lt "$attempts" ]; then
      sleep "$((backoff_s * attempt))"
    fi
  done

  cat "$errfile" >&2 || true
  return 1
}

ci_git_fetch() {
  local attempts="${CI_GIT_FETCH_ATTEMPTS:-5}"
  local timeout_s="${CI_GIT_FETCH_TIMEOUT_SECONDS:-60}"
  local backoff_s="${CI_GIT_FETCH_BACKOFF_SECONDS:-5}"
  local attempt
  for attempt in $(seq 1 "$attempts"); do
    if timeout --signal=TERM --kill-after=10s "${timeout_s}s" git fetch "$@"; then
      return 0
    fi
    echo "WARN: git fetch failed (attempt ${attempt}/${attempts})" >&2
    if [ "$attempt" -lt "$attempts" ]; then
      sleep "$((backoff_s * attempt))"
    fi
  done
  return 1
}
