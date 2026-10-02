#!/usr/bin/env bash
# Shared bounded retry helpers for GitHub Actions network operations.
# Never waits forever: each attempt has a finite timeout and the retry count is bounded.
set -euo pipefail

ci_gh() {
  local attempts="${CI_GH_ATTEMPTS:-5}"
  local timeout_s="${CI_GH_TIMEOUT_SECONDS:-30}"
  local backoff_s="${CI_GH_BACKOFF_SECONDS:-3}"
  local attempt out errfile
  errfile="$(mktemp)"
  for attempt in $(seq 1 "$attempts"); do
    if out="$(timeout --signal=TERM --kill-after=5s "${timeout_s}s" gh "$@" 2>"$errfile")"; then
      printf '%s\n' "$out"
      rm -f "$errfile"
      return 0
    fi
    echo "WARN: gh command failed (attempt ${attempt}/${attempts}): gh $*" >&2
    if [ "$attempt" -lt "$attempts" ]; then
      sleep "$((backoff_s * attempt))"
    fi
  done
  cat "$errfile" >&2
  rm -f "$errfile"
  return 1
}

ci_gh_api_get() {
  local endpoint="$1"
  shift || true
  ci_gh api "$endpoint" "$@"
}

ci_git_fetch() {
  ci_git_fetch_cwd "" "$@"
}

ci_git_fetch_cwd() {
  local cwd="$1"
  shift || true
  local attempts="${CI_GIT_FETCH_ATTEMPTS:-5}"
  local timeout_s="${CI_GIT_FETCH_TIMEOUT_SECONDS:-60}"
  local backoff_s="${CI_GIT_FETCH_BACKOFF_SECONDS:-5}"
  local attempt
  local -a cmd
  if [ -n "$cwd" ]; then
    cmd=(git -C "$cwd" fetch "$@")
  else
    cmd=(git fetch "$@")
  fi
  for attempt in $(seq 1 "$attempts"); do
    if timeout --signal=TERM --kill-after=10s "${timeout_s}s" "${cmd[@]}"; then
      return 0
    fi
    echo "WARN: git fetch failed (attempt ${attempt}/${attempts})" >&2
    if [ "$attempt" -lt "$attempts" ]; then
      sleep "$((backoff_s * attempt))"
    fi
  done
  return 1
}
