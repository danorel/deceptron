#!/usr/bin/env bash
# SessionEnd hook: run the test suite and warn if the working tree was left
# dirty. It does NOT commit — commits are made during wrap-up as a series of
# logically-grouped commits (see CLAUDE.md "Session recap").
set -uo pipefail

root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}"
[ -z "$root" ] && exit 0
cd "$root" || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

# run pytest only if it is actually installed; tolerate "no tests collected" (5)
tests="none"
if command -v uv >/dev/null 2>&1 && uv run --quiet python -c "import pytest" >/dev/null 2>&1; then
  uv run --quiet pytest -q >/dev/null 2>&1
  case $? in
    0) tests="pass" ;;
    5) tests="none" ;;
    *) tests="FAIL" ;;
  esac
fi

dirty=$(git status --porcelain)
if [ -n "$dirty" ]; then
  n=$(printf '%s\n' "$dirty" | grep -c .)
  printf '{"systemMessage": "Session end — tests: %s. %s uncommitted change(s): commit them as logically-grouped commits before leaving."}\n' \
    "$tests" "$n"
else
  printf '{"systemMessage": "Session end — tests: %s. Tree clean."}\n' "$tests"
fi
exit 0
