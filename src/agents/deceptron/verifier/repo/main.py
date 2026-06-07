"""Stage 2 — verify mined repos in Docker (clone + run test suite).

Ground-truth check for whether each repo's tests pass on a clean checkout —
results gate which repos proceed to task generation (stage 3).

Usage:
    python -m agents.deceptron.verifier.repo.main \\
        --repos data/agents/deceptron/mining/repos.jsonl \\
        --out data/agents/deceptron/mining/repos_verified.jsonl
"""

import argparse
import fcntl
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from agents.deceptron.miner.schema import Repo
from agents.deceptron.verifier.repo.schema import VerifiedRepo
from agents.deceptron.verifier.repo.verify import verify_repo

DEFAULT_VERIFY_REPOS_OUT = "data/agents/deceptron/mining/repos_verified.jsonl"


def load_verified_ids(path: str) -> set[int]:
    """Return repo IDs already present in `path` (.jsonl) — used to resume
    both the batch CLI loop and Dagster's per-repo dynamic fan-out."""
    p = Path(path)
    if not p.exists():
        return set()
    seen = set()
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return seen


def _append_jsonl_locked(path: str, line: str) -> None:
    """Append one line to `path`, holding an exclusive flock for the write.

    Verified repo lines can be large (install/test logs), so a plain append
    from multiple concurrent writers (Dagster's parallel verify ops each
    appending to the same `out`) could interleave and corrupt the file —
    flock serializes them without giving up per-repo incremental writes."""
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def verify_and_append_repo(repo: Repo, out: str) -> VerifiedRepo:
    """Clone + test `repo`, lock-append to `out`, print status. Shared by
    the CLI loop and Dagster's per-repo op."""
    print(f"{repo.full_name} — cloning + testing...")
    result = verify_repo(repo.clone_url, repo.default_branch)

    verified = VerifiedRepo(**repo.model_dump(), **result)
    _append_jsonl_locked(out, verified.model_dump_json())

    if verified.test_verified:
        status = "✓ verified"
    elif verified.error == "timeout":
        status = "⏱ timeout"
    else:
        status = "✗ failed"

    detail = (
        f"{verified.tests_passed} passed, {verified.tests_failed} failed"
        if not verified.error
        else f"error: {verified.error}"
    )
    print(f"  {status} — {detail}")
    return verified


def run(repos: str, out: str) -> None:
    """CLI loop: verify each repo in `repos`, resume by id from `out`."""
    repos_path = Path(repos)
    if not repos_path.exists():
        sys.exit(f"Error: {repos} not found. Run miner first.")

    repo_list = []
    with repos_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                repo_list.append(Repo.model_validate_json(line))

    seen_ids = load_verified_ids(out)
    if seen_ids:
        print(f"Resuming: {len(seen_ids)} repos already verified.")

    pending = [r for r in repo_list if r.id not in seen_ids]
    print(f"Verifying {len(pending)} repos (Docker required)...\n")

    for i, repo in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] ", end="")
        verify_and_append_repo(repo, out)

    total = len(seen_ids) + len(pending)
    passed = sum(
        1 for r in [json.loads(l) for l in Path(out).read_text().splitlines() if l]
        if r.get("test_verified")
    )
    print(f"\nDone. {passed}/{total} repos verified.")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", required=True, help="Input repos.jsonl (output of miner)")
    parser.add_argument("--out", default=DEFAULT_VERIFY_REPOS_OUT, help="Output repos_verified.jsonl")
    args = parser.parse_args()

    run(args.repos, args.out)


if __name__ == "__main__":
    main()
