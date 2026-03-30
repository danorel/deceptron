"""Verify mined repos by running their test suites in Docker.

Usage:
    python -m verifier.run --repos data/repos.jsonl --out data/repos_verified.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from miner.schema import Repo
from verifier.docker import verify_repo
from verifier.schema import VerifiedRepo


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify repos via Docker test runs")
    parser.add_argument("--repos", required=True, help="Input repos.jsonl from miner")
    parser.add_argument("--out", required=True, help="Output repos_verified.jsonl")
    return parser.parse_args()


def _load_seen(path: str) -> set[int]:
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


def main() -> None:
    args = _parse_args()

    repos_path = Path(args.repos)
    if not repos_path.exists():
        sys.exit(f"Error: {args.repos} not found. Run miner first.")

    repos = []
    with repos_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                repos.append(Repo.model_validate_json(line))

    seen_ids = _load_seen(args.out)
    if seen_ids:
        print(f"Resuming: {len(seen_ids)} repos already verified.")

    pending = [r for r in repos if r.id not in seen_ids]
    print(f"Verifying {len(pending)} repos (Docker required)...\n")

    with open(args.out, "a", encoding="utf-8") as out_file:
        for i, repo in enumerate(pending, 1):
            print(f"[{i}/{len(pending)}] {repo.full_name} — cloning + testing...")
            result = verify_repo(repo.clone_url, repo.default_branch)

            verified = VerifiedRepo(**repo.model_dump(), **result)
            out_file.write(verified.model_dump_json() + "\n")
            out_file.flush()

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
            if verified.deselect_args:
                print(f"  deselected {len(verified.deselect_args)} failing tests")

    total = len(seen_ids) + len(pending)
    passed = sum(1 for r in [json.loads(l) for l in Path(args.out).read_text().splitlines() if l]
                 if r.get("test_verified"))
    print(f"\nDone. {passed}/{total} repos verified.")


if __name__ == "__main__":
    main()
