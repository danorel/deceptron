"""Mine GitHub repositories for adversarial training data.

Usage:
    python -m deceptron.miner.query --lang python --limit 50 --out data/deceptron/mining/repos.jsonl
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from deceptron.miner.github import GitHubAPIError, GitHubClient, SearchExhausted
from deceptron.miner.schema import Repo

OPEN_LICENSES = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "GPL-2.0",
    "GPL-3.0",
    "ISC",
}
SUPPORTED_LANGS = ("python",)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine GitHub repos for deceptron")
    parser.add_argument(
        "--lang",
        required=True,
        choices=SUPPORTED_LANGS,
        help="Programming language to search",
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Maximum number of repos to emit",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output .jsonl file path",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="GitHub token (overrides GITHUB_TOKEN env var)",
    )
    return parser.parse_args()


def _load_seen_ids(path: str) -> set[int]:
    """Return repo IDs already present in the output file."""
    p = Path(path)
    if not p.exists():
        return set()
    seen = set()
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return seen


def main() -> None:
    load_dotenv()
    args = _parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Error: GITHUB_TOKEN is not set. Pass --token or set the env var.")

    client = GitHubClient(token)
    three_months_ago = datetime.now(UTC) - timedelta(days=90)

    seen_ids = _load_seen_ids(args.out)
    already = len(seen_ids)
    if already:
        print(f"Resuming: {already} repos already in {args.out}, need {args.limit - already} more.")

    collected = 0
    page = 1

    try:
        with open(args.out, "a", encoding="utf-8") as out_file:
            while collected + already < args.limit:
                try:
                    items = client.search_repos(args.lang, page)
                except SearchExhausted:
                    print("Search results exhausted.")
                    break

                if not items:
                    break

                for raw in items:
                    if collected + already >= args.limit:
                        break

                    # Skip already collected repos
                    if raw["id"] in seen_ids:
                        continue

                    # Cheap filters first — skip before any extra API calls
                    license_info = raw.get("license") or {}
                    if license_info.get("spdx_id") not in OPEN_LICENSES:
                        continue

                    stars = raw["stargazers_count"]
                    if not (100 <= stars <= 2_000):
                        continue

                    full_name = raw["full_name"]
                    default_branch = raw["default_branch"]

                    # Expensive filters — hit the API (cheapest first)
                    last_commit, head_sha = client.get_default_branch_head(full_name, default_branch)
                    if last_commit is None or last_commit < three_months_ago:
                        continue

                    try:
                        contrib = client.get_contributors_count(full_name)
                    except GitHubAPIError as e:
                        print(f"  skip {full_name}: contributors check failed ({e})")
                        continue
                    if not (3 <= contrib <= 30):
                        continue

                    if not client.check_ci_passing(full_name, default_branch):
                        print(f"  skip {full_name}: CI not passing on {default_branch}")
                        continue

                    try:
                        has_tests = client.check_has_tests(full_name, args.lang)
                    except GitHubAPIError as e:
                        print(f"  skip {full_name}: tests check failed ({e})")
                        continue
                    if not has_tests:
                        continue

                    try:
                        repo = Repo.from_github_response(raw, contrib, has_tests, head_sha or "")
                    except (ValidationError, KeyError) as e:
                        print(f"  skip {full_name}: schema error ({e})")
                        continue

                    out_file.write(repo.model_dump_json() + "\n")
                    out_file.flush()
                    seen_ids.add(repo.id)
                    collected += 1
                    print(f"[{already + collected}/{args.limit}] {full_name} ({repo.stars}★)")

                page += 1

    except KeyboardInterrupt:
        print(f"\nInterrupted. {already + collected}/{args.limit} repos in {args.out}")
        return

    print(f"\nDone. {already + collected}/{args.limit} repos in {args.out}")


if __name__ == "__main__":
    main()
