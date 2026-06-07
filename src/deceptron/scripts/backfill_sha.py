"""Backfill missing SHA fields into repos.jsonl or repos_verified.jsonl.

Usage:
    python -m deceptron.scripts.backfill_sha --file data/mining/repos.jsonl
    python -m deceptron.scripts.backfill_sha --file data/mining/repos_verified.jsonl --field verified_sha
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from deceptron.miner.github import GitHubClient


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/mining/repos.jsonl")
    parser.add_argument(
        "--field",
        default=None,
        help="Field name to backfill (default: head_sha for repos.jsonl, verified_sha for repos_verified.jsonl)",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"{path} not found")

    field = args.field or ("verified_sha" if "verified" in path.name else "head_sha")

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN not set")

    client = GitHubClient(token)

    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    updated = 0
    for row in rows:
        if row.get(field):
            continue
        _, sha = client.get_default_branch_head(row["full_name"], row["default_branch"])
        row[field] = sha or ""
        updated += 1
        print(f"  {row['full_name']} → {row[field][:12]}")

    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nDone. {updated}/{len(rows)} records updated in '{field}'.")


if __name__ == "__main__":
    main()
