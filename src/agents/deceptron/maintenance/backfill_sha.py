"""Backfill missing SHA fields into repos.jsonl or repos_verified.jsonl via the
GitHub API — see `main.py` for the CLI."""

import json
import os
import sys
from pathlib import Path

from agents.deceptron.miner.github import GitHubClient


def run(file: str, field: str | None = None) -> None:
    path = Path(file)
    if not path.exists():
        sys.exit(f"{path} not found")

    field = field or ("verified_sha" if "verified" in path.name else "head_sha")

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
