"""One-off maintenance utilities for mining-pipeline data — repos.jsonl
(miner stage 1 output) and repos_verified*.jsonl (verifier stage 2 output).
Not part of `agents.deceptron.main`'s PIPELINE; run directly when needed.

Usage:
    python -m agents.deceptron.maintenance.main backfill-sha --file data/agents/deceptron/mining/repos.jsonl
    python -m agents.deceptron.maintenance.main backfill-sha --file data/agents/deceptron/mining/repos_verified.jsonl --field verified_sha
    python -m agents.deceptron.maintenance.main strip-logs
"""

import argparse

from dotenv import load_dotenv

from agents.deceptron.maintenance.backfill_sha import run as run_backfill_sha
from agents.deceptron.maintenance.strip_logs import DEFAULT_DST, DEFAULT_SRC, run as run_strip_logs


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_backfill = sub.add_parser("backfill-sha", help="Backfill missing SHA fields via the GitHub API")
    p_backfill.add_argument("--file", default="data/agents/deceptron/mining/repos.jsonl")
    p_backfill.add_argument(
        "--field", default=None,
        help="Field to backfill (default: head_sha for repos.jsonl, verified_sha for repos_verified.jsonl)",
    )

    p_strip = sub.add_parser("strip-logs", help="Strip install/test log fields for a git-committable copy")
    p_strip.add_argument("--src", default=DEFAULT_SRC)
    p_strip.add_argument("--dst", default=DEFAULT_DST)

    args = parser.parse_args()

    if args.command == "backfill-sha":
        run_backfill_sha(args.file, args.field)
    elif args.command == "strip-logs":
        run_strip_logs(args.src, args.dst)


if __name__ == "__main__":
    main()
