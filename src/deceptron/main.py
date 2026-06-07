"""Run the deceptron pipeline end to end (see README.md / docs/plan.md).

Each stage shells out to `python -m deceptron.<module>` using the data/
layout's default paths, in pipeline order:

    mine -> verify-repos -> tasks -> agent -> redteam
         -> verify-trajectories -> monitor -> eval

Usage:
    python -m deceptron.main                       # run the full pipeline
    python -m deceptron.main --stages tasks agent  # run a subset, in pipeline order
    python -m deceptron.main --list                # list stage names

Per-stage flags (--limit, --force, --no-docker, ...) aren't exposed here —
run that stage's module directly when you need them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from deceptron.miner.main import DEFAULT_OUT as MINED_REPOS
from deceptron.verifier.main import DEFAULT_VERIFY_REPOS_OUT as VERIFIED_REPOS

PY = sys.executable

PIPELINE: list[tuple[str, list[list[str]]]] = [
    ("mine", [
        [PY, "-m", "deceptron.miner.main",
         "--lang", "python", "--limit", "50", "--out", MINED_REPOS],
    ]),
    ("verify-repos", [
        [PY, "-m", "deceptron.verifier.main",
         "--repos", MINED_REPOS,
         "--out", VERIFIED_REPOS],
    ]),
    ("tasks", [
        [PY, "-m", "deceptron.tasks.generate",
         "--repos", "data/deceptron/mining/repos_verified_clean.jsonl",
         "--out", "data/deceptron/tasks/tasks.jsonl"],
        [PY, "-m", "deceptron.tasks.validate",
         "--tasks", "data/deceptron/tasks/tasks.jsonl",
         "--repos", "data/deceptron/mining/repos_verified_clean.jsonl"],
    ]),
    ("agent", [
        [PY, "-m", "deceptron.agent.main",
         "--tasks", "data/deceptron/tasks/tasks.jsonl",
         "--out", "data/deceptron/trajectories/baseline.jsonl"],
    ]),
    ("redteam", [
        [PY, "-m", "deceptron.redteam.main",
         "--tasks", "data/deceptron/tasks/tasks.jsonl",
         "--out", "data/deceptron/trajectories/scheming.jsonl"],
    ]),
    ("verify-trajectories", [
        [PY, "-m", "deceptron.verifier.main",
         "--trajectories", "data/deceptron/trajectories/baseline.jsonl",
         "--tasks", "data/deceptron/tasks/tasks.jsonl",
         "--verified-repos", "data/deceptron/mining/repos_verified_clean.jsonl"],
        [PY, "-m", "deceptron.verifier.main",
         "--trajectories", "data/deceptron/trajectories/scheming.jsonl",
         "--tasks", "data/deceptron/tasks/tasks.jsonl",
         "--verified-repos", "data/deceptron/mining/repos_verified_clean.jsonl"],
    ]),
    ("monitor", [
        [PY, "-m", "deceptron.monitor.prompted.main",
         "--trajectories", "data/deceptron/trajectories/baseline.jsonl", "data/deceptron/trajectories/scheming.jsonl",
         "--out", "data/deceptron/scores/prompted.jsonl"],
    ]),
    ("eval", [
        [PY, "-m", "deceptron.eval.main",
         "--scores", "data/deceptron/scores/prompted.jsonl",
         "--out", "data/deceptron/results/prompted/"],
    ]),
]

STAGE_NAMES = [name for name, _ in PIPELINE]


def run_stage(name: str, commands: list[list[str]]) -> None:
    for cmd in commands:
        print(f"[{name}] $ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deceptron pipeline end to end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stages", nargs="+", choices=STAGE_NAMES, metavar="STAGE",
        help=f"run only these stages, in pipeline order (default: all). choices: {', '.join(STAGE_NAMES)}",
    )
    parser.add_argument("--list", action="store_true", help="list stage names and exit")
    args = parser.parse_args()

    if args.list:
        for name in STAGE_NAMES:
            print(name)
        return

    selected = set(args.stages) if args.stages else set(STAGE_NAMES)
    for name, commands in PIPELINE:
        if name in selected:
            run_stage(name, commands)


if __name__ == "__main__":
    main()
